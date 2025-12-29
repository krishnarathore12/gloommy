import shutil
import os
import json
import zipfile
import base64
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import shutil
import os
import json
import zipfile
import base64
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from PIL import Image

from core.rag import RAGEngine, DEVICE
from core.vector_db import SimpleVectorDB

import requests
from io import BytesIO

# --- Configuration ---
UPLOAD_DIR = Path("data/uploads")
DB_PATH = Path("data/vector_db.pkl")
ADAPTER_PATH = "data/vision_adapter.pth" # Ensure this exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Nano Banana Backend")

# Global instances
rag_engine = None
vector_db = None

class GenerateRequest(BaseModel):
    query: str
    api_key: str # Pass Gemini Key per request or use env var

class RetrievalResult(BaseModel):
    image_base64: str
    caption: str
    matches: List[str]
    dense_score: float

class GenerateResponse(BaseModel):
    generated_image_base64: str
    retrieved_context: List[RetrievalResult]

@app.on_event("startup")
async def startup_event():
    global rag_engine, vector_db
    # Load Models
    rag_engine = RAGEngine(adapter_path=ADAPTER_PATH)
    # Load DB
    vector_db = SimpleVectorDB.load(str(DB_PATH))
    print(f"Server started. DB loaded with {len(vector_db.image_ids)} images.")

# --- API 1: Add Image/Caption via Zip ---
@app.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    """
    Upload a zip file containing:
    1. Images (.jpg, .png)
    2. metadata.json (List of objects with 'file_name' and 'caption')
    """
    global vector_db
    
    # 1. Save Zip
    zip_path = UPLOAD_DIR / file.filename
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Extract
    extract_path = UPLOAD_DIR / file.filename.split(".")[0]
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # 3. Read Metadata
    metadata_file = extract_path / "metadata.json"
    if not metadata_file.exists():
        raise HTTPException(status_code=400, detail="metadata.json not found in zip")
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # 4. Process Images (This could be a background task for large files)
    processed_count = 0
    for item in metadata:
        img_name = item.get("file_name")
        caption = item.get("caption")
        img_full_path = extract_path / img_name
        
        if img_full_path.exists():
            # Get CLIP Embedding using the RAG engine's processor
            # Note: We need to access the processor from rag_engine
            # Use the CLIPImageProcessor logic from your script
            try:
                pil_img = Image.open(img_full_path).convert("RGB")
                inputs = rag_engine.processor(images=pil_img, return_tensors="pt").to(DEVICE)
                import torch
                with torch.no_grad():
                     # Use the visual projection from the loaded CLIP model
                    embed = rag_engine.clip_model.get_image_features(**inputs)
                    embed_np = embed.squeeze(0).cpu().numpy()
                
                vector_db.add_image(
                    image_id=img_name,
                    embedding=embed_np,
                    caption=caption,
                    image_path=str(img_full_path)
                )
                processed_count += 1
            except Exception as e:
                print(f"Failed to process {img_name}: {e}")

    # 5. Save DB
    vector_db.save(str(DB_PATH))
    
    return {"status": "success", "processed_images": processed_count}

# --- API 2: Query -> Retrieve -> Generate ---
@app.post("/generate", response_model=GenerateResponse)
async def generate_image(req: GenerateRequest):
    global vector_db, rag_engine
    
    # 1. Load Images from DB Paths
    # We need to map DB indices to actual PIL images for the retrieve function
    db_pil_images = []
    # Optimization: Only load images if we had a massive DB, but for now load all 
    # (or refactor retrieve to take paths, but let's stick to your logic)
    for path in vector_db.image_paths:
        try:
            db_pil_images.append(Image.open(path).convert("RGB"))
        except:
            # Handle missing files gracefully
            db_pil_images.append(Image.new('RGB', (224, 224)))

    # 2. Retrieve (Using logic extracted from Streamlit)
    # Note: You need to implement the .retrieve method in RAGEngine similar to your class
    results = rag_engine.retrieve_pareto(
        query=req.query,
        database_images=db_pil_images,
        database_captions=vector_db.captions,
        database_patches=None # Optim: Compute patches on fly or cache them in DB
    )

    if not results or not results['pareto_indices']:
        raise HTTPException(status_code=404, detail="No relevant images found")

    # 3. Format Context for Gemini
    pareto_indices = results['pareto_indices']
    retrieved_captions = [vector_db.captions[i] for i in pareto_indices]
    
    # Create context string
    style_context = ". ".join(retrieved_captions[:3])
    enhanced_prompt = (
        f"Generate an image of: {req.query}. "
        f"Ensure the style and composition incorporates these elements: {style_context}. "
        f"Photorealistic, high quality."
    )

    # 4. Generate Image
    gen_base64 = rag_engine.generate_gemini(req.api_key, enhanced_prompt)

    if not gen_base64:
        raise HTTPException(status_code=500, detail="Gemini generation failed")

    # 5. Prepare Response
    response_context = []
    for idx in pareto_indices:
        # Convert retrieved image to base64 for frontend display
        img_path = vector_db.image_paths[idx]
        with open(img_path, "rb") as img_f:
            b64_str = base64.b64encode(img_f.read()).decode("utf-8")
        
        # Calculate matches
        satisfied = [results['subqueries'][i] for i in range(len(results['subqueries'])) 
                     if results['sparse_scores'][idx, i] == 1]
        
        dense_score = float(results['dense_scores'][idx].mean())

        response_context.append(RetrievalResult(
            image_base64=b64_str,
            caption=vector_db.captions[idx],
            matches=satisfied,
            dense_score=dense_score
        ))

    return GenerateResponse(
        generated_image_base64=gen_base64,
        retrieved_context=response_context
    )

from core.rag import RAGEngine, DEVICE
from core.vector_db import SimpleVectorDB

# --- Configuration ---
UPLOAD_DIR = Path("data/uploads")
DB_PATH = Path("data/vector_db.pkl")
# Ensure this points to your actual adapter path
ADAPTER_PATH = "data/vision_adapter_epoch_10.pth" 

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Nano Banana Backend")

# Add CORS so your frontend (React/Next.js) can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_engine = None
vector_db = None

# --- Schemas ---
class GenerateRequest(BaseModel):
    query: str
    api_key: str 

class RetrievalResult(BaseModel):
    image_base64: str
    caption: str
    matches: List[str]
    dense_score: float

class GenerateResponse(BaseModel):
    generated_image_base64: str
    retrieved_context: List[RetrievalResult]

class ImageItem(BaseModel):
    id: str
    caption: str
    image_base64: str

class GalleryResponse(BaseModel):
    total: int
    page: int
    limit: int
    images: List[ImageItem]

# --- Lifecycle ---
@app.on_event("startup")
async def startup_event():
    global rag_engine, vector_db
    print("Startup: Loading RAG Engine...")
    rag_engine = RAGEngine(adapter_path=str(ADAPTER_PATH))
    
    print("Startup: Loading Vector DB...")
    vector_db = SimpleVectorDB.load(str(DB_PATH))
    print(f"Server ready. DB loaded with {len(vector_db.image_ids)} images.")

# --- API 1: Add Image/Caption via Zip ---
@app.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    global vector_db
    
    # 1. Save Zip
    zip_path = UPLOAD_DIR / file.filename
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Extract
    extract_path = UPLOAD_DIR / file.filename.split(".")[0]
    if extract_path.exists():
        shutil.rmtree(extract_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # 3. Read Metadata
    metadata_file = extract_path / "metadata.json"
    if not metadata_file.exists():
        # Try finding it in a subfolder (common zip issue)
        found = list(extract_path.rglob("metadata.json"))
        if found:
            metadata_file = found[0]
            # Update base path to where metadata is found
            extract_path = metadata_file.parent 
        else:
            raise HTTPException(status_code=400, detail="metadata.json not found in zip")
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # 4. Process Images
    processed_count = 0
    import torch 
    
    for item in metadata:
        img_name = item.get("file_name")
        caption = item.get("caption")
        img_full_path = extract_path / img_name
        
        if img_full_path.exists():
            try:
                pil_img = Image.open(img_full_path).convert("RGB")
                inputs = rag_engine.processor(images=pil_img, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    embed = rag_engine.clip_model.get_image_features(**inputs)
                    embed_np = embed.squeeze(0).cpu().numpy()
                
                vector_db.add_image(
                    image_id=img_name,
                    embedding=embed_np,
                    caption=caption,
                    image_path=str(img_full_path)
                )
                processed_count += 1
            except Exception as e:
                print(f"Failed to process {img_name}: {e}")

    # 5. Save DB
    vector_db.save(str(DB_PATH))
    
    return {"status": "success", "processed_images": processed_count}

# --- API 2: Query -> Retrieve -> Generate ---
@app.post("/generate", response_model=GenerateResponse)
async def generate_image(req: GenerateRequest):
    global vector_db, rag_engine
    
    # 1. Load Images
    db_pil_images = []
    valid_indices = []
    
    # Only load images that actually exist on disk (sanitization)
    for idx, path in enumerate(vector_db.image_paths):
        if Path(path).exists():
            try:
                db_pil_images.append(Image.open(path).convert("RGB"))
                valid_indices.append(idx)
            except:
                pass
    
    # Filter captions to match valid images
    valid_captions = [vector_db.captions[i] for i in valid_indices]

    # 2. Retrieve
    results = rag_engine.retrieve_pareto(
        query=req.query,
        database_images=db_pil_images,
        database_captions=valid_captions,
        database_patches=None 
    )

    if not results or not results['pareto_indices']:
        # Fallback if no images found: Generate without context or error out
        # Here we error out to enforce RAG usage
        raise HTTPException(status_code=404, detail="No relevant images found")

    # 3. Format Context
    pareto_indices_mapped = results['pareto_indices'] # These are indices into 'valid_indices' list
    
    # Get original DB indices
    original_db_indices = [valid_indices[i] for i in pareto_indices_mapped]
    
    retrieved_captions = [vector_db.captions[i] for i in original_db_indices]
    
    style_context = ". ".join(retrieved_captions[:3])
    enhanced_prompt = (
        f"Generate an image of: {req.query}. "
        f"Ensure the style and composition incorporates these elements: {style_context}. "
        f"Photorealistic, high quality."
    )

    # 4. Generate
    gen_base64 = rag_engine.generate_gemini(req.api_key, enhanced_prompt)

    if not gen_base64:
        raise HTTPException(status_code=500, detail="Gemini generation failed")

    # 5. Response
    response_context = []
    for i, idx in enumerate(original_db_indices):
        # Convert image to base64
        img_path = vector_db.image_paths[idx]
        with open(img_path, "rb") as img_f:
            b64_str = base64.b64encode(img_f.read()).decode("utf-8")
        
        # Calculate matches (using the mapped index 'i' which corresponds to results array)
        result_idx = pareto_indices_mapped[i] 
        satisfied = [results['subqueries'][k] for k in range(len(results['subqueries'])) 
                     if results['sparse_scores'][result_idx, k] == 1]
        
        dense_score = float(results['dense_scores'][result_idx].mean())

        response_context.append(RetrievalResult(
            image_base64=b64_str,
            caption=vector_db.captions[idx],
            matches=satisfied,
            dense_score=dense_score
        ))

    return GenerateResponse(
        generated_image_base64=gen_base64,
        retrieved_context=response_context
    )

# --- API 3: Get All Images (Gallery) ---
@app.get("/images", response_model=GalleryResponse)
async def get_images(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """
    Get paginated list of images currently in the DB.
    """
    global vector_db
    
    total_images = len(vector_db.image_ids)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    # Slice the data
    slice_ids = vector_db.image_ids[start_idx:end_idx]
    slice_captions = vector_db.captions[start_idx:end_idx]
    slice_paths = vector_db.image_paths[start_idx:end_idx]
    
    images_list = []
    
    for i, path in enumerate(slice_paths):
        if Path(path).exists():
            try:
                with open(path, "rb") as img_f:
                    b64_str = base64.b64encode(img_f.read()).decode("utf-8")
                
                images_list.append(ImageItem(
                    id=slice_ids[i],
                    caption=slice_captions[i],
                    image_base64=b64_str
                ))
            except Exception as e:
                # If image is corrupted or fails to read, skip it
                print(f"Error reading {path}: {e}")
                continue
                
    return GalleryResponse(
        total=total_images,
        page=page,
        limit=limit,
        images=images_list
    )

class IngestUrlRequest(BaseModel):
    url: str
    caption: str

# ... existing imports ...

@app.post("/ingest-url")
async def ingest_url(req: IngestUrlRequest):
    global vector_db, rag_engine
    
    print(f"Attempting to download: {req.url}") # Debug log
    
    try:
        # --- FIX: FULL BROWSER HEADERS ---
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1"
        }
        
        # 1. Download Image with ROBUST Headers
        # We use stream=True to avoid reading massive files into memory if they fail
        response = requests.get(req.url, headers=headers, timeout=15, stream=True)
        
        # Check for specific 403 error and print text
        if response.status_code == 403:
            print(f"403 Blocked. Server said: {response.text[:200]}")
            
        response.raise_for_status()
        
        # 2. Convert to PIL
        image_data = BytesIO(response.content)
        pil_img = Image.open(image_data).convert("RGB")
        
        # 3. Create Filename
        # Sanitize filename to remove weird characters from URL
        import re
        raw_filename = req.url.split("/")[-1].split("?")[0]
        filename = re.sub(r'[^\w\-_.]', '', raw_filename) # Keep only alphanumeric, dash, underscore, dot
        
        if not filename or len(filename) > 100:
             filename = f"url_import_{len(vector_db.image_ids)}.jpg"
        
        # Save locally
        local_path = UPLOAD_DIR / filename
        pil_img.save(local_path)
        
        # 4. Generate Embedding
        import torch
        inputs = rag_engine.processor(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embed = rag_engine.clip_model.get_image_features(**inputs)
            embed_np = embed.squeeze(0).cpu().numpy()
            
        # 5. Add to DB
        vector_db.add_image(
            image_id=filename,
            embedding=embed_np,
            caption=req.caption,
            image_path=str(local_path),
            source_url=req.url
        )
        
        # 6. Save DB
        vector_db.save(str(DB_PATH))
        
        return {"status": "success", "image_id": filename}

    except Exception as e:
        print(f"Error in ingest_url: {e}") # Server log
        raise HTTPException(status_code=400, detail=f"Failed to ingest URL: {str(e)}")