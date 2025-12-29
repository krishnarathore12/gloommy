import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
import spacy
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import base64
from io import BytesIO
import os

# Google GenAI Imports
from google import genai
from google.genai import types

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
NANO_BANANA_MODEL = "gemini-2.5-flash-image"
BETA = 0.015  # Hyperparameter for Pareto weighting

# ============================================================================
# 1. VISION ADAPTER
# ============================================================================
class VisionAdapter(nn.Module):
    """
    Vision adapter for sub-dimensional dense retrieval.
    aligns visual patches with textual concepts.
    """
    def __init__(self, embed_dim=768, text_dim=512, output_dim=512):
        super().__init__()
        # Learnable query token
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-Attention layers
        self.vision_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.text_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(self, patches, concept_embedding):
        # patches: (Batch, Seq_Len, Embed_Dim)
        # concept_embedding: (Batch, Text_Dim)
        
        batch_size = patches.shape[0]
        queries = self.query_token.expand(batch_size, -1, -1)
        
        # Project text concept to visual dimension
        concept_kv = self.text_proj(concept_embedding).unsqueeze(1)
        
        # Attention over Vision Patches
        attn_out_vis, _ = self.vision_attn(query=queries, key=patches, value=patches)
        queries = self.norm1(queries + attn_out_vis)
        
        # Attention over Text Concept
        attn_out_txt, _ = self.text_attn(query=queries, key=concept_kv, value=concept_kv)
        queries = self.norm2(queries + attn_out_txt)
        
        # Final Projection
        out = self.mlp(queries)
        v_ji = out.squeeze(1) # Remove sequence dim -> (Batch, Output_Dim)
        
        # L2 Normalize
        return v_ji / (v_ji.norm(dim=-1, keepdim=True) + 1e-8)

# ============================================================================
# 2. QUERY DECOMPOSER
# ============================================================================
class QueryDecomposer:
    """Decompose query into subqueries and core concepts using Spacy"""
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.IGNORE_ENTITIES = {
            "photo", "image", "picture", "shot", "view", "scene", 
            "overhead shot", "window showing", "one", "other", "part"
        }

    def decompose(self, caption: str) -> Tuple[List[str], List[str]]:
        doc = self.nlp(caption)
        subqueries = []
        core_concepts = []
        
        for ent in doc.noun_chunks:
            subquery = ent.text.lower().strip()
            core_concept = ent.root.text.lower().strip()
            
            # Filtering logic
            if core_concept in self.IGNORE_ENTITIES:
                continue
            if ent.root.pos_ == 'PRON':
                continue
            if len(core_concept) < 2:
                continue
            
            subqueries.append(subquery)
            core_concepts.append(core_concept)
        
        # Fallback if no entities found
        if not subqueries:
            subqueries = [caption]
            core_concepts = [caption]
            
        return subqueries, core_concepts

# ============================================================================
# 3. MAIN RAG ENGINE
# ============================================================================
class RAGEngine:
    def __init__(self, adapter_path: str = None):
        print(f"Initializing RAG Engine on {DEVICE}...")
        
        # Load CLIP Components
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.clip_model.eval()
        self.text_model.eval()
        
        # Load Adapter
        self.adapter = VisionAdapter(embed_dim=768, text_dim=512, output_dim=512).to(DEVICE)
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading Adapter weights from {adapter_path}")
            self.adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        else:
            print("WARNING: Adapter weights not found or not provided. Using random initialization.")
        self.adapter.eval()
        
        # Initialize Decomposer
        self.decomposer = QueryDecomposer()

    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """Encode text using CLIP Text Encoder"""
        if not text_list:
            return torch.tensor([]).to(DEVICE)
            
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeds = outputs.pooler_output
        return embeds

    def extract_patches(self, image: Image.Image) -> torch.Tensor:
        """Extract visual patches from an image using CLIP Vision Model"""
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(**inputs)
            # last_hidden_state shape: (Batch, Sequence_Length, Hidden_Size)
            patches = vision_outputs.last_hidden_state
        return patches

    def compute_sparse_scores(self, query_subqueries: List[str], database_captions: List[str]) -> np.ndarray:
        """Compute Keyword/Lexical overlap scores"""
        num_images = len(database_captions)
        num_subqueries = len(query_subqueries)
        sparse_matrix = np.zeros((num_images, num_subqueries))
        
        for i, caption in enumerate(database_captions):
            caption_doc = self.decomposer.nlp(caption.lower())
            caption_lemmas = set([token.lemma_ for token in caption_doc])
            caption_text = set([token.text for token in caption_doc])
            
            for j, subquery in enumerate(query_subqueries):
                subquery_doc = self.decomposer.nlp(subquery.lower())
                subquery_lemmas = set([token.lemma_ for token in subquery_doc])
                subquery_text = set([token.text for token in subquery_doc])
                
                # Match logic: Exact substring OR Lemma overlap OR Token overlap
                if (subquery.lower() in caption.lower() or 
                    subquery_lemmas & caption_lemmas or 
                    subquery_text & caption_text):
                    sparse_matrix[i, j] = 1.0
        
        return sparse_matrix

    def compute_dense_scores(self, 
                             query_subqueries: List[str],
                             query_core_concepts: List[str],
                             database_patches: List[torch.Tensor]) -> np.ndarray:
        """
        Compute dense scores using Vision Adapter.
        Iterates through images and calculates alignment between
        specific image patches and textual concepts.
        """
        num_images = len(database_patches)
        num_subqueries = len(query_subqueries)
        dense_matrix = np.zeros((num_images, num_subqueries))
        
        # Pre-compute text embeddings
        subquery_embeds = self.encode_text(query_subqueries)
        concept_embeds = self.encode_text(query_core_concepts)
        
        with torch.no_grad():
            for i, patches in enumerate(database_patches):
                # patches shape: (1, Seq_Len, Embed_Dim)
                for j in range(num_subqueries):
                    c_emb = concept_embeds[j].unsqueeze(0) # (1, 512)
                    
                    # Forward pass through adapter
                    # Returns visual vector aligned with concept
                    v_ji = self.adapter(patches, c_emb) 
                    
                    t_target = subquery_embeds[j].unsqueeze(0)
                    
                    # Normalize
                    v_ji = v_ji / (v_ji.norm(dim=-1, keepdim=True) + 1e-8)
                    t_target = t_target / (t_target.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    # Cosine Similarity
                    score = torch.cosine_similarity(v_ji, t_target).item()
                    dense_matrix[i, j] = score
        
        return dense_matrix

    def find_pareto_optimal(self, sparse_scores: np.ndarray, dense_scores: np.ndarray, beta: float = BETA) -> List[int]:
        """Identify Pareto-Optimal indices based on Sparse and Dense scores"""
        num_images, num_subqueries = sparse_scores.shape
        
        # Filter: Only consider images that have at least one sparse match
        # (This is an optimization to avoid processing irrelevant images)
        valid_mask = sparse_scores.sum(axis=1) > 0
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # Fallback: if no sparse matches, assume all are candidates (pure dense retrieval)
            valid_indices = np.arange(num_images)
        
        combined_scores = []
        for idx in valid_indices:
            sparse_sum = sparse_scores[idx].sum()
            dense_sum = dense_scores[idx].sum()
            
            # Simple scalarization for sorting/filtering if needed, but we use dominance check
            combined = sparse_sum + beta * num_subqueries * (dense_sum / (num_subqueries + 1e-8))
            combined_scores.append((idx, sparse_sum, dense_sum, combined))
        
        pareto_set = []
        # Check for dominance
        for i, (idx_i, sparse_i, dense_i, _) in enumerate(combined_scores):
            is_dominated = False
            for j, (idx_j, sparse_j, dense_j, _) in enumerate(combined_scores):
                if i != j:
                    # Logic: J dominates I if J is better in both dimensions
                    # or better in one and equal in the other
                    if sparse_j >= sparse_i and dense_j >= dense_i:
                        if sparse_j > sparse_i or dense_j > dense_i:
                            is_dominated = True
                            break
            if not is_dominated:
                pareto_set.append(idx_i)
        
        # Sort pareto set by dense score descending as a tie-breaker
        pareto_set.sort(key=lambda idx: dense_scores[idx].sum(), reverse=True)
        return pareto_set

    def retrieve_pareto(self, 
                        query: str, 
                        database_images: List[Image.Image], 
                        database_captions: List[str],
                        database_patches: Optional[List[torch.Tensor]] = None) -> Dict:
        """
        Main Retrieval Function.
        1. Decompose Query
        2. Extract Patches (if not provided)
        3. Compute Sparse & Dense Scores
        4. Find Pareto Frontier
        """
        # 1. Decompose
        subqueries, core_concepts = self.decomposer.decompose(query)
        if not subqueries:
            return None
        
        # 2. Extract Patches
        # Note: In production, patches should be cached in the VectorDB to avoid re-computation
        if database_patches is None:
            database_patches = [self.extract_patches(img) for img in database_images]
            
        # 3. Compute Scores
        sparse_scores = self.compute_sparse_scores(subqueries, database_captions)
        dense_scores = self.compute_dense_scores(subqueries, core_concepts, database_patches)
        
        # 4. Pareto Optimization
        pareto_indices = self.find_pareto_optimal(sparse_scores, dense_scores)
        
        return {
            'query': query,
            'subqueries': subqueries,
            'core_concepts': core_concepts,
            'pareto_indices': pareto_indices,
            'sparse_scores': sparse_scores,
            'dense_scores': dense_scores
        }

    def generate_gemini(self, api_key: str, prompt: str) -> Optional[str]:
        """
        Generate image using Google's Nano Banana (Gemini 2.5 Flash Image).
        Returns: Base64 string of the image or None.
        """
        try:
            client = genai.Client(api_key=api_key)
            
            response = client.models.generate_content(
                model=NANO_BANANA_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    temperature=1.0, 
                )
            )
            
            # Extract image from response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # part.inline_data.data is usually a base64 string or bytes
                        # If it's bytes, convert to b64 string
                        data = part.inline_data.data
                        if isinstance(data, bytes):
                            return base64.b64encode(data).decode('utf-8')
                        return data # Already string
                        
            return None

        except Exception as e:
            print(f"Error generating image with Nano Banana: {str(e)}")
            return None