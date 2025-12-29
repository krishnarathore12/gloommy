import os
import requests
import json
import zipfile
import shutil
from duckduckgo_search import DDGS

def download_company_images(company_name, num_images=10):
    """
    Downloads images using DuckDuckGo and creates metadata.json
    Returns the path of the created directory.
    """
    # Create directory
    dir_name = company_name.replace(" ", "_")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    metadata = []
    print(f"Searching for images of '{company_name}'...")

    with DDGS() as ddgs:
        # keywords: query string
        # max_results: how many images to fetch
        results = ddgs.images(
            keywords=company_name,
            region="wt-wt",
            safesearch="off",
            max_results=num_images
        )

        # enumerate allows us to get index (i) and item (result)
        for i, result in enumerate(results):
            image_url = result['image']
            caption = result['title']  # This serves as the caption
            
            # File setup - clean up extension
            ext = image_url.split(".")[-1].split("?")[0]
            if len(ext) > 4 or ext.lower() not in ['jpg', 'jpeg', 'png', 'webp']:
                ext = 'jpg'
                
            filename = f"{dir_name}_{i}.{ext}"
            filepath = os.path.join(dir_name, filename)

            try:
                # Download Image
                print(f"Downloading {i+1}/{num_images}: {caption[:30]}...")
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    # Save Metadata (Image-Caption Pair)
                    metadata.append({
                        "file_name": filename,
                        "caption": caption,
                        "source_url": result.get('url', image_url)
                    })
                else:
                    print(f"Skipping {i+1}: Status code {response.status_code}")
                
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")

    # Save captions to JSON
    json_path = os.path.join(dir_name, 'metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nMetadata saved to: {json_path}")
    return dir_name

def create_zip(source_folder):
    """
    Zips the contents of the folder into {source_folder}.zip
    """
    zip_filename = f"{source_folder}.zip"
    print(f"Zipping folder '{source_folder}' into '{zip_filename}'...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Arcname is the name inside the zip file
                # We want the files to be at the root of the zip or inside the folder structure
                # This puts them inside a folder matching the directory name
                arcname = os.path.relpath(file_path, os.path.dirname(source_folder))
                zipf.write(file_path, arcname)
                
    print(f"✅ Zip created successfully: {zip_filename}")
    return zip_filename

if __name__ == "__main__":
    # 1. Download Real Data
    topic = "Tesla Cybertruck"  # Change this to whatever you want
    folder_path = download_company_images(topic, num_images=5)
    
    # 2. Create Zip
    if os.path.exists(folder_path):
        zip_file = create_zip(folder_path)
        
        # Optional: Cleanup raw folder to keep things clean
        # shutil.rmtree(folder_path)
        # print(f"Cleaned up temporary folder: {folder_path}")
        
        print(f"\nYou can now upload '{zip_file}' to the /ingest endpoint.")