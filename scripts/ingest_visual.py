import os
import sys
# sys.path hack for Object_Detection conflict
sys.path = [p for p in sys.path if "Object_Detection" not in p]

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np
from tqdm import tqdm

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_FOLDERS = ["train", "test", "quarry-fullsize", "turntable-cropped"]
# Update DATA_FOLDERS to absolute paths
DATA_FOLDERS = [os.path.join(DATA_ROOT, folder) for folder in DATA_FOLDERS]

OUTPUT_INDEX = os.path.join(PROJECT_ROOT, "models", "vector_indices", "visual_index")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- UATA VISUAL INGESTOR (Scale: 58GB Dataset) ---")
print(f"Device: {DEVICE}")

# Load CLIP Model
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(DEVICE)
processor = CLIPProcessor.from_pretrained(model_id)

def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        embedding = image_features.cpu().numpy().flatten()
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        return None

def process_file_batch(files, folder_path, category, documents, all_embeddings, batch_size=32):
    # Batching Loop for Speed
    for i in tqdm(range(0, len(files), batch_size), desc=f"Ingesting {category}"):
        batch_files = files[i : i + batch_size]
        batch_paths = [os.path.join(folder_path, f) for f in batch_files]
        images = []
        valid_paths = []
        
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except: continue
        
        if not images: continue
        
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize and store
        features = image_features.cpu().numpy()
        for j, emb in enumerate(features):
            emb = emb / np.linalg.norm(emb)
            doc_content = f"Sonar pattern: {category}."
            metadata = {
                "path": os.path.abspath(valid_paths[j]), 
                "category": category, 
                "source": folder_path
            }
            documents.append(Document(page_content=doc_content, metadata=metadata))
            all_embeddings.append(emb.tolist())

def ingest_sonar_data(sample_per_category=2000, batch_size=32):
    documents = []
    all_embeddings = []
    
    # 1. Ingest Standard Folders
    for root_dir in DATA_FOLDERS:
        if not os.path.exists(root_dir): continue
        
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path): continue
            
            print(f"Processing Category: {category}...")
            files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            files = files[:sample_per_category]
            
            process_file_batch(files, category_path, category, documents, all_embeddings, batch_size)

    # 2. Ingest Navy Images (New Dataset Support)
    NAVY_IMAGES_ROOT = os.path.join(DATA_ROOT, "Navy_images")
    if os.path.exists(NAVY_IMAGES_ROOT):
        print(f"Scanning Navy Images at {NAVY_IMAGES_ROOT}...")
        for rec_dir in os.listdir(NAVY_IMAGES_ROOT):
            rec_path = os.path.join(NAVY_IMAGES_ROOT, rec_dir)
            if not os.path.isdir(rec_path): continue

            # Search for 'sonar' subfolder
            sonar_path = os.path.join(rec_path, "sonar")
            if os.path.exists(sonar_path) and os.path.isdir(sonar_path):
                category = "navy_sonar_scan" # Standardized category
                print(f"Processing Navy Recording: {rec_dir} (Category: {category})...")
                
                files = [f for f in os.listdir(sonar_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                # Apply sampling limit if needed, or take all if critical
                files = files[:sample_per_category]
                
                process_file_batch(files, sonar_path, category, documents, all_embeddings, batch_size)

    if not documents:
        print("Error: No sonar images found.")
        return

    class VisualSearchEmbeddings:
        def __call__(self, text):
            inputs = processor(text=[text], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
            emb = text_features.cpu().numpy().flatten()
            return (emb / np.linalg.norm(emb)).tolist()
        def embed_query(self, text): return self.__call__(text)
        def embed_documents(self, texts): return all_embeddings

    db = FAISS.from_documents(documents, VisualSearchEmbeddings())
    db.save_local(OUTPUT_INDEX)
    print(f"\nSUCCESS: Visual Survey Index built with {len(documents)} samples.")
    print(f"Index Location: {OUTPUT_INDEX}")

if __name__ == "__main__":
    # You can increase sample_per_category to index the full 58GB dataset
    ingest_sonar_data(sample_per_category=1000)
