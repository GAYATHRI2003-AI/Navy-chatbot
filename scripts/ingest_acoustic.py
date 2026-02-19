import os
import sys
# sys.path hack for Object_Detection conflict
sys.path = [p for p in sys.path if "Object_Detection" not in p]

import torch
import librosa
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "DeepShip-main")
OUTPUT_INDEX = os.path.join(PROJECT_ROOT, "models", "vector_indices", "acoustic_index")
SAMPLE_RATE = 22050
DURATION = 5 # Process 5-second snippets

print(f"--- UATA ACOUSTIC INGESTOR (Scale: DeepShip Dataset) ---")

def get_audio_embedding(audio_path):
    try:
        # Load audio snippet
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < sr * DURATION:
            y = np.pad(y, (0, sr * DURATION - len(y)))
        
        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Simple Feature Vector: Mean across time (Acoustic Signature)
        # In a production system, we'd use a CNN or AST here.
        embedding = np.mean(mel_db, axis=1)
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        return None

def ingest_acoustic_data():
    documents = []
    all_embeddings = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    # DeepShip structure: DeepShip-main/{Class}/{ID}/{File.wav}
    classes = [c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))]
    
    for category in classes:
        cat_path = os.path.join(DATA_DIR, category)
        print(f"Processing Vessel Class: {category}...")
        
        # Find all .wav files in subdirectories
        vessel_audio_files = []
        for root, dirs, files in os.walk(cat_path):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3')):
                    vessel_audio_files.append(os.path.join(root, file))
        
        # Limit sampling for the index prototype
        for audio_path in tqdm(vessel_audio_files[:100], desc=f"Ingesting {category}"):
            emb = get_audio_embedding(audio_path)
            
            if emb is not None:
                doc_content = f"Acoustic signature of {category} vessel."
                metadata = {
                    "path": os.path.abspath(audio_path),
                    "category": category,
                    "type": "acoustic_signal"
                }
                documents.append(Document(page_content=doc_content, metadata=metadata))
                all_embeddings.append(emb.tolist())

    if not documents:
        print("Error: No acoustic data found.")
        return

    # Helper for FAISS
    class AcousticSearchEmbeddings:
        def embed_query(self, query_text):
            # This index is designed for audio-to-audio matching, 
            # for text-to-audio we'd need a multi-modal model.
            # For now, we'll use a neutral vector for text queries.
            return [0.0] * 128
        def embed_documents(self, texts): return all_embeddings

    db = FAISS.from_documents(documents, AcousticSearchEmbeddings())
    db.save_local(OUTPUT_INDEX)
    print(f"\nSUCCESS: Acoustic Survey Index built with {len(documents)} samples.")

if __name__ == "__main__":
    ingest_acoustic_data()
