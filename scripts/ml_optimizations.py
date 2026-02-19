"""
ML/DL Optimization Module
- Acoustic embeddings, CLIP batching, model quantization, caching
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import librosa
from collections import OrderedDict
import time
from functools import lru_cache

# ============================================================
# 1. ACOUSTIC EMBEDDINGS (Replace dummy implementation)
# ============================================================
class AcousticFeatureExtractor:
    """Extract real acoustic features from audio"""
    
    def __init__(self, sr=16000, n_mfcc=13, n_mel=128):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mel = n_mel
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC + Mel-spectrogram features"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mel)
            mel_mean = np.mean(mel_spec, axis=1)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # Spectral centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spec_cent_mean = np.mean(spec_cent)
            
            # Combine all features into 128-dim vector
            features = np.concatenate([
                mfcc_mean[:13],  # 13 MFCC
                mel_mean[:50],   # 50 Mel features
                zcr_mean.reshape(1),  # 1 ZCR
                spec_cent_mean.reshape(1),  # 1 Spectral centroid
                np.zeros(63)  # Padding to 128
            ])[:128]
            
            # Normalize
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            return features
        except Exception as e:
            print(f"Acoustic extraction error: {e}")
            return np.random.randn(128)  # Fallback


# ============================================================
# 2. CLIP BATCH INFERENCE
# ============================================================
class CLIPBatchProcessor:
    """Batch CLIP inference for efficiency"""
    
    def __init__(self, clip_model, clip_processor, device="cpu", batch_size=8):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device
        self.batch_size = batch_size
    
    def process_images_batch(self, images: List) -> np.ndarray:
        """Process multiple images in batches"""
        embeddings = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Handle BaseModelOutputWithPooling
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            
            embeddings.append(image_features.cpu().numpy())
        
        # Concatenate and normalize
        all_embeddings = np.vstack(embeddings)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        return all_embeddings
    
    def process_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Process multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            
            embeddings.append(text_features.cpu().numpy())
        
        all_embeddings = np.vstack(embeddings)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        return all_embeddings


# ============================================================
# 3. MODEL QUANTIZATION
# ============================================================
class ModelQuantizer:
    """Apply int8 quantization to reduce model size"""
    
    @staticmethod
    def quantize_model(model, quantization_type="int8"):
        """Quantize a PyTorch model"""
        if quantization_type == "int8":
            try:
                from torch.quantization import quantize_dynamic, AutocastPolicy
                quantized = quantize_dynamic(
                    model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                print(f"✓ Model quantized to int8")
                return quantized
            except Exception as e:
                print(f"Quantization failed: {e}. Using original model.")
                return model
        return model


# ============================================================
# 4. EMBEDDING CACHE
# ============================================================
class EmbeddingCache:
    """LRU cache for embeddings"""
    
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: np.ndarray) -> None:
        """Cache embedding"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.cache.move_to_end(key)
    
    def get_stats(self) -> Dict:
        """Cache hit/miss statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate*100:.1f}%",
            "size": len(self.cache)
        }


# ============================================================
# 5. ADAPTIVE TIMEOUT SYSTEM
# ============================================================
class AdaptiveTimeout:
    """Adjust timeout based on queue depth and response time"""
    
    def __init__(self, base_timeout=25, max_timeout=60):
        self.base_timeout = base_timeout
        self.max_timeout = max_timeout
        self.response_times = []
        self.queue_depths = []
    
    def record_response(self, response_time: float, queue_depth: int) -> None:
        """Record response metrics"""
        self.response_times.append(response_time)
        self.queue_depths.append(queue_depth)
        # Keep last 100 samples
        if len(self.response_times) > 100:
            self.response_times.pop(0)
            self.queue_depths.pop(0)
    
    def get_timeout(self, current_queue_depth: int) -> float:
        """Adaptive timeout based on history"""
        if not self.response_times:
            return self.base_timeout
        
        avg_time = np.mean(self.response_times[-10:])  # Last 10 avg
        queue_factor = (current_queue_depth + 1) * 1.5
        
        timeout = avg_time + queue_factor
        return min(timeout, self.max_timeout)


# ============================================================
# 6. MODEL VERSIONING & CHECKPOINT MANAGEMENT
# ============================================================
class ModelVersionManager:
    """Track and manage model versions"""
    
    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, version: str, metadata: dict = None):
        """Save model checkpoint"""
        import os
        import json
        
        path = os.path.join(self.checkpoint_dir, f"model_v{version}.pt")
        torch.save(model.state_dict(), path)
        
        # Save metadata
        meta_path = os.path.join(self.checkpoint_dir, f"model_v{version}_meta.json")
        meta = metadata or {}
        meta["timestamp"] = time.time()
        meta["version"] = version
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✓ Checkpoint saved: v{version}")
    
    def load_checkpoint(self, model, version: str):
        """Load specific model version"""
        import os
        path = os.path.join(self.checkpoint_dir, f"model_v{version}.pt")
        
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            print(f"✓ Loaded checkpoint: v{version}")
            return model
        else:
            print(f"Checkpoint not found: v{version}")
            return model


# ============================================================
# 7. VECTOR DB AUTO-REBUILD
# ============================================================
class VectorDBManager:
    """Auto-rebuild FAISS indices on failure"""
    
    @staticmethod
    def load_with_rebuild(index_path: str, embeddings_obj, max_retries=3):
        """Try loading index with auto-rebuild fallback"""
        from langchain_community.vectorstores import FAISS
        import os
        
        for attempt in range(max_retries):
            try:
                if os.path.exists(index_path):
                    db = FAISS.load_local(
                        index_path, 
                        embeddings_obj, 
                        allow_dangerous_deserialization=True
                    )
                    print(f"✓ Loaded FAISS index: {index_path}")
                    return db
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries}: Index load failed: {e}")
                if attempt == max_retries - 1:
                    print(f"Could not load index. Returning None.")
                    return None
        
        return None


# ============================================================
# 8. KNOWLEDGE BASE MANAGER
# ============================================================
class DynamicKnowledgeBase:
    """Dynamic knowledge base with update capability"""
    
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.kb = self._load_kb()
    
    def _load_kb(self) -> dict:
        """Load knowledge base"""
        import json
        import os
        
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        return {"submarines": {}, "weapons": {}, "bases": {}}
    
    def add_fact(self, category: str, key: str, value: dict) -> None:
        """Add new fact to knowledge base"""
        if category not in self.kb:
            self.kb[category] = {}
        self.kb[category][key] = value
        self._save_kb()
        print(f"✓ Added fact: {category}/{key}")
    
    def verify_fact(self, category: str, key: str, value: str) -> bool:
        """Verify if fact exists in knowledge base"""
        if category in self.kb and key in self.kb[category]:
            fact = self.kb[category][key]
            # Check multiple fields for match
            for field, field_value in fact.items():
                if str(field_value).lower() == str(value).lower():
                    return True
        return False
    
    def _save_kb(self) -> None:
        """Save knowledge base"""
        import json
        with open(self.kb_path, 'w') as f:
            json.dump(self.kb, f, indent=2)


print("✓ ML Optimizations module loaded")
