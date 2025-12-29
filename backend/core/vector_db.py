import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class SimpleVectorDB:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.image_ids: List[str] = []
        self.embeddings: List[np.ndarray] = [] 
        self.captions: List[str] = []
        self.image_paths: List[str] = []
        self.source_urls: List[str] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

    def add_image(self, image_id: str, embedding: np.ndarray, caption: str, image_path: str, source_url: str = ""):
        self.image_ids.append(image_id)
        self.embeddings.append(embedding) # Store raw list for appending
        self.captions.append(caption)
        self.image_paths.append(image_path)
        self.source_urls.append(source_url)
        # Invalidate matrix cache
        self.embeddings_matrix = None

    def build_index(self):
        """Convert list to matrix for search"""
        if self.embeddings:
            self.embeddings_matrix = np.vstack(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.embeddings_matrix is None:
            self.build_index()
            
        # Normalize
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        db_norm = self.embeddings_matrix / (np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(db_norm, q_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices, similarities[top_indices]

    def save(self, path: str):
        self.build_index()
        data = {
            'embedding_dim': self.embedding_dim,
            'image_ids': self.image_ids,
            'embeddings_matrix': self.embeddings_matrix, # Save the matrix
            'captions': self.captions,
            'image_paths': self.image_paths,
            'source_urls': self.source_urls,
            # We also save the raw list to allow easy appending later
            'embeddings_list': self.embeddings 
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        if not Path(path).exists():
            return cls()
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(embedding_dim=data.get('embedding_dim', 512))
        db.image_ids = data['image_ids']
        db.embeddings_matrix = data['embeddings_matrix']
        db.captions = data['captions']
        db.image_paths = data['image_paths']
        db.source_urls = data.get('source_urls', [])
        
        # Load raw list if available, else reconstruct from matrix
        if 'embeddings_list' in data:
            db.embeddings = data['embeddings_list']
        else:
            db.embeddings = list(db.embeddings_matrix)
            
        return db