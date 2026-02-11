import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple

class FAISSVectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        # Using IndexFlatL2 for L2 (Euclidean) distance
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        
    def add_vectors(self, chunks: List[str], embeddings: np.ndarray):
        """
        Add vectors to the FAISS index.
        
        Args:
            chunks: List of text chunks
            embeddings: Numpy array of embeddings with shape (n, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")
        
        # Add embeddings to FAISS index
        self.index.add(np.array(embeddings, dtype=np.float32))
        
        # Store chunks for retrieval
        self.chunks.extend(chunks)
        
        print(f"Added {len(chunks)} vectors to FAISS index")
        print(f"Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of tuples (chunk_text, distance)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Ensure index is valid
                results.append((self.chunks[idx], float(distances[0][i])))
        
        return results
    
    def save(self, index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl"):
        """
        Save FAISS index and chunks to disk.
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save chunks using pickle
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved FAISS index to {index_path}")
        print(f"Saved chunks to {chunks_path}")
    
    def load(self, index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl"):
        """
        Load FAISS index and chunks from disk.
        
        Args:
            index_path: Path to load FAISS index from
            chunks_path: Path to load chunks from
        """
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Index or chunks file not found")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded FAISS index from {index_path}")
        print(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
        print(f"Index contains {self.index.ntotal} vectors")


# Convenience function for backward compatibility
def store_in_faiss(chunks: List[str], embeddings: np.ndarray, save_path: str = "./") -> FAISSVectorStore:
    """
    Store chunks and embeddings in FAISS vector store.
    
    Args:
        chunks: List of text chunks
        embeddings: Numpy array of embeddings
        save_path: Directory path to save the index
        
    Returns:
        FAISSVectorStore instance
    """
    dimension = embeddings.shape[1]
    vector_store = FAISSVectorStore(dimension=dimension)
    vector_store.add_vectors(chunks, embeddings)
    
    # Save to disk
    index_file = os.path.join(save_path, "faiss_index.bin")
    chunks_file = os.path.join(save_path, "chunks.pkl")
    vector_store.save(index_file, chunks_file)
    
    return vector_store