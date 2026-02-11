from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Initialize the sentence transformer model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Embeds chunks using sentence-transformers model."""
    print(f"Embedding {len(chunks)} chunks using sentence-transformers...")
    
    # Generate embeddings with progress bar
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Embedded {len(embeddings)} chunks")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First embedding sample (first 5 values): {embeddings[0][:5]}")
    
    return embeddings