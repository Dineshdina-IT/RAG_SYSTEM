"""
RAG System Demo - FAISS Vector Search
This script demonstrates the complete RAG system with FAISS.
"""

from vectorstore import FAISSVectorStore
from embedder import embed_chunks
import numpy as np

def demo_search():
    """Demonstrate semantic search with FAISS."""
    
    print("=" * 80)
    print("RAG SYSTEM DEMO - FAISS Vector Search")
    print("=" * 80)
    print()
    
    # Load FAISS index
    print("📚 Loading FAISS vector store...")
    vector_store = FAISSVectorStore(dimension=384)
    vector_store.load("faiss_index.bin", "chunks.pkl")
    print(f"✅ Loaded {vector_store.index.ntotal} document chunks\n")
    
    # Example queries
    queries = [
        "What were the key financial highlights of the year?",
        "Tell me about employee statistics and recruitment",
        "What are the examination details?"
    ]
    
    print("🔍 Running example queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print('='*80)
        
        # Generate embedding for query
        query_embedding = embed_chunks([query])[0]
        
        # Search for top 3 results
        results = vector_store.search(query_embedding, k=3)
        
        for rank, (chunk, distance) in enumerate(results, 1):
            similarity = 1 / (1 + distance)
            print(f"\n[Result {rank}] Similarity: {similarity:.4f} | L2 Distance: {distance:.4f}")
            print("-" * 80)
            # Show first 300 characters
            preview = chunk[:300].replace('\n', ' ').strip()
            print(preview)
            if len(chunk) > 300:
                print("...")
        
        print()
    
    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print("\n💡 Tips:")
    print("  - Lower L2 distance = more similar")
    print("  - Higher similarity score = better match")
    print("  - Modify queries in this script to try different searches")
    print()

if __name__ == "__main__":
    demo_search()
