"""
Query script for FAISS-based semantic search.
This demonstrates how to search the vector store with natural language queries.
"""

from vectorstore import FAISSVectorStore
from embedder import embed_chunks

def search_documents(query: str, top_k: int = 5):
    """
    Search the FAISS index with a natural language query.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    """
    print(f"Query: {query}\n")
    print("Loading FAISS index...")
    
    # Load existing FAISS index
    vector_store = FAISSVectorStore(dimension=384)
    vector_store.load("faiss_index.bin", "chunks.pkl")
    
    print(f"Loaded index with {vector_store.index.ntotal} vectors\n")
    
    # Generate query embedding
    print("Generating query embedding...")
    query_embedding = embed_chunks([query])[0]
    
    # Search for similar chunks
    print(f"Searching for top {top_k} results...\n")
    results = vector_store.search(query_embedding, k=top_k)
    
    # Display results
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)
    
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"\n[Result {i}] Similarity Score: {1 / (1 + distance):.4f} | Distance: {distance:.4f}")
        print("-" * 80)
        print(chunk[:500])  # Show first 500 characters
        if len(chunk) > 500:
            print("...")
        print()

if __name__ == "__main__":
    # Example queries
    queries = [
        "What were the key financial highlights?",
        "What are the main risks facing the company?",
        "Tell me about the company's sustainability initiatives"
    ]
    
    # Run first query by default
    search_documents(queries[0], top_k=3)
    
    # Uncomment to try other queries:
    # search_documents(queries[1], top_k=3)
    # search_documents(queries[2], top_k=3)
