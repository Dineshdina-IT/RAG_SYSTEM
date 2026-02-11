from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_faiss
from typing import List

pdf_path = "./resources/Annual_Report_2022-2023.pdf"

def run():
    # Read Annaul Report PDF and extract text
    pages= read_pdf(pdf_path)
    # print (f"Extracted {len (pages)} pages from the PDF.")
    # print("First page content:")
    # print (pages [0] if pages else "No content found.")

    # Chunk the data into smaller pieces 
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)
    # print (f"Total chunks created: {len(chunks)}")
    # print("First chunks: ")
    # print (chunks [0]) # Print first chunk for verification

    # Embed the chunks using sentence-transformers to create vector representations
    embedded_chunks = embed_chunks(chunks)
    print(f"Total chunks embedded: {len(embedded_chunks)}")
    print(f"Embedding shape: {embedded_chunks.shape}")

    # Store the chunks and their embeddings in FAISS vector database
    vector_store = store_in_faiss(chunks, embedded_chunks, save_path="./")
    print("Successfully stored vectors in FAISS!")

if __name__=="__main__":
    run()