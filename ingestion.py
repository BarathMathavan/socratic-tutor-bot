# ingestion.py (Simple & Fast Version)

# --- 1. Imports ---
import os
import uuid
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- 2. Core Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def simple_ingestion(pdf_path: str, collection_name: str = "socratic_collection"):
    """
    Performs a fast, simple ingestion without any AI-based enrichment.
    Finishes in seconds.
    """
    print(f"\n--- Starting SIMPLE ingestion for {pdf_path} ---")
    filename = os.path.basename(pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = text_splitter.split_text(raw_text)
    
    print(f"  - Split into {len(all_chunks)} chunks.")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(all_chunks)
    
    qdrant_client = QdrantClient(host="localhost", port=6333)

    try:
        # Check if collection exists. If not, create it.
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        print(f"  - Collection '{collection_name}' not found. Creating it...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )
    
    # Use deterministic IDs based on file and chunk index
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}-{i}")) for i, _ in enumerate(all_chunks)]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=ids[i],
                vector=vector.tolist(),
                payload={"text": chunk, "source": filename}
            ) for i, (chunk, vector) in enumerate(zip(all_chunks, embeddings))
        ],
        wait=True
    )
    print(f"--- SIMPLE ingestion complete for {filename} ---")