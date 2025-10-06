# ingestion.py
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
def process_and_store_pdf(pdf_path, collection_name="socratic_collection"):
    """Extracts, chunks, embeds, and stores text from a PDF."""
    print(f"Processing {pdf_path}...")
    
    # 1. Extract Text
    raw_text = extract_text_from_pdf(pdf_path)
    
    # 2. Chunk Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"Split text into {len(chunks)} chunks.")
    
    # 3. Embed Chunks
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks)
    print("Created embeddings for chunks.")
    
    # 4. Store in ChromaDB
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name=collection_name)
    
    # Create unique IDs for each chunk
    ids = [f"{os.path.basename(pdf_path)}-{i}" for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    print(f"Successfully stored chunks in collection: {collection_name}")

# Example of how to run this:
if __name__ == '__main__':
    # Place a sample PDF in your 'uploads' folder, e.g., 'stability.pdf'
    sample_pdf_path = os.path.join('uploads','the-thinkers-guide-to-the-art-of-socratic-questioning-kindle-edition.pdf')
    if os.path.exists(sample_pdf_path):
        process_and_store_pdf(sample_pdf_path)
    else:
        print(f"Please put a PDF file at {sample_pdf_path} to begin.")