# ingestion.py
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def process_and_store_pdf(pdf_path, collection_name="socratic_collection"):
    """
    Extracts, chunks, embeds, and stores text from a PDF.
    This version is context-aware, prepending headings to the text chunks.
    """
    print(f"Processing {pdf_path} with context-aware chunking...")
    
    filename = os.path.basename(pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Split the document into lines
    lines = raw_text.split('\n')
    
    all_chunks = []
    current_heading = ""
    current_paragraph = ""

    # Regex to identify lines that are likely headings (e.g., "Amendment of section 183.")
    # This looks for lines that start with "Amendment of section" or "Insertion of new section" etc.
    heading_pattern = re.compile(r"^(Amendment of section|Insertion of new section|Substitution of new section for section)\s+\d+[A-Z]?\.")

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Check if the line is a heading
        if heading_pattern.match(stripped_line):
            # If we have a paragraph under the *previous* heading, process it
            if current_paragraph:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                # Prepend the heading to the paragraph text
                contextual_text = f"Regarding {current_heading}: {current_paragraph}"
                chunks = text_splitter.split_text(contextual_text)
                all_chunks.extend(chunks)
            
            # Start a new paragraph and update the current heading
            current_paragraph = ""
            current_heading = stripped_line
        else:
            # If it's not a heading, append it to the current paragraph
            current_paragraph += " " + stripped_line

    # Process the very last paragraph in the file
    if current_paragraph:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        contextual_text = f"Regarding {current_heading}: {current_paragraph}"
        chunks = text_splitter.split_text(contextual_text)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("Warning: No chunks were generated. The document might be empty or in an unusual format.")
        # Fallback to simple chunking if the advanced method fails
        print("Falling back to simple chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_chunks = text_splitter.split_text(raw_text)

    print(f"Split text into {len(all_chunks)} context-aware chunks.")
    
    # 3. Embed Chunks
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(all_chunks)
    print("Created embeddings for chunks.")
    
    # 4. Store in ChromaDB with Metadata
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name=collection_name)
    
    ids = [f"{filename}-{i}" for i in range(len(all_chunks))]
    metadatas = [{'source': filename} for _ in all_chunks]
    
    # Add to the collection (use upsert to overwrite if IDs already exist)
    collection.upsert(
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully stored/updated chunks from '{filename}' with metadata.")

# Main execution block for testing
if __name__ == '__main__':
    # --- IMPORTANT: Change this to the exact name of your PDF file ---
    PDF_FILENAME = "Motor_Vehicles_Act_1988_with_Amendments_2019.pdf"
    # -------------------------------------------------------------
    sample_pdf_path = os.path.join('uploads', PDF_FILENAME)
    if os.path.exists(sample_pdf_path):
        process_and_store_pdf(sample_pdf_path)
    else:
        print(f"Please put a PDF file at {sample_pdf_path} to begin.")