# main.py

# --- 1. Imports ---
import os
import datetime
import chromadb
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any

from dotenv import load_dotenv
from bot_logic import SocraticTutor
from ingestion import process_and_store_pdf

# --- 2. Initial Setup ---
load_dotenv()
app = FastAPI()

# Setup for serving HTML templates from the "templates" directory
templates = Jinja2Templates(directory="templates")

# Setup for serving static files (like CSS or JS if you add them later)
app.mount("/static", StaticFiles(directory="templates"), name="static")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 3. API Key and Tutor Init ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found!")
tutor = SocraticTutor(api_key=API_KEY)

# --- 4. Pydantic Models (Defines the structure of our API requests) ---
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]]
    document_source: str

# --- 5. API Endpoints ---
@app.get("/")
async def serve_chat_page(request: Request):
    """Serves the main index.html chat page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload")
async def serve_upload_page(request: Request):
    """Serves the upload.html page."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def handle_file_upload(file: UploadFile = File(...)):
    """Handles PDF uploads and triggers the ingestion process."""
    if not file.filename.endswith('.pdf'):
        # In a real frontend, you would handle this error more gracefully
        return JSONResponse(content={"error": "Invalid file type. Please upload a PDF."}, status_code=400)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Asynchronously save the uploaded file
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Run the ingestion process
    try:
        process_and_store_pdf(filepath)
    except Exception as e:
        # If something goes wrong during processing, return an error
        return JSONResponse(content={"error": f"Failed to process file: {e}"}, status_code=500)
    
    # After successful upload and processing, redirect the user back to the upload page.
    # The frontend would ideally show a success message.
    return RedirectResponse(url="/upload", status_code=303)

@app.get("/get_documents")
async def get_documents_list():
    """API endpoint for the frontend to fetch the list of available documents."""
    try:
        client = chromadb.PersistentClient(path="db")
        collection = client.get_or_create_collection(name="socratic_collection")
        items = collection.get(include=["metadatas"])

        # Check if the database has any entries
        if not items or 'metadatas' not in items or not items['metadatas']:
            return JSONResponse(content=[])
        
        # Create a unique, sorted list of document filenames
        sources = sorted(list(set(item['source'] for item in items['metadatas'])))
        return JSONResponse(content=sources)
    except Exception as e:
        print(f"Error getting documents: {e}")
        return JSONResponse(content={"error": "Could not retrieve document list"}, status_code=500)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """API endpoint to handle a user's chat message."""
    # Generate a response using the bot logic
    bot_response = tutor.generate_response(
        student_question=request.message,
        chat_history=request.history,
        document_source=request.document_source
    )
    
    # Log the conversation turn to a text file
    try:
        with open("chat_log.txt", "a", encoding="utf-8") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"--- Turn at {timestamp} for doc: {request.document_source} ---\n")
            log_file.write(f"User: {request.message}\n")
            log_file.write(f"Bot: {bot_response}\n\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")
        
    return JSONResponse(content={"response": bot_response})