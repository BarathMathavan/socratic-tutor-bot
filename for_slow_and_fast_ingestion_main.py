# main.py

# --- 1. Imports ---
import os
import datetime
import uuid
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any

# --- New/Updated Imports ---
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from pymongo import MongoClient

# --- Local Application Imports ---
from bot_logic import SocraticTutor
from ingestion import process_and_store_pdf

# --- 2. Initial Setup ---
load_dotenv()
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 3. Database and AI Service Connections ---
# Google Gemini API Key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# MongoDB Connection
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise ValueError("MONGODB_URI not found in .env file!")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["socratic_tutor_db"]
sessions_collection = db["chat_sessions"]

# Initialize the Socratic Tutor (which connects to Qdrant inside its class)
tutor = SocraticTutor(api_key=API_KEY)

# --- 4. Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]]
    document_source: str

class ChatLog(BaseModel):
    session_id: str
    timestamp: str
    document_source: str
    user_message: str
    bot_response: str
    chat_history: List[Dict[str, Any]]

# --- 5. API Endpoints ---
@app.get("/")
async def serve_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload")
async def serve_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def handle_file_upload(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return JSONResponse(content={"error": "Invalid file type. Please upload a PDF."}, status_code=400)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        process_and_store_pdf(filepath)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to process file: {e}"}, status_code=500)
    
    return RedirectResponse(url="/upload", status_code=303)

@app.get("/get_documents")
async def get_documents_list():
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        collection_name = "socratic_collection"

        scrolled_points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=["source"],
            with_vectors=False
        )
        
        if not scrolled_points:
            return JSONResponse(content=[])
        
        sources = sorted(list(set(point.payload["source"] for point in scrolled_points)))
        return JSONResponse(content=sources)
    except (UnexpectedResponse, Exception) as e:
        print(f"Could not get documents (collection might not exist yet): {e}")
        return JSONResponse(content=[])

# --- THIS ENDPOINT IS FULLY REPLACED ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat, gets a response, and logs the entire turn to MongoDB.
    """
    # In a full app with user logins, you would manage session_id on the frontend
    # For now, we'll treat each turn as part of a newly identified session for logging
    session_id = str(uuid.uuid4())

    # Generate a response using the bot logic
    bot_response = tutor.generate_response(
        student_question=request.message,
        chat_history=request.history,
        document_source=request.document_source
    )
    
    # --- MongoDB Logging Logic ---
    try:
        # Create a structured log entry using our Pydantic model
        log_entry = ChatLog(
            session_id=session_id,
            timestamp=datetime.datetime.now().isoformat(),
            document_source=request.document_source,
            user_message=request.message,
            bot_response=bot_response,
            chat_history=request.history
        )
        # Insert the log entry (as a dictionary) into the MongoDB collection
        sessions_collection.insert_one(log_entry.dict())
        print(f"Successfully logged chat turn for session {session_id} to MongoDB.")
    except Exception as e:
        print(f"!!! Error logging to MongoDB: {e}")
    # --------------------------------
        
    return JSONResponse(content={"response": bot_response})