# main.py (Final "Hackathon Mode" Version)

# --- 1. Imports ---
import os
import datetime
import uuid
import time
import urllib.parse
import google.generativeai as genai
from typing import List, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from pymongo import MongoClient

# --- 2. Local Application Imports ---
from bot_logic import SocraticTutor
from ingestion import simple_ingestion, extract_text_from_pdf
# --- 3. Initial Application Setup ---
load_dotenv()
app = FastAPI(title="Socratic Tutor Bot API")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 4. Database and AI Service Connections ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY: raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=API_KEY)

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI: raise ValueError("MONGODB_URI not found in .env file!")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["socratic_tutor_db"]
sessions_collection = db["chat_sessions"]

tutor = SocraticTutor(api_key=API_KEY)

# --- 5. Pydantic Models for API Data Validation ---
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]]
    document_source: str

# --- 6. Helper Function for Proactive Welcome Message ---
def generate_starting_points(pdf_path: str, max_retries: int = 3) -> dict:
    print(f"--- Generating starting points for {os.path.basename(pdf_path)} ---")
    full_text = extract_text_from_pdf(pdf_path)
    if len(full_text) > 200000: full_text = full_text[:200000]

    model = genai.GenerativeModel('gemini-pro-latest')
    prompt = f"""
    Based on the following document text, please do two things:
    1. Summarize the top 5-7 main topics discussed. Present this as a simple, comma-separated list.
    2. Generate 3 open-ended, engaging questions a student might have about this text.
    DOCUMENT TEXT: "{full_text}"
    ---
    Provide your response in a structured format. Example:
    TOPICS: Topic A, Topic B, Topic C
    QUESTIONS:
    1. What is the significance of [Concept X]?
    2. How does [Event Y] affect [Outcome Z]?
    3. Can you explain the process of [Mechanism W]?
    """
    for attempt in range(max_retries):
        try:
            request_options = {'timeout': 300}
            response = model.generate_content(prompt, request_options=request_options)
            text = response.text
            topics_line = next((line for line in text.split('\n') if line.startswith("TOPICS:")), "TOPICS: General Information")
            questions_section = text.split("QUESTIONS:")[-1]
            topics = topics_line.replace("TOPICS:", "").strip()
            questions = [q.strip().lstrip('0123456789. ') for q in questions_section.split('\n') if q.strip()]
            return {"topics": topics, "questions": questions[:3]}
        except Exception as e:
            print(f"--> Starting points generation attempt {attempt + 1}/{max_retries} failed.")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"    ...retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("--> Max retries reached for starting points. Falling back to default.")
                return {"topics": "General discussion", "questions": []}
    return {"topics": "General discussion", "questions": []}

# --- 7. API Endpoints ---
@app.get("/", summary="Serve the main chat interface")
async def serve_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", summary="Serve the PDF upload page")
async def serve_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", summary="Instantly handle PDF upload and queue processing")
async def handle_file_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Queue BOTH ingestion tasks to run in the background
    background_tasks.add_task(simple_ingestion, filepath)
    
    # Instantly redirect with a "processing" status
    redirect_url = f"/?doc={urllib.parse.quote(file.filename)}&status=processing"
    return RedirectResponse(url=redirect_url, status_code=303)

@app.get("/get_documents", summary="Get a list of all processed documents")
async def get_documents_list():
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        try:
            qdrant_client.get_collection(collection_name="socratic_collection")
        except (UnexpectedResponse, Exception):
            return JSONResponse(content=[])
        scrolled_points, _ = qdrant_client.scroll(
            collection_name="socratic_collection", limit=10000,
            with_payload=["source"], with_vectors=False
        )
        if not scrolled_points: return JSONResponse(content=[])
        sources = sorted(list(set(point.payload["source"] for point in scrolled_points)))
        return JSONResponse(content=sources)
    except Exception as e:
        print(f"An unexpected error occurred in get_documents: {e}")
        return JSONResponse(content=[])

@app.get("/get_starting_points/{filename}", summary="Generate and get welcome topics/questions for a doc")
async def get_starting_points_for_doc(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    starting_points = generate_starting_points(filepath)
    return JSONResponse(content=starting_points)

@app.post("/chat", summary="Process a user chat message")
async def chat_endpoint(request: ChatRequest):
    session_id = str(uuid.uuid4())
    bot_response = tutor.generate_response(
        student_question=request.message,
        chat_history=request.history,
        document_source=request.document_source
    )
    try:
        log_document = {
            "session_id": session_id, "timestamp": datetime.datetime.now().isoformat(),
            "document_source": request.document_source, "user_message": request.message,
            "bot_response": bot_response, "chat_history": request.history
        }
        sessions_collection.insert_one(log_document)
        print(f"Successfully logged chat turn to MongoDB.")
    except Exception as e:
        print(f"!!! Error logging to MongoDB: {e}")
    return JSONResponse(content={"response": bot_response})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)