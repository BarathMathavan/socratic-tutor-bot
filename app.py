# ==============================================================================
# 1. IMPORTS
# ==============================================================================
# --- Flask and Web-related Imports ---
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- Standard Library Imports ---
import os
import datetime

# --- Third-party Library Imports ---
from dotenv import load_dotenv

# --- Local Application Imports ---
from bot_logic import SocraticTutor
from ingestion import process_and_store_pdf
import chromadb

# ==============================================================================
# 2. INITIAL SETUP AND CONFIGURATION
# ==============================================================================
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Configuration for File Uploads ---
# Define the folder where uploaded PDFs will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# A secret key is required by Flask to use "flash" messages (for success/error notifications)
app.config['SECRET_KEY'] = 'a-super-secret-and-unique-key'

# ==============================================================================
# 3. API KEY AND TUTOR INITIALIZATION
# ==============================================================================
# Load the Google API key from the .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables!")

# Initialize the SocraticTutor once when the app starts
tutor = SocraticTutor(api_key=API_KEY)

# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================

@app.route('/get_documents', methods=['GET'])
def get_documents():
    """Lists all unique document sources from the database."""
    try:
        client = chromadb.PersistentClient(path="db")
        collection = client.get_or_create_collection(name="socratic_collection")
        
        # Get all items and extract the 'source' from metadata
        items = collection.get()
        if not items or not items['metadatas']:
            return jsonify([])
            
        sources = sorted(list(set(item['source'] for item in items['metadatas'])))
        return jsonify(sources)
    except Exception as e:
        print(f"Error getting documents: {e}")
        return jsonify({"error": "Could not retrieve document list"}), 500


@app.route('/')
def index():
    """Serves the main HTML chat page."""
    return render_template('index.html')

# @app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat request, including the selected document."""
    data = request.json
    user_message = data.get('message')
    chat_history = data.get('history', [])
    document_source = data.get('document_source') # Get the selected document

    if not user_message or not document_source:
        return jsonify({"error": "Message or document source missing"}), 400

    # Pass the selected document to the bot logic
    bot_response = tutor.generate_response(user_message, chat_history, document_source)


    # --- ADDED: Logging Logic ---
    # Append the conversation turn to a text file for instructor review
    try:
        with open("chat_log.txt", "a", encoding="utf-8") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"--- Session Turn at {timestamp} ---\n")
            log_file.write(f"User: {user_message}\n")
            log_file.write(f"Bot: {bot_response}\n\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")
    # --- END of Logging Logic ---

    return jsonify({"response": bot_response})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handles instructor PDF uploads and processing."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Now, run the ingestion process on the newly uploaded file
            try:
                process_and_store_pdf(filepath)
                flash(f"'{filename}' was successfully uploaded and processed!", 'success')
            except Exception as e:
                flash(f"An error occurred during processing: {e}", 'error')

            return redirect(url_for('upload_file'))

    return render_template('upload.html')

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5000)