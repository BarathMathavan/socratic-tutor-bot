# list_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=API_KEY)

print("Listing all available models...\n")

# List all models and check which ones support 'generateContent'
try:
    for m in genai.list_models():
        # Check if the 'generateContent' method is supported
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"An error occurred: {e}")
    print("\n--- DEBUGGING TIPS ---")
    print("1. Double-check that your API key in the .env file is correct and has no extra spaces.")
    print("2. Ensure the 'Generative Language API' or 'Vertex AI API' is enabled in your Google Cloud project.")
    print("3. Make sure your project has billing enabled (many APIs require this, even for the free tier).")