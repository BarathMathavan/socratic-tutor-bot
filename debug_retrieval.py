# debug_retrieval.py
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# The question we are testing
USER_QUESTION = "what is the fine amount for speeding"

# The document we are searching within
DOCUMENT_SOURCE = "Motor_Vehicles_Act_1988_with_Amendments_2019.pdf"

# Number of results to retrieve
NUM_RESULTS = 5 # Let's get a few more than usual to see a wider context

# --- SCRIPT LOGIC ---
print(f"--- Starting Retrieval Debugger ---")
print(f"Searching for: '{USER_QUESTION}'")
print(f"Inside document: '{DOCUMENT_SOURCE}'\n")

try:
    # 1. Initialize models and database connection
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="socratic_collection")

    # 2. Embed the user's question
    print("Step 1: Creating embedding for the user's question...")
    query_embedding = embedding_model.encode([USER_QUESTION])
    print("Embedding created successfully.\n")

    # 3. Query the database with the metadata filter
    print(f"Step 2: Querying the database for the top {NUM_RESULTS} most relevant chunks...")
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=NUM_RESULTS,
        where={"source": DOCUMENT_SOURCE}
    )
    print("Query complete.\n")

    # 4. Analyze and print the results
    if not results or not results['documents'] or not results['documents'][0]:
        print("---!!! CRITICAL FAILURE !!!---")
        print("The query returned ZERO results. This means either:")
        print("  a) The database is empty or does not contain this document source.")
        print("  b) There was a fundamental error during ingestion.")
        print("\nACTION: Delete your 'db' folder and run 'python ingestion.py' again carefully.")
    else:
        print("--- RETRIEVAL RESULTS ---")
        for i, doc in enumerate(results['documents'][0]):
            print(f"\n--- Result {i+1} ---")
            print(doc)
            print("-" * 20)

except Exception as e:
    print(f"\n---!!! AN ERROR OCCURRED !!!---")
    print(f"Error: {e}")