# test_mongo_connection.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

print("--- MongoDB Connection Test ---")

# Load the connection string from your .env file
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    print("!!! ERROR: MONGODB_URI not found in your .env file. Please check the file.")
else:
    print("Found MONGODB_URI. Attempting to connect...")
    try:
        # Try to create a client and connect
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000) # 20 second timeout
        
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        
        print("\n*** SUCCESS! ***")
        print("Successfully connected to the MongoDB server.")
        print("The problem is likely not your network, but something in the main app.")

    except ConnectionFailure as e:
        print("\n---!!! CONNECTION FAILED !!!---")
        print("This confirms the issue is with your network connection or firewall.")
        print("The error is the same 'timeout' you saw before.")
        print(f"Details: {e}")
        print("\nPlease proceed to the 'Plan B' solution below.")
    except Exception as e:
        print(f"\n---!!! AN UNEXPECTED ERROR OCCURRED !!!---")
        print(f"Details: {e}")