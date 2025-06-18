# api/index.py (TEMPORARY DIAGNOSTIC CODE)

import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

@app.get("/api/test-connection")
def test_connection():
    """
    This endpoint does only one thing: tests the Pinecone API key.
    It will either return a success message or the exact error.
    """
    print("--- Test endpoint hit. Attempting to connect to Pinecone. ---")
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        print("FATAL: PINECONE_API_KEY environment variable is NOT SET.")
        raise HTTPException(status_code=500, detail="Server is missing PINECONE_API_KEY.")

    try:
        print(f"Initializing Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        print("Successfully initialized client. Fetching index list...")
        indexes = pc.list_indexes().names()
        
        print(f"Successfully connected. Found indexes: {indexes}")
        return {
            "status": "SUCCESS",
            "message": "Successfully connected to Pinecone and listed indexes.",
            "found_indexes": indexes
        }
    except Exception as e:
        # This is the most important part: we catch the error and RETURN it.
        error_message = f"Failed to connect to Pinecone. Error: {str(e)}"
        print(f"FATAL: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
def read_root():
    return {"message": "Diagnostic API is running. Hit /api/test-connection to debug."}