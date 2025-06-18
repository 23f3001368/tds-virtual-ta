# api/index.py (TEMPORARY DIAGNOSTIC CODE)

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# We only import the Pinecone library for this test
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# Add CORS middleware so the evaluator website can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/test")
def run_connection_test():
    """
    This simple endpoint will try to connect to Pinecone.
    It is designed to ALWAYS return a JSON response, either success or failure.
    This will tell us exactly what the error is.
    """
    print("--- DIAGNOSTIC: Test endpoint has been called. ---")

    # 1. Check if the Environment Variable exists on Vercel
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("--- DIAGNOSTIC: FAILURE! The PINECONE_API_KEY environment variable was not found on the server.")
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error: PINECONE_API_KEY environment variable is not set."
        )

    print("--- DIAGNOSTIC: PINECONE_API_KEY found. Proceeding to connect. ---")

    # 2. Try to connect to Pinecone and list indexes
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        print("--- DIAGNOSTIC: Pinecone client initialized. Listing indexes... ---")
        
        # This is the actual network call that might be failing
        list_of_indexes = pc.list_indexes().names()
        
        success_message = f"Successfully connected to Pinecone. Found indexes: {list_of_indexes}"
        print(f"--- DIAGNOSTIC: SUCCESS! {success_message} ---")
        
        return {
            "status": "SUCCESS",
            "message": success_message
        }
    except Exception as e:
        # If ANY error happens during the connection, catch it and return it
        error_message = f"An error occurred while connecting to Pinecone. ERROR: {str(e)}"
        print(f"--- DIAGNOSTIC: FAILURE! {error_message} ---")
        raise HTTPException(
            status_code=500,
            detail=error_message
        )

@app.get("/")
def read_root():
    return {"message": "Diagnostic API is running. Hit /api/test to debug."}
    