# api/index.py (FINAL APPLICATION CODE)

import os
import base64
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# --- Pydantic Models ---
class APIRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class APIResponse(BaseModel):
    answer: str
    links: List[Link]

app = FastAPI(title="TDS Virtual TA", version="4.1.0-lazy-init")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- LAZY INITIALIZATION SETUP ---
retrieval_chain = None

def get_retrieval_chain():
    """
    Initializes and returns the RAG chain. Uses a global variable
    to ensure it's only created once.
    """
    global retrieval_chain
    if retrieval_chain is not None:
        print("--- Using existing RAG chain. ---")
        return retrieval_chain

    print("--- Initializing RAG chain for the first time... ---")
    
    # Load all required API Keys
    AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
    AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "tds-virtual-ta"

    if not all([AIPIPE_TOKEN, AIPIPE_BASE_URL, PINECONE_API_KEY]):
        raise RuntimeError("FATAL: One or more required environment variables are not set on the server.")

    # Initialize all clients
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=AIPIPE_TOKEN, openai_api_base=AIPIPE_BASE_URL)
    vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0, openai_api_key=AIPIPE_TOKEN, openai_api_base=AIPIPE_BASE_URL)
    
    # Build RAG Chain
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    system_prompt = (
        "You are a helpful virtual TA for the 'Tools in Data Science' course. "
        "Answer the student's question based *only* on the provided context. "
        "Your answer must be concise and accurate. Do not mention 'the context'. "
        "If the context does not contain the answer, state that you could not find a definitive answer."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Store the chain in the global variable and return it
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("--- RAG chain initialized successfully. ---")
    return retrieval_chain

def get_image_description(base64_image: str) -> str:
    vision_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0, 
        max_tokens=200, 
        openai_api_key=os.getenv("AIPIPE_TOKEN"),
        openai_api_base=os.getenv("AIPIPE_BASE_URL")
    )
    vision_prompt = "Analyze this screenshot. Concisely describe any code, commands, or errors shown."
    msg = vision_llm.invoke([HumanMessage(content=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}"}}])])
    return msg.content

@app.post("/api/", response_model=APIResponse)
async def ask_question(request: APIRequest):
    try:
        chain = get_retrieval_chain()
    except Exception as e:
        print(f"FATAL: Could not initialize RAG chain: {e}")
        raise HTTPException(status_code=500, detail=f"Could not initialize RAG chain: {e}")

    question = request.question
    if request.image:
        try:
            image_description = get_image_description(request.image)
            question = f"Question: '{question}'. This question relates to a screenshot showing: {image_description}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

    try:
        response = await chain.ainvoke({"input": question})
        answer_text = response.get('answer', "No answer could be generated.")
        retrieved_docs = response.get('context', [])
        
        links = []
        unique_urls = set()
        for doc in retrieved_docs:
            url = doc.metadata.get('source')
            if url and url not in unique_urls:
                title = doc.metadata.get('title', 'Source Link')
                links.append(Link(url=url, text=title[:90] + '...' if len(title) > 90 else title))
                unique_urls.add(url)
        
        return APIResponse(answer=answer_text, links=links[:3])

    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during question processing.")

@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA is running. POST to /api/ with your question."}