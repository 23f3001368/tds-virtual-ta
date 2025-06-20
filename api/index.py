# api/index.py (Version that produced 1 PASS, 3 ERRORS)

import os
import base64
import json
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# --- Pydantic Models for Output ---
class Link(BaseModel):
    url: str
    text: str

class APIResponse(BaseModel):
    answer: str
    links: List[Link]

# --- FastAPI App Setup ---
app = FastAPI(title="TDS Virtual TA", version="5.2.1-debug", redirect_slashes=False)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- LAZY INITIALIZATION SETUP ---
retrieval_chain = None

def get_retrieval_chain():
    """Initializes and returns the RAG chain, created only once."""
    global retrieval_chain
    if retrieval_chain is not None:
        return retrieval_chain
    print("--- Initializing RAG chain... ---")
    AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
    AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not all([AIPIPE_TOKEN, AIPIPE_BASE_URL, PINECONE_API_KEY]):
        raise RuntimeError("FATAL: Required env vars not set.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=AIPIPE_TOKEN, openai_api_base=AIPIPE_BASE_URL)
    vector_store = PineconeVectorStore.from_existing_index(index_name="tds-virtual-ta", embedding=embeddings)
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0, openai_api_key=AIPIPE_TOKEN, openai_api_base=AIPIPE_BASE_URL)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    system_prompt = ("You are a helpful virtual TA for 'Tools in Data Science'. "
                   "Answer the student's question based *only* on the provided context. "
                   "Your answer must be concise and accurate. Do not mention 'the context'. "
                   "If the context is insufficient, state that you could not find a definitive answer.\n\n"
                   "CONTEXT:\n{context}")
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("--- RAG chain initialized. ---")
    return retrieval_chain

def get_image_description(base64_image: str) -> str:
    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200, openai_api_key=os.getenv("AIPIPE_TOKEN"), openai_api_base=os.getenv("AIPIPE_BASE_URL"))
    vision_prompt = "Analyze this screenshot. Concisely describe any code, commands, or errors shown."
    msg = vision_llm.invoke([HumanMessage(content=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}"}}])])
    return msg.content

@app.post("/api/", response_model=APIResponse)
async def ask_question(request: Request):
    # This uses the original, stricter JSON parser.
    # It will fail on requests with a trailing comma.
    try:
        body_str = (await request.body()).decode('utf-8')
        data = json.loads(body_str)
        question = data.get("question")
        if not question: raise ValueError("'question' field is missing.")
        image = data.get("image")
    except Exception as e:
        # This is where the error happens for the 3 failing tests.
        raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {e}")

    try:
        chain = get_retrieval_chain()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not initialize RAG chain: {e}")

    if image and isinstance(image, str):
        try:
            image_description = get_image_description(image)
            question = f"Question: '{question}'. Screenshot context: {image_description}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

    try:
        response = await chain.ainvoke({"input": question})
        llm_output_text = response.get('answer', "No answer could be generated.")
        retrieved_docs = response.get('context', [])
        
        # This logic parses a potentially double-encoded JSON response from the LLM.
        final_answer = llm_output_text
        final_links_data = []
        try:
            parsed_data = json.loads(llm_output_text)
            if isinstance(parsed_data, dict) and "answer" in parsed_data:
                final_answer = parsed_data["answer"]
                final_links_data = parsed_data.get("links", [])
        except json.JSONDecodeError:
            pass

        if final_links_data:
            links = [Link(**link_data) for link_data in final_links_data]
        else:
            links = []
            unique_urls = set()
            for doc in retrieved_docs:
                url = doc.metadata.get('source')
                if url and url not in unique_urls:
                    title = doc.metadata.get('title', 'Source Link')
                    links.append(Link(url=url, text=title[:90]))
                    unique_urls.add(url)
        
        return APIResponse(answer=final_answer, links=links[:3])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {e}")

@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA is running."}