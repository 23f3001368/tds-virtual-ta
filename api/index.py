# api/index.py
import os
import base64
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

class APIRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class APIResponse(BaseModel):
    answer: str
    links: List[Link]

app = FastAPI(title="TDS Virtual TA", version="2.0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

retrieval_chain = None
aipipe_config = {}

@app.on_event("startup")
def load_models_and_index():
    global retrieval_chain, aipipe_config
    api_token = os.getenv("AIPIPE_TOKEN")
    base_url = os.getenv("AIPIPE_BASE_URL")
    if not api_token or not base_url:
        raise RuntimeError("FATAL: AIPIPE_TOKEN and AIPIPE_BASE_URL env variables are not set.")
    
    aipipe_config = {"api_key": api_token, "base_url": base_url}

    vector_store_path = "data/faiss_index"
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"FATAL: Vector store not found. Run 'python scripts/preprocess.py' locally.")
    
    embeddings = OpenAIEmbeddings(
        # CORRECTED: Remove prefix
        model="text-embedding-3-small", 
        openai_api_key=aipipe_config["api_key"], 
        openai_api_base=aipipe_config["base_url"]
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatOpenAI(
        # CORRECTED: Remove prefix
        model="gpt-3.5-turbo-0125", 
        temperature=0.0, 
        openai_api_key=aipipe_config["api_key"], 
        openai_api_base=aipipe_config["base_url"]
    )
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    system_prompt = (
        "You are a helpful virtual TA for the 'Tools in Data Science' course. "
        "Answer the student's question based *only* on the provided context. "
        "Your answer must be concise, accurate, and directly address the question. "
        "Do not mention 'the context' or 'the provided material' in your answer. "
        "If the context does not contain the answer, state that you could not find a definitive answer in the course materials."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("--- TDS Virtual TA is ready to answer questions. ---")

def get_image_description(base64_image: str) -> str:
    vision_llm = ChatOpenAI(
        # CORRECTED: Remove prefix
        model="gpt-4o-mini", 
        temperature=0, 
        max_tokens=200, 
        openai_api_key=aipipe_config["api_key"], 
        openai_api_base=aipipe_config["base_url"]
    )
    vision_prompt = "Analyze this screenshot from a student. Concisely describe any code, commands, UI elements, or error messages shown. This description will be used as context to answer a question."
    msg = vision_llm.invoke([HumanMessage(content=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}"}}])])
    return msg.content

@app.post("/api/", response_model=APIResponse)
async def ask_question(request: APIRequest):
    if not retrieval_chain:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")

    question = request.question
    
    if request.image:
        try:
            image_description = get_image_description(request.image)
            question = f"Question: '{question}'. This question is related to a screenshot showing the following: {image_description}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

    try:
        response = retrieval_chain.invoke({"input": question})
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
        raise HTTPException(status_code=500, detail="An error occurred while generating the answer.")

@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA is running. POST to /api/ with your question."}