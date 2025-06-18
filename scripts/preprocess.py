# scripts/preprocess.py
import os
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --- Configuration ---
DISCOURSE_DATA_DIR = "discourse_json"
TDS_PAGES_DIR = "tds_pages_md"
PINECONE_INDEX_NAME = "tds-virtual-ta"
# We will process and upload in batches of this size to avoid API limits.
UPLOAD_BATCH_SIZE = 100

# --- Data Parsing Functions (UNCHANGED) ---
def parse_discourse_data() -> list:
    print(f"--- Parsing Discourse data from '{DISCOURSE_DATA_DIR}' ---")
    all_posts = []
    base_url = "https://discourse.onlinedegree.iitm.ac.in"
    if not os.path.isdir(DISCOURSE_DATA_DIR):
        raise FileNotFoundError(f"Directory not found: '{DISCOURSE_DATA_DIR}'.")
    for filename in os.listdir(DISCOURSE_DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DISCOURSE_DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            topic_title = data.get('title', 'Untitled Topic')
            post_stream = data.get('post_stream', {})
            for post in post_stream.get('posts', []):
                soup = BeautifulSoup(post.get('cooked', ''), 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
                post_url = base_url + post.get('post_url', '')
                if content:
                    all_posts.append({"url": post_url, "title": f"Post in: {topic_title}", "content": content})
    print(f"Parsed {len(all_posts)} posts from Discourse.")
    return all_posts

def parse_course_content_data() -> list:
    print(f"--- Parsing Course Content data from '{TDS_PAGES_DIR}' ---")
    all_pages = []
    if not os.path.isdir(TDS_PAGES_DIR):
        raise FileNotFoundError(f"Directory not found: '{TDS_PAGES_DIR}'.")
    for filename in os.listdir(TDS_PAGES_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(TDS_PAGES_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                title = content.split('\n')[0].replace('---', '').replace('title:', '').strip() or os.path.splitext(filename)[0].replace('_', ' ')
                all_pages.append({"url": f"https://tds.s-anand.net/#/{filename.replace('.md', '')}", "title": title, "content": content})
    print(f"Parsed {len(all_pages)} pages from course content.")
    return all_pages

def run_preprocessing():
    """
    Processes data and uploads it to Pinecone in manageable batches.
    """
    # --- 1. Parsing & Chunking ---
    all_data = parse_discourse_data() + parse_course_content_data()
    if not all_data: return

    documents = [Document(page_content=item['content'], metadata={'source': item['url'], 'title': item['title']}) for item in all_data if item['content']]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks.")

    # --- 2. Initialize Clients ---
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
    AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL")
    if not all([PINECONE_API_KEY, AIPIPE_TOKEN, AIPIPE_BASE_URL]):
        raise ValueError("One or more required environment variables are missing.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=AIPIPE_TOKEN, openai_api_base=AIPIPE_BASE_URL)
    
    # --- 3. Create or Connect to Index ---
    print(f"Checking for Pinecone index '{PINECONE_INDEX_NAME}'...")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
        print("Index created.")
    else:
        print("Index already exists.")

    # --- 4. THE FIX: Upload in Batches ---
    print(f"\nUploading documents to Pinecone in batches of {UPLOAD_BATCH_SIZE}...")
    
    # First, get a client pointed at the specific index
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    for i in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        batch = chunks[i:i + UPLOAD_BATCH_SIZE]
        print(f"  - Uploading batch {i//UPLOAD_BATCH_SIZE + 1} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)
    
    print("\n--- Preprocessing and upload complete! ---")
    print("Your Pinecone index is now populated.")

if __name__ == '__main__':
    run_preprocessing()