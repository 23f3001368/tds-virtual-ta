# scripts/preprocess.py
import os
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()

# --- Configuration ---
DISCOURSE_DATA_DIR = "discourse_json"
TDS_PAGES_DIR = "tds_pages_md"
DATA_PATH = "data/processed_data.json"
VECTOR_STORE_PATH = "data/faiss_index"

def parse_discourse_data() -> list:
    print(f"--- Parsing Discourse data from '{DISCOURSE_DATA_DIR}' ---")
    all_posts = []
    base_url = "https://discourse.onlinedegree.iitm.ac.in"
    if not os.path.isdir(DISCOURSE_DATA_DIR):
        print(f"FATAL: Directory not found: '{DISCOURSE_DATA_DIR}'.")
        return []
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
        print(f"FATAL: Directory not found: '{TDS_PAGES_DIR}'.")
        return []
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
    discourse_data = parse_discourse_data()
    course_content_data = parse_course_content_data()
    all_data = discourse_data + course_content_data
    if not all_data:
        print("FATAL: No data was processed. Halting.")
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_data)} total documents to {DATA_PATH}")

    documents = [Document(page_content=item['content'], metadata={'source': item['url'], 'title': item['title']}) for item in all_data if item['content']]
    print("\n--- Splitting Documents into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    print("\n--- Creating Embeddings via Proxy ---")
    api_token = os.getenv("AIPIPE_TOKEN")
    base_url = os.getenv("AIPIPE_BASE_URL")
    if not api_token or not base_url:
        raise ValueError("AIPIPE_TOKEN and AIPIPE_BASE_URL must be set in .env file.")
        
    embeddings = OpenAIEmbeddings(
        # CORRECTED: Remove the 'openai/' prefix. The proxy expects the direct model name.
        model="text-embedding-3-small",
        openai_api_key=api_token,
        openai_api_base=base_url
    )
    
    print("This may take a few minutes...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"--- Vector store saved to {VECTOR_STORE_PATH} ---")
    print("\nPreprocessing complete! Commit the 'data' directory and deploy.")

if __name__ == '__main__':
    run_preprocessing()