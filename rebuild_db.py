import os
import shutil
import time
import requests
from dotenv import load_dotenv
import chromadb

load_dotenv()

# 1. Authenticate with your existing NVIDIA Key
nvidia_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_key:
    raise ValueError("Missing NVIDIA_API_KEY in your .env file. Please add it.")

API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
headers = {
    "Authorization": f"Bearer {nvidia_key}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# 2. Database Purge and Setup
DB_PATH = "./hindu_db"
if os.path.exists(DB_PATH):
    print(f"Purging old database at {DB_PATH}...")
    shutil.rmtree(DB_PATH)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.create_collection(name="scriptures")

# 3. Dynamic File Parser Loop
DATA_DIR = "."  

documents = []
metadata = []
ids = []
chunk_counter = 0

print("Scanning root directory for scripture files...")

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt") and filename != "requirements.txt":
        file_path = os.path.join(DATA_DIR, filename)
        print(f"Processing: {filename}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            for i, para in enumerate(paragraphs):
                documents.append(para)
                metadata.append({"source": filename, "chunk": i})
                ids.append(f"{filename}_chunk_{chunk_counter}")
                chunk_counter += 1

# 4. The NVIDIA API Query Function
def query_nvidia(texts):
    payload = {
      "input": texts,
      "model": "nvidia/nemotron-3-embed-1b",
      "input_type": "passage", # CRITICAL: Tags this as database text
      "encoding_format": "float",
      "truncate": "END"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json().get("data", [])
        # Extract the pure float arrays
        return [item["embedding"] for item in data]
    else:
        raise Exception(f"NVIDIA API Error {response.status_code}: {response.text}")

# 5. Batch Push
if documents:
    print(f"Generating embeddings for {len(documents)} chunks via NVIDIA API...")
    
    raw_embeddings = []
    # NVIDIA comfortably handles batches of 100
    API_BATCH_SIZE = 100 
    
    for i in range(0, len(documents), API_BATCH_SIZE):
        batch_docs = documents[i : i + API_BATCH_SIZE]
        
        print(f"Processing chunks {i} to {i + len(batch_docs)} of {len(documents)}...")
        
        # Fetch vectors from NVIDIA
        embeddings = query_nvidia(batch_docs)
        raw_embeddings.extend(embeddings)
            
        # A tiny 1-second safety pause to respect general server traffic
        time.sleep(1)
    
    print("Pushing raw vectors into ChromaDB in database-safe batches...")
    
    # ChromaDB transaction limit is 5461; we use 4000 to be completely safe
    DB_BATCH_SIZE = 4000 
    
    for i in range(0, len(documents), DB_BATCH_SIZE):
        end_idx = i + DB_BATCH_SIZE
        batch_docs = documents[i:end_idx]
        batch_embs = raw_embeddings[i:end_idx]
        batch_meta = metadata[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        print(f"Writing database batch: records {i} to {min(end_idx, len(documents))}...")
        collection.add(
            documents=batch_docs,
            embeddings=batch_embs, 
            metadatas=batch_meta,
            ids=batch_ids
        )
        
    print(f"Database successfully rebuilt! Total indexed vectors: {collection.count()}")
else:
    print("No text data found. Ingestion aborted.")