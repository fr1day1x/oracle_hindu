from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Step A: Load your text files ---
print("Loading texts...")
loaders = [
    TextLoader("bhagavad_gita.txt", encoding="utf-8"),
    TextLoader("The Upanishads, Part 1.txt", encoding="utf-8"),
]
all_docs = []
for loader in loaders:
    all_docs.extend(loader.load())

print(f"Loaded {len(all_docs)} documents")

# --- Step B: Split into chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks")

# --- Step C: Convert chunks to local vectors and store ---
print("Building vector database locally (this may take a minute)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./hindu_db"  # saves locally to disk
)

print("Done! Database successfully saved to ./hindu_db")