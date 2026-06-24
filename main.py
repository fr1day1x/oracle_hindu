from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import chromadb
import io
import urllib.parse
from fastapi.responses import StreamingResponse
from groq import Groq
import edge_tts
import io
import urllib.parse
from fastapi.responses import StreamingResponse

load_dotenv()

app = FastAPI()

# Allow the frontend browser to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 0. FRONTEND ROUTING ─────────────────────────────────
# Serve the static directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve the HTML UI on the base URL
@app.get("/")
async def serve_frontend():
    return FileResponse("app/static/index.html")

# ── 1. CLOUD-ONLY INITIALIZATION ────────────────────────
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Connect to the freshly baked database
try:
    chroma_client = chromadb.PersistentClient(path="./hindu_db")
    collection = chroma_client.get_collection(name="scriptures")
except Exception as e:
    print(f"Database error: Ensure ./hindu_db exists. {e}")

# ── 2. DATA MODELS ──────────────────────────────────────
class Question(BaseModel):
    question: str

# ── 3. THE API ENDPOINT ─────────────────────────────────
@app.post("/ask")
async def ask_oracle(q: Question):
    print(f"Received question from frontend: {q.question}")
    
    # Step A: Get the mathematical query vector from NVIDIA
    try:
        payload = {
            "input": [q.question],
            "model": "nvidia/nv-embedqa-e5-v5",
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END"
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_KEY}",
            "Content-Type": "application/json"
        }
        res = requests.post("https://integrate.api.nvidia.com/v1/embeddings", headers=headers, json=payload)
        res.raise_for_status()
        query_vector = res.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"NVIDIA Embedding Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to embed question")

    # Step B: Search the Chroma database
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        context_text = "\n\n".join(results['documents'][0])
    except Exception as e:
        print(f"Chroma Search Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to search scriptures")

    # Step C: Generate the Sage's response via Groq
    system_persona = (
        "You are an ancient, enlightened Hindu sage. Speak in poetic, mystical, and profoundly calm language. "
        "Address the user gently as 'Seeker'. Draw your wisdom strictly from the sacred text provided below, "
        "but weave it into spiritual counsel. Keep your answer to 3 brief sentences. Do not use markdown.\n\n"
        f"Context:\n{context_text}"
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_persona},
                {"role": "user", "content": q.question}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.35,
        )
        reply_text = chat_completion.choices[0].message.content.strip()
        
        # URL encode the text to safely pass it to the TTS endpoint
        encoded_text = urllib.parse.quote(reply_text)
        
        return {
            "answer": reply_text,
            "audio_url": f"/tts?text={encoded_text}"
        }
        
    except Exception as e:
        print(f"Groq Generation Error: {e}")
        raise HTTPException(status_code=500, detail="The Oracle is currently meditating.")

# ── 4. LIGHTWEIGHT NEURAL VOICE GENERATOR ───────────────────────────
@app.get("/tts")
async def stream_audio(text: str):
    # This bypasses the heavy NVIDIA SDK and uses Microsoft's lightweight neural engine
    communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural", rate="-9%")
    audio_data = b""
    
    # Generate the audio in system RAM to avoid hard drive writes
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
            
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")