# ==============================================================================
# SECTION 1: IMPORTS & SYSTEM CONFIGURATION
# ==============================================================================
import os
import shutil
import warnings
import logging
import time
from typing import List

# --- NUCLEAR CLEANUP BLOCK (Suppress Logs) ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
# ---------------------------------------------

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from google import genai
from PIL import Image

# ==============================================================================
# SECTION 2: GLOBAL SETUP (KEYS, DB, & CLIENTS)
# ==============================================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
DB_PATH = "./chroma_db"
TEMP_IMAGE_FOLDER = "./temp_images"

if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file.")

# 1. Initialize Gemini Client (For Image Analysis)
client = genai.Client(api_key=GOOGLE_API_KEY)

# 2. Initialize Vector DB Connection
embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_function,
    client_settings=Settings(anonymized_telemetry=False)
)

# 3. In-Memory Session Storage (RAM Memory)
chat_sessions = {}

# ==============================================================================
# SECTION 3: AI LOGIC (THE "BRAIN")
# ==============================================================================

def analyze_image_with_retry(image, retries=3):
    """
    Sends an image to Gemini Vision. Retries if the server is overloaded.
    """
    prompt = "Analyze this image in detail. Extract text, layout, objects, and data."
    
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, 
                contents=[prompt, image]
            )
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                time.sleep(2)
                continue
            raise e
    raise Exception("Gemini API is too busy. Please try again.")

def buildragchain():
    """Constructs the LangChain pipeline for answering questions."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Context-aware prompt for rephrasing query with history
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the user question to be a standalone question, based on the chat history:\n{chat_history}\n{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # History-aware retriever (replaces old pattern)
    history_retriever = create_history_aware_retriever(
        llm, retriever, context_prompt
    )

    # QA prompt for answering from context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the following context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Document chain and full retrieval chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_retriever, qa_chain)

    return retrieval_chain

# ==============================================================================
# SECTION 4: API ENDPOINTS (THE INTERFACE)
# ==============================================================================

app = FastAPI(title="Image RAG API")

# Allow Frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    session_id: str
    message: str

@app.get("/")
def health_check():
    return {"status": "online", "model": MODEL_NAME}

@app.post("/ingest")
async def upload_images(files: List[UploadFile] = File(...)):
    """Uploads files -> Analyzes with Gemini -> Saves to DB"""
    if not os.path.exists(TEMP_IMAGE_FOLDER):
        os.makedirs(TEMP_IMAGE_FOLDER)

    count = 0
    errors = []

    for file in files:
        try:
            # 1. Save File
            temp_path = os.path.join(TEMP_IMAGE_FOLDER, file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # 2. Analyze
            img = Image.open(temp_path)
            response = analyze_image_with_retry(img)

            # 3. Index
            doc = Document(page_content=response.text, metadata={"source": file.filename})
            vectorstore.add_documents([doc])
            
            # 4. Cleanup
            os.remove(temp_path)
            count += 1
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    return {"message": f"Processed {count} images", "errors": errors}

@app.post("/chat")
async def chat(request: ChatInput):
    """Manages chat history and generates answers"""
    sid = request.session_id
    if sid not in chat_sessions: chat_sessions[sid] = []

    try:
        chain = buildragchain()
        response = chain.invoke({"input": request.message, "chat_history": chat_sessions[sid]})
        
        # Save to memory
        chat_sessions[sid].extend([
            HumanMessage(content=request.message),
            AIMessage(content=response["answer"])
        ])
        
        return {"response": response["answer"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"🚀 Starting API on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)