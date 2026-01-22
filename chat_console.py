import os
import sys
import warnings
import logging

# --- NUCLEAR CLEANUP BLOCK ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
# -----------------------------

import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
DB_PATH = "./chroma_db"

if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file.")

print("Loading Database and Models...")

# ==========================================
# 2. INITIALIZE MODELS
# ==========================================

embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

if not os.path.exists(DB_PATH):
    print(f"Error: Database folder '{DB_PATH}' not found. Run ingest_database.py first.")
    exit()

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_function,
    client_settings=Settings(anonymized_telemetry=False)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3 
)

# ==========================================
# 3. BUILD THE RAG CHAINS
# ==========================================

context_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

context_prompt = ChatPromptTemplate.from_messages([
    ("system", context_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, context_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If the answer is not in the context, say "I cannot find that information in the provided images." 
Do NOT use outside knowledge. Keep the answer concise.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ==========================================
# 4. CHAT LOOP
# ==========================================
def start_chat():
    print("\n" + "="*50)
    print(f"🤖 Image RAG Chatbot (Model: {MODEL_NAME})")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("="*50)
    
    chat_history = [] 

    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break

        print("Thinking...", end="", flush=True)
        start_time = time.time()

        try:
            response = rag_chain.invoke(
                {"input": query, "chat_history": chat_history}
            )
            
            answer = response["answer"]
            elapsed = time.time() - start_time
            
            print(f"\rAI ({elapsed:.2f}s): {answer}")

            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=answer)
            ])
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    start_chat()