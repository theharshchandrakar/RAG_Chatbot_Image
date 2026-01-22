RAG Chatbot Image🤖🖼️

A powerful backend API that allows users to "chat" with their images. This project uses Retrieval-Augmented Generation (RAG) to analyze images, index their visual content, and enable context-aware Q&A using Google's Gemini 2.5 Flash model.

🔗 Live API URL: [Add your Render/Cloud URL here] (or run locally on http://127.0.0.1:8000)

🚀 Features

Multimodal Ingestion: Uploads images (JPG/PNG), extracts text/objects/layout using Gemini Vision, and creates vector embeddings.

Context-Aware Chat: Remembers conversation history (e.g., "What is the total?" -> "Is that in USD?") using LangChain.

Persistent Memory: Stores vector data in a local ChromaDB instance (persists across restarts locally).

FastAPI Backend: Robust, high-performance API with CORS enabled for easy frontend integration.

Self-Healing: Built-in retry logic to handle API rate limits (503 errors) gracefully.

🛠️ Tech Stack

Language: Python 3.10+

Core Logic: LangChain

AI Engine: Google Gemini 2.5 Flash (via google-genai SDK)

Vector Database: ChromaDB

API Framework: FastAPI

Dependency Management: uv (recommended) or pip

📦 Installation & Setup

1. Clone the Repository

git clone [https://github.com/theharshchandrakar/RAG_Chatbot_Image.git](https://github.com/theharshchandrakar/RAG_Chatbot_Image.git)
cd RAG_Chatbot_Image


2. Install Dependencies

It is recommended to use a virtual environment.

# Using standard pip
pip install -r requirements.txt

# OR using uv (faster)
uv pip install -r requirements.txt


3. Configure Environment Variables

Create a .env file in the root directory and add your Google API key:

GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash


🏃‍♂️ Running the Server

Start the API server locally:

python api_server.py


You will see: 🚀 Starting API on port 8000...

🔌 API Endpoints

Once the server is running, access the automatic interactive documentation at:
http://127.0.0.1:8000/docs

1. Upload Images (Ingestion)

Endpoint: POST /ingest

Description: Uploads one or multiple images to be indexed.

Body: files (Multipart/Form-Data)

2. Chat (Retrieval)

Endpoint: POST /chat

Description: Ask a question about the uploaded images.

Body:

{
  "session_id": "unique_session_id",
  "message": "What is the total amount in the invoice?"
}


📂 Project Structure

api_server.py: The main entry point. Contains the FastAPI app, DB connection, and RAG logic.

requirements.txt: List of Python dependencies.

chroma_db/: (Generated) Local folder storing the vector database.

temp_images/: (Generated) Temporary folder for processing uploads.

🤝 Integration Guide

To connect any frontend application (React, Vue, mobile app, etc.) to this backend:

Ensure the server is running.

Use the base URL http://127.0.0.1:8000 (or your deployed URL).

CORS is enabled, allowing requests from any origin.

Example Fetch (JavaScript):

const response = await fetch("[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: "user1", message: "Describe the image" })
});
