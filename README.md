# Image RAG Chatbot 🤖🖼️

A powerful backend API that allows users to **chat** with their images using Retrieval-Augmented Generation (RAG). This project analyzes images, indexes their visual content, and enables context-aware Q&A using Google's Gemini 2.5 Flash model.

🔗 **Live API URL**: _[to be added]_ (or run locally on `http://127.0.0.1:8000`)

---

## 🚀 Features

- **Multimodal Ingestion**  
  Upload images (JPG/PNG), extract text, objects, and layout using Gemini Vision, and create vector embeddings.

- **Context-Aware Chat**  
  Remembers conversation history (e.g., `What is the total?` → `Is that in USD?`) using LangChain.

- **Persistent Memory**  
  Stores vector data in a local ChromaDB instance that persists across restarts.

- **FastAPI Backend**  
  High-performance API with CORS enabled for easy frontend integration.

- **Self-Healing**  
  Built-in retry logic to handle API rate limits and 503 errors gracefully.

---

## 🛠️ Tech Stack

| Layer        | Technology                                  |
|-------------|----------------------------------------------|
| Language    | Python 3.10+                                 |
| Core Logic  | LangChain                                    |
| AI Engine   | Google Gemini 2.5 Flash (google-genai SDK)   |
| Vector DB   | ChromaDB                                     |
| API Server  | FastAPI                                      |
| Dependency  | `uv` (recommended) or `pip`                  |

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/theharshchandrakar/RAG_Chatbot_Image.git
cd RAG_Chatbot_Image
```


### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
# Using standard pip
pip install -r requirements.txt

# OR using uv (faster)
uv pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

---

## 🏃‍♂️ Running the Server

Start the API server locally:

```bash
python main.py
