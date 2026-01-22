import os
import sys
import warnings
import logging

# --- NUCLEAR CLEANUP BLOCK ---
# 1. Disable Telemetry Environment Variable
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 2. Suppress Warnings
warnings.filterwarnings("ignore")

# 3. Mute Specific Loggers (The Fix)
# This specifically targets the logger causing your error
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
# -----------------------------

import glob
import time
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_core.documents import Document
from PIL import Image

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file.")

DB_PATH = "./chroma_db"
IMAGE_FOLDER = "./images"

client = genai.Client(api_key=GOOGLE_API_KEY)

embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def generate_with_retry(model_name, prompt, image, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image]
            )
            return response
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg or "overloaded" in error_msg.lower():
                if attempt < retries - 1:
                    print(f"   ⚠️ Server busy. Retrying in {delay}s... (Attempt {attempt+1}/{retries})")
                    time.sleep(delay)
                    continue
            raise e

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def load_and_process_images():
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Created folder: {IMAGE_FOLDER}. Please add images and run again.")
        return

    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
                  glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))

    if not image_paths:
        print("No images found in ./images folder.")
        return

    print(f"🚀 Processing {len(image_paths)} images using: {MODEL_NAME}")

    documents_to_add = []

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        print(f"Analyzing: {file_name}...")
        
        try:
            pil_image = Image.open(img_path)
            prompt = "Analyze this image in detail. Extract all visible text, describe the layout, objects, colors, and any data shown."
            
            response = generate_with_retry(MODEL_NAME, prompt, pil_image)
            
            doc = Document(
                page_content=response.text,
                metadata={"source": file_name}
            )
            documents_to_add.append(doc)
            print(f" -> Success: {file_name}")
            time.sleep(2)

        except Exception as e:
            print(f" -> ❌ Failed processing {file_name}: {e}")

    if documents_to_add:
        print("Adding data to Vector Database...")
        # Note: We still pass Settings, but the logger mute above does the real work
        Chroma.from_documents(
            documents=documents_to_add,
            embedding=embedding_function,
            persist_directory=DB_PATH,
            client_settings=Settings(anonymized_telemetry=False)
        )
        print(f"Success! {len(documents_to_add)} images indexed.")
    else:
        print("No new documents were processed.")

if __name__ == "__main__":
    load_and_process_images()