import os
from pathlib import Path
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

# Create necessary directories
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    # Embedding and Reranking Models
    "BGE_MODEL_NAME": "BAAI/bge-m3",
    "RERANKER_MODEL_NAME": "BAAI/bge-reranker-v2-m3",
    
    # Fine-tuned Models
    "QWEN_MODEL_NAME": str(MODELS_DIR / "qwen_finetune"),
    "LLAMA_MODEL_NAME": str(MODELS_DIR / "llama_finetune"),
    "GEMMA_MODEL_NAME": str(MODELS_DIR / "gemma_finetune"),
}

# Available Models for UI
AVAILABLE_MODELS = {
    "Qwen 2.5B (Fine-tuned)": MODEL_CONFIG["QWEN_MODEL_NAME"],
    "Llama 3.2B (Fine-tuned)": MODEL_CONFIG["LLAMA_MODEL_NAME"],
    "Gemma 2B (Fine-tuned)": MODEL_CONFIG["GEMMA_MODEL_NAME"]
}

# Data Paths
DATA_CONFIG = {
    "KNOWLEDGE_BASE": str(DATA_DIR / "data.txt"),
    "QUESTIONS": str(DATA_DIR / "question.txt"),
}

# Database Configuration
DB_CONFIG = {
    # ChromaDB
    "CHROMA_DB_PATH": str(CACHE_DIR / "chroma_db_bge"),
    "CHROMA_QUESTION_DB_PATH": str(CACHE_DIR / "chroma_question_db"),
    
    # MongoDB
    "MONGO_URI": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME", "chatbot"),
    "MONGO_COLLECTION": os.getenv("MONGO_COLLECTION", "chat_history")
}

# API Keys and Tokens
API_KEYS = {
    "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN", "")
}

# Model Parameters
MODEL_PARAMS = {
    "MAX_LENGTH": 512,
    "TOP_K": 50,
    "TOP_P": 0.95,
    "TEMPERATURE": 0.7,
    "REPETITION_PENALTY": 1.1
}

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")