"""
Module: config.py
Purpose:
    Centralizes configuration and environment knobs used across the pipeline:
    - Model name, API base URLs/tokens
    - Qdrant host/port, collection name, vector size
    - Context/token budgets

Design notes:
    - Keep read-only constants lightweight to avoid runtime overhead.
    - Do not perform any heavy imports (e.g., Qdrant/OpenAI) here.
    - Favor environment variables with safe defaults so local runs "just work".

How to use:
    from rag_plotting.config import (
        MODEL_NAME, VLLM_API_URL, VLLM_API_TOKEN,
        QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE,
        OLLAMA_SERVER_URL, EMBEDDING_MODEL, COLLECTION_NAME,
        MODEL_CONTEXT_WINDOW_TOKENS, RESP_TOKENS_PLAN, RESP_TOKENS_ENUM,
        RESP_TOKENS_COMBINED, SAFETY_MARGIN_TOKENS
    )
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()
# =========================
# ===== CONFIGURATION =====
# =========================

# Model name
MODEL_NAME = os.getenv("MODEL_NAME")

# VLLM API configuration
VLLM_API_URL = os.getenv("VLLM_API_URL")
VLLM_API_TOKEN = os.getenv("VLLM_API_TOKEN")  # API token for authentication

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))  # Size of vectors used in the application

# Ollama server configuration
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")  # URL for generating embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")  # Name of the embedding model

# Qdrant collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  # Name of the Qdrant collection

# === Token/Context budgets ===
MODEL_CONTEXT_WINDOW_TOKENS = int(os.getenv("MODEL_CONTEXT_WINDOW_TOKENS", "8192"))
RESP_TOKENS_PLAN = int(os.getenv("RESP_TOKENS_PLAN", "384"))
RESP_TOKENS_ENUM = int(os.getenv("RESP_TOKENS_ENUM", "512"))
RESP_TOKENS_COMBINED = int(os.getenv("RESP_TOKENS_COMBINED", "900"))
SAFETY_MARGIN_TOKENS = int(os.getenv("SAFETY_MARGIN_TOKENS", "256"))
