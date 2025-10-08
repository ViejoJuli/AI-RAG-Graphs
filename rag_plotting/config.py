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

def _as_int(env_name: str, default: int) -> int:
    v = os.getenv(env_name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

# Model name
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"

# VLLM API configuration
VLLM_API_URL   = os.getenv("VLLM_API_URL") or ""
VLLM_API_TOKEN = os.getenv("VLLM_API_TOKEN") or ""

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST") or "localhost"
QDRANT_PORT = _as_int("QDRANT_PORT", 6333)
VECTOR_SIZE = _as_int("VECTOR_SIZE", 1536)  # valor razonable por defecto

# Ollama / embeddings
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL") or ""   # p.ej. http://localhost:11434/api/embeddings
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL")   or "nomic-embed-text"

# Qdrant collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "rag_plotting"

# === Token/Context budgets ===
MODEL_CONTEXT_WINDOW_TOKENS = _as_int("MODEL_CONTEXT_WINDOW_TOKENS", 8192)
RESP_TOKENS_PLAN            = _as_int("RESP_TOKENS_PLAN", 384)
RESP_TOKENS_ENUM            = _as_int("RESP_TOKENS_ENUM", 512)
RESP_TOKENS_COMBINED        = _as_int("RESP_TOKENS_COMBINED", 900)
SAFETY_MARGIN_TOKENS        = _as_int("SAFETY_MARGIN_TOKENS", 256)
