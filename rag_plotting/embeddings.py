"""
Module: embeddings.py
Purpose:
    Thin, fast client for your embedding endpoint (Ollama or custom HTTP API).
    Exposes a single public function: `get_embedding(text: str) -> np.ndarray`.

Design:
    - Zero heavy imports beyond `requests` and `numpy`.
    - Pure function; no global state.
    - Strict error handling with concise logs.
    - Returns a 1D numpy array to match the expected Qdrant vector format.

Usage:
    from rag_plotting.embeddings import get_embedding
"""

from __future__ import annotations
from typing import Any, Dict, List
import json
import numpy as np
import requests

from rag_plotting.config import OLLAMA_SERVER_URL, EMBEDDING_MODEL
from rag_plotting.features import _now, _dur

def _extract_vector(payload: Dict[str, Any]) -> List[float]:
    """
    Admite formatos comunes:
    - {"embedding":[...]}
    - {"embeddings":[...]}
    - {"data":[{"embedding":[...]}]}
    """
    if "embedding" in payload and isinstance(payload["embedding"], list):
        return payload["embedding"]
    if "embeddings" in payload and isinstance(payload["embeddings"], list):
        # algunos devuelven [[...]]
        inner = payload["embeddings"]
        return inner[0] if inner and isinstance(inner[0], list) else inner
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
        return data[0]["embedding"]
    raise RuntimeError(f"[Embedding][ERROR] Unexpected payload keys: {list(payload.keys())}")

def get_embedding(text: str) -> np.ndarray:
    """
    Llama a tu endpoint de embeddings (Ollama/compat OpenAI/vLLM). Devuelve (VECTOR_SIZE,).
    """
    if not OLLAMA_SERVER_URL:
        raise RuntimeError("[Embedding][ERROR] OLLAMA_SERVER_URL is empty.")
    payload: Dict[str, Any] = {"model": EMBEDDING_MODEL, "input": text}
    print(f"[Embed] Calling embeddings for text len={len(text)}…")
    t0 = _now()
    try:
        resp = requests.post(OLLAMA_SERVER_URL, json=payload, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"[Embedding][ERROR] request failed: {type(e).__name__}: {e}") from e

    print(f"[Embed] HTTP {resp.status_code} in {_dur(t0)}")
    if resp.status_code != 200:
        preview = resp.text
        if len(preview) > 300:
            preview = preview[:300] + "..."
        raise RuntimeError(f"[Embedding][ERROR] HTTP {resp.status_code}: {preview}")

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[Embedding][ERROR] invalid JSON: {e}") from e

    vec = _extract_vector(data)
    arr = np.array(vec).squeeze()
    if arr.ndim != 1:
        arr = np.array(arr).reshape(-1)
    return arr
