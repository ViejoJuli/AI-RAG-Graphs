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
from typing import Any, Dict
import json
import numpy as np
import requests

from rag_plotting.config import OLLAMA_SERVER_URL, EMBEDDING_MODEL
from rag_plotting.features import _now, _dur


def get_embedding(text: str) -> np.ndarray:
    """
    Calls your embedding endpoint; expects a dict with 'embedding' or 'embeddings' in the JSON.
    Returns:
        np.ndarray shape (VECTOR_SIZE,)
    """
    payload: Dict[str, Any] = {"model": EMBEDDING_MODEL, "input": text}
    print(f"[Embed] Calling embeddings for text len={len(text)}…")
    t0 = _now()

    try:
        resp = requests.post(OLLAMA_SERVER_URL, json=payload, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"[Embedding][ERROR] request failed: {type(e).__name__}: {e}") from e

    print(f"[Embed] HTTP {resp.status_code} in {_dur(t0)}")
    if resp.status_code != 200:
        # Try to show a short preview of error payload for debugging
        preview = resp.text
        if len(preview) > 300:
            preview = preview[:300] + "..."
        raise RuntimeError(f"[Embedding][ERROR] HTTP {resp.status_code}: {preview}")

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[Embedding][ERROR] invalid JSON: {e}") from e

    emb = data.get("embeddings") or data.get("embedding")
    if emb is None:
        raise RuntimeError(f"[Embedding][ERROR] Unexpected payload keys: {list(data.keys())}")

    arr = np.array(emb).squeeze()
    if arr.ndim != 1:
        # If API returns [[...]] we squeeze to 1D; otherwise raise
        raise RuntimeError(f"[Embedding][ERROR] expected 1D vector, got shape={arr.shape}")
    return arr
