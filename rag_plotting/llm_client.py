"""
Module: llm_client.py
Purpose:
    Thin client wrapper to call a vLLM/OpenAI-compatible chat API:
      - call_openai(messages, max_tokens=None) -> str

Design:
    - Uses OpenAI SDK pointed to a custom base_url (vLLM server).
    - Strict parse: returns content text (string) or "" on error to trigger fallbacks.
    - No retries here; upstream handles fallback paths.

Usage:
    from rag_plotting.llm_client import call_openai
"""

from __future__ import annotations
from typing import List, Dict, Optional
from openai import OpenAI

from rag_plotting.config import MODEL_NAME, VLLM_API_URL, VLLM_API_TOKEN
from rag_plotting.features import _now, _dur


def call_openai(messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
    """
    Calls your reasoning model (e.g., vLLM server compatible).
    Robust to cases where choices[0].message.content is None.
    """
    print(f"[LLM] Sending {len(messages)} messages, max_tokens={max_tokens}")
    t0 = _now()

    client = OpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_TOKEN)
    kwargs = dict(model=MODEL_NAME, messages=messages, temperature=0.0)
    if max_tokens is not None:
        kwargs["max_tokens"] = max(1, int(max_tokens))

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"[LLM][ERROR] API call failed: {type(e).__name__}: {e}")
        return ""  # empty → caller will fallback

    content = ""
    try:
        choice = completion.choices[0] if getattr(completion, "choices", None) else None
        if choice is not None:
            msg = getattr(choice, "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or ""
            if not content:
                content = getattr(choice, "text", "") or ""
    except Exception as e:
        print(f"[LLM][WARN] Failed to parse completion: {type(e).__name__}: {e}")
        content = ""

    out = (content or "").strip()
    print(f"[LLM] Reply len={len(out)} in {_dur(t0)}")
    return out
