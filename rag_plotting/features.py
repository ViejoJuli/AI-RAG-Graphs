"""
Module: features.py
Purpose:
    Central place for feature toggles and retrieval "modes" (tight/balanced/wide).
    Also provides tiny time helpers used across the pipeline.

Performance notes:
    - Import-only module: constant-time.
    - No heavy libs imported here to keep startup time minimal.

How to use:
    from rag_plotting.features import (
        FEATURE_QUERY_PLANNING, FEATURE_MULTI_SEARCH, FEATURE_RRF,
        FEATURE_JSON_EXTRACTION, FEATURE_RETRY_IF_SPARSE, FEATURE_HYBRID_FILTERED,
        FEATURE_LEXICAL_RERANK, FEATURE_LLM_REGION_ENUM, FEATURE_LLM_MARKET_ENUM,
        MODES, CURRENT_MODE, set_mode, _now, _dur
    )
"""

from __future__ import annotations
import time

# =========================
# === Feature toggles ====
# =========================
FEATURE_QUERY_PLANNING  = True
FEATURE_MULTI_SEARCH    = True
FEATURE_RRF             = True
FEATURE_JSON_EXTRACTION = True
FEATURE_RETRY_IF_SPARSE = True   # widen & retry once if chart is blank/too sparse
FEATURE_HYBRID_FILTERED = True   # run filtered + unfiltered searches and fuse
FEATURE_LEXICAL_RERANK  = True   # add keyword-based bump from payload text/fields

# Prefer using the LLM to enumerate entities/regions first (minimize static lists)
FEATURE_LLM_REGION_ENUM = True
FEATURE_LLM_MARKET_ENUM = True

# =========================
# ====== Modes/Knobs ======
# =========================
MODES = {
    "tight": {
        "k_each": 4,   "k_total": 24,
        "max_context": 24,  "max_entities": 20,
        "max_syn": 3,  "max_constr": 3, "max_subq": 48
    },
    "balanced": {
        "k_each": 6,   "k_total": 64,
        "max_context": 128, "max_entities": 80,
        "max_syn": 6,  "max_constr": 5, "max_subq": 160
    },
    "wide": {
        "k_each": 10,  "k_total": 196,
        "max_context": 256, "max_entities": 200,
        "max_syn": 8,  "max_constr": 8, "max_subq": 320
    },
}

CURRENT_MODE = "balanced"

def _now() -> float:
    return time.time()

def _dur(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"

def set_mode(name: str) -> None:
    """Switch runtime knobs preset."""
    global CURRENT_MODE
    if name in MODES:
        CURRENT_MODE = name
        print(f"[Mode] '{name}' => {MODES[name]}")
    else:
        print(f"[Mode][WARN] Unknown '{name}'. Keeping '{CURRENT_MODE}'.")

# Initialize to env/default
set_mode(CURRENT_MODE)
