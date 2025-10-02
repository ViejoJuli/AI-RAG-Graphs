"""
Module: qdrant_io.py
Purpose:
    Fast, minimal wrappers for Qdrant search with payload normalization.
    - `search_qdrant_with_query`: one-shot vector search with optional filter.
    - Lightweight de-duplication based on a stable signature.
    - No client global: create QdrantClient on demand for thread-safety.

Performance notes:
    - Keeps imports local to this module.
    - Does not perform schema validation at this layer.
    - Returns plain dict rows ready for downstream fusion/LLM steps.

Usage:
    from rag_plotting.qdrant_io import search_qdrant_with_query
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels  # type: ignore

from rag_plotting.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from rag_plotting.features import _now, _dur
from rag_plotting.embeddings import get_embedding
from rag_plotting.aliases_filters import _extract_payload_fields


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Signature-based de-duplication to reduce repeated chunks while keeping highest quality spread.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (
            r.get("text"),
            r.get("country"),
            r.get("region"),
            r.get("market"),
            r.get("metric"),
            r.get("committed_date"),
        )
        if key not in seen:
            out.append(r)
            seen.add(key)
    return out


def search_qdrant_with_query(
    collection_name: str,
    query: str,
    k: int,
    qfilter: Optional[qmodels.Filter] = None
) -> List[Dict[str, Any]]:
    """
    Single vector search against Qdrant + payload normalization.

    Args:
        collection_name: Qdrant collection name.
        query: natural-language or keyword query to embed.
        k: number of results to return.
        qfilter: optional qmodels.Filter to constrain payload search.

    Returns:
        List[Dict]: normalized rows from payload (see _extract_payload_fields).
    """
    print(f"[Qdrant] search_qdrant_with_query q='{query[:80]}...' k={k} filter={bool(qfilter)}")
    t0 = _now()

    # 1) Embed query
    emb = get_embedding(query)

    # 2) Client + search
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    result = client.search(
        collection_name=collection_name or COLLECTION_NAME,
        query_vector=emb.tolist(),
        limit=int(k),
        query_filter=qfilter,
    )

    # 3) Normalize payload
    rows: List[Dict[str, Any]] = []
    for hit in result:
        payload = (hit.payload or {}).copy()
        payload["_score"] = getattr(hit, "score", None)
        rows.append(_extract_payload_fields(payload))

    unique = _dedupe_rows(rows)
    print(f"[Qdrant] got {len(unique)} rows in {_dur(t0)}")
    return unique[:k]
