# -*- coding: utf-8 -*-
# rag_plotting/qdrant_io.py
"""
Qdrant IO utilities with deterministic payload selection, ISO normalization,
and light deduplication. Optimized for RAG→Charts stability.

Assumptions:
- You have a running Qdrant (gRPC enabled is recommended).
- COLLECTION_NAME exists and payload fields below are present for most docs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels  # type: ignore

from rag_plotting.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from rag_plotting.features import _now, _dur
from rag_plotting.embeddings import get_embedding
from rag_plotting.aliases_filters import _extract_payload_fields

def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (
            r.get("text"), r.get("country"), r.get("region"),
            r.get("market"), r.get("metric"), r.get("committed_date"),
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
    Vector search contra Qdrant + normalización de payload.
    Si qfilter viene, Qdrant lo aplica en el grafo (filterable HNSW).
    """
    print(f"[Qdrant] search_qdrant_with_query q='{query[:80]}...' k={k} filter={bool(qfilter)}")
    t0 = _now()

    emb = get_embedding(query)

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    result = client.search(
        collection_name=collection_name or COLLECTION_NAME,
        query_vector=emb.tolist(),
        limit=int(k),
        query_filter=qfilter,
        with_payload=True,
    )

    rows: List[Dict[str, Any]] = []
    for hit in result:
        payload = (hit.payload or {}).copy()
        payload["_score"] = getattr(hit, "score", None)
        rows.append(_extract_payload_fields(payload))

    unique = _dedupe_rows(rows)
    print(f"[Qdrant] got {len(unique)} rows in {_dur(t0)}")
    return unique[:k]

