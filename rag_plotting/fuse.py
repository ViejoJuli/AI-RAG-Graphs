"""
Module: fuse.py
Purpose:
    Ranking fusion and light lexical reranking:
      - rrf_merge_rows(): Reciprocal Rank Fusion with optional lexical bump
      - _lexical_score(): tiny keyword presence score
      - export _build_keywords from expansion when needed upstream

Design:
    - Allocation-light: iterates once over rankings and uses tuple signatures.
    - Deterministic: preserves stable ordering through scores and input order.
    - No external dependencies beyond stdlib.

Usage:
    from rag_plotting.fuse import rrf_merge_rows
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from rag_plotting.expansion import _build_keywords  # re-exposed for callers


def _lexical_score(row: Dict[str, str], keywords: List[str]) -> float:
    """
    Very small lexical bump based on presence of keywords across common fields.
    """
    fields = [
        "text","region","market","entity","product","section","subsection",
        "datagroup_title","metric","fitch_rating","settlement_cycle",
        "stock_market","market_cap"
    ]
    blob = " ".join(str(row.get(k, "")) for k in fields).lower()
    score = 0.0
    for kw in keywords:
        if kw in blob:
            score += 1.0
    # light bias for structured context
    if row.get("region"): score += 0.5
    if row.get("entity"): score += 0.25
    if row.get("market"): score += 0.25
    if row.get("market_cap"): score += 0.25
    return score


def rrf_merge_rows(
    rankings: List[List[Dict[str, str]]],
    k: int,
    c: int = 60,
    keywords: Optional[List[str]] = None,
    lexical_weight: float = 0.3
) -> List[Dict[str, str]]:
    """
    Reciprocal Rank Fusion with optional lexical bump.

    Args:
        rankings: list of ranked lists (each item is a normalized row dict).
        k: number of fused results to keep.
        c: RRF constant (higher → flatter).
        keywords: optional keyword list for lexical bump.
        lexical_weight: scaling applied to lexical score.

    Returns:
        fused top-k list of rows.
    """
    scores: Dict[Tuple, float] = {}
    row_map: Dict[Tuple, Dict[str, str]] = {}

    def _sig(r: Dict[str, str]) -> Tuple:
        return (
            r.get("country"), r.get("iso3"), r.get("region"),
            r.get("market"), r.get("metric"), r.get("value"),
            r.get("fitch_rating"), r.get("settlement_cycle"),
            r.get("stock_market"), r.get("market_cap"),
            r.get("committed_date"), r.get("text"),
        )

    # RRF aggregation
    for rlist in rankings:
        for rank, row in enumerate(rlist):
            sig = _sig(row)
            scores[sig] = scores.get(sig, 0.0) + 1.0 / (c + rank + 1)
            if sig not in row_map:
                row_map[sig] = row

    # Lexical bump
    if keywords:
        for sig, row in row_map.items():
            scores[sig] = scores.get(sig, 0.0) + lexical_weight * _lexical_score(row, keywords)

    # Sort and take top-k
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = [row_map[sig] for sig, _ in merged][:k]
    print(f"[RRF] rankings={len(rankings)} → fused_rows={len(fused)}")
    return fused
