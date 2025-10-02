"""
Module: expansion.py
Purpose:
    - Construct retrieval queries from plan + entities + synonyms/constraints.
    - Provide synonym inference and lightweight lexical helpers.

APIs:
    build_search_queries(plan, enumerated, max_entities, max_syn, max_constr, max_subq) -> List[str]
    _infer_extra_synonyms(plan) -> List[str]
    _build_keywords(plan, user_query) -> List[str]
    is_compare_query(text) -> bool

Notes:
    - Keeps string ops fast and allocation-light.
    - Dedupes results preserving order, then caps by max_subq.

Usage:
    from rag_plotting.expansion import build_search_queries, _build_keywords, is_compare_query
"""

from __future__ import annotations
from typing import Dict, List, Optional
import re
from rag_plotting.schemas import QueryPlan, EntityList
from rag_plotting.aliases_filters import (
    SYN_DEFAULT, SYN_COUNTRY, SYN_FITCH, SYN_SETTLE, SYN_MARKET, normalize_region
)


# -------- public helpers also used by fusion -----------
def is_compare_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["compare","compar","vs","versus","contra"])


def _infer_extra_synonyms(plan: QueryPlan) -> List[str]:
    q = (plan.normalized_question or "").lower()
    extra: List[str] = []
    if "fitch" in q or "credit rating" in q:
        extra += SYN_FITCH
    if "t+2" in q or "settlement" in q:
        extra += SYN_SETTLE
    if any(k in q for k in ["market cap","stock market","exchange","bourse"]):
        extra += SYN_MARKET
    if (plan.entity_type or "").lower() in ("country","countries"):
        extra += SYN_COUNTRY
    extra += SYN_DEFAULT
    # unique preserving order
    seen, out = set(), []
    for s in extra:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def build_search_queries(
    plan: QueryPlan,
    enumerated: EntityList,
    max_entities: int,
    max_syn: int,
    max_constr: int,
    max_subq: int
) -> List[str]:
    """
    Construct a set of retrieval queries mixing entities + synonyms + constraints + region-wide fallbacks.
    """
    print("[Expand] building search queries...")
    queries: List[str] = []

    # Seed with provided subqueries (already retrieval-friendly)
    for q in (plan.subqueries or [])[:max_subq]:
        q2 = q.strip()
        if q2:
            queries.append(q2)

    # Constraints into friendly tokens
    constr_terms: List[str] = []
    for k, v in (plan.constraints or {}).items():
        if v:
            constr_terms.extend([f"{v} {k}", f"{k} {v}", f"{v}"])

    # Synonyms (explicit + inferred)
    syn = (plan.synonyms or []) + _infer_extra_synonyms(plan)
    syn = syn[:max_syn]

    # Entities slice
    entity_list = (enumerated.entities or plan.entities)[:max_entities]

    # Per-entity expansions
    per_entity: List[str] = []
    for ent in entity_list:
        ent = ent.strip()
        if not ent:
            continue
        per_entity.append(ent)
        for s in syn:
            per_entity.append(f"{ent} {s}")
        for t in constr_terms[:max_constr]:
            per_entity.append(f"{ent} {t}")
        if (plan.entity_type or '').lower() == "stock_market":
            per_entity.append(f"{ent} primary stock exchange")
            per_entity.append(f"{ent} stock market market cap")
            per_entity.append(f"{ent} stock exchange market capitalization USD")

    queries.extend(per_entity)

    # Region-wide patterns when no entities (or to reinforce region signal)
    if (not entity_list) and plan.region:
        region_name = normalize_region(plan.region) or plan.region
        queries.append(f"{region_name} " + (plan.normalized_question or ""))
        for s in syn[:max_syn // 2]:
            queries.append(f"{region_name} {s}")
        if (plan.entity_type or "").lower() == "stock_market":
            queries.append(f"{region_name} stock markets market cap")
            queries.append(f"{region_name} primary exchanges capitalization")

    # Deduplicate while preserving order, then cap
    seen, dedup = set(), []
    for q in queries:
        q2 = re.sub(r"\s+", " ", q.strip())
        if q2 and q2 not in seen:
            dedup.append(q2); seen.add(q2)

    out = dedup[:max_subq]
    print(f"[Expand] queries={len(out)} (dedup from {len(queries)})")
    return out


def _build_keywords(plan: QueryPlan, user_query: str) -> List[str]:
    """
    Build a small keyword set for lexical bumps (used in RRF fusion stage).
    """
    seeds = [user_query]
    seeds += plan.subqueries or []
    seeds += plan.synonyms or []
    seeds += _infer_extra_synonyms(plan)
    seeds += ["Fitch","credit rating","T+2","settlement","market cap","stock market","exchange","bourse","market capitalization"]

    toks: List[str] = []
    for s in seeds:
        s = (s or "").lower()
        toks += re.findall(r"[a-z0-9\+\-]+", s)

    # unique + length filter
    uniq = []
    seen = set()
    for t in toks:
        if len(t) >= 2 and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq[:80]
