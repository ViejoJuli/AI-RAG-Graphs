"""
Module: planning_enum.py
Purpose:
    Implements:
      - interpret_query(): LLM-backed (with heuristic fallback) query planning → QueryPlan
      - enumerate_entities(): entity enumeration (countries by region or stock markets), with LLM + cache
      - llm_region_countries(): macro-region → exhaustive country list via LLM (cached)

Performance & resilience:
    - If LLM client is not wired yet, functions fall back to robust heuristics to avoid breaking the run.
    - Minimal imports; heavy clients only in llm_client (to be provided).
    - Region normalization uses aliases_filters helpers (no I/O).

Usage:
    from rag_plotting.planning_enum import interpret_query, enumerate_entities
"""

from __future__ import annotations
from typing import Dict, List, Optional
import json
import re
import math

from rag_plotting.features import (
    FEATURE_QUERY_PLANNING, FEATURE_LLM_REGION_ENUM
)
from rag_plotting.schemas import (
    QueryPlan, RetrievalHints, EntityList
)
from rag_plotting.prompts import (
    build_plan_prompt, build_entity_prompt, build_region_enum_prompt, detect_requested_chart_type
)
from rag_plotting.aliases_filters import (
    REGION_ALIASES, normalize_region
)
from rag_plotting.config import (
    RESP_TOKENS_PLAN, RESP_TOKENS_ENUM
)

# Optional LLM client import with graceful fallback
try:
    from rag_plotting.llm_client import call_openai  # to be provided next
except Exception:  # pragma: no cover
    def call_openai(messages, max_tokens=None) -> str:  # type: ignore
        # Safe stub: return empty → triggers heuristic fallback paths
        print("[LLM][WARN] llm_client not available, using heuristic fallback.")
        return ""


# --------- lightweight text utils (shared) ----------
_COUNTRY_WORD = re.compile(r"[A-Za-z][A-Za-z\.\-\s']{1,}")

def _normalize_country_name(name: str) -> Optional[str]:
    key = name.strip()
    if not key:
        return None
    # Title-case without forcing specific alias maps (kept minimal here)
    cleaned = re.sub(r"[^A-Za-z\s\.\-']", " ", key).strip()
    return cleaned.title() if cleaned else None

def extract_countries_from_text(text: str) -> List[str]:
    """
    Heuristically extract explicit country mentions when user compares a few.
    """
    text = re.sub(r"\bvs\.?\b", ",", text, flags=re.I)
    text = re.sub(r"\bversus\b", ",", text, flags=re.I)
    text = re.sub(r"\bcontra\b", ",", text, flags=re.I)
    candidates = re.split(r"[;,/]| and | y | e | vs | versus ", text, flags=re.I)

    found: List[str] = []
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        words = _COUNTRY_WORD.findall(cand)
        for w in words:
            norm = _normalize_country_name(w)
            if norm:
                found.append(norm)
    # unique preserve order
    seen, out = set(), []
    for c in found:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def is_compare_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["compare","compar","vs","versus","contra"])


# --------- planning ---------
def interpret_query(user_query: str) -> QueryPlan:
    """
    Produce a QueryPlan from the user question and add deterministic heuristics.
    If LLM is unavailable or fails to parse, uses a safe heuristic plan.
    """
    print(f"[Plan] Interpreting: {user_query}")
    msgs = build_plan_prompt(user_query)
    raw = call_openai(msgs, max_tokens=RESP_TOKENS_PLAN)

    manual_countries = extract_countries_from_text(user_query)

    if raw:
        try:
            plan = QueryPlan(**json.loads(raw))
            print(f"[Plan] Parsed JSON OK → region={plan.region} entity_type={plan.entity_type} chart={plan.chart_hint}")
        except Exception as e:
            print(f"[Plan][WARN] Parse error: {e} → using heuristic fallback.")
            plan = QueryPlan(
                normalized_question=user_query, subqueries=[user_query],
                chart_hint=None, region=None, task_kind=None,
                entity_type=None, entities=[], constraints={}, synonyms=[],
                retrieval_hints=None
            )
    else:
        print("[Plan][WARN] Empty LLM reply; using heuristic fallback.")
        plan = QueryPlan(
            normalized_question=user_query, subqueries=[user_query],
            chart_hint=None, region=None, task_kind=None,
            entity_type=None, entities=[], constraints={}, synonyms=[],
            retrieval_hints=None
        )

    # Region normalization from aliases in the raw query if model didn't set it
    qlow = user_query.lower()
    if plan.region is None:
        for canon, aliases in REGION_ALIASES.items():
            if any(a.lower() in qlow for a in aliases + [canon]):
                plan.region = "Middle East" if canon == "middle east" else canon.title()
                break
    if plan.region is None and any(tok in qlow for tok in ["world","global","all countries"]):
        plan.region = "World"

    # Entity type guess
    if plan.entity_type is None:
        if any(k in qlow for k in ["market cap","stock market","exchange","bourse"]):
            plan.entity_type = "stock_market"
            plan.task_kind = plan.task_kind or "market_bar"
        else:
            plan.entity_type = "country"
            plan.task_kind = plan.task_kind or "country_metric_map"

    # Respect user's explicit chart request
    enforced = detect_requested_chart_type(user_query)
    if enforced:
        plan.chart_hint = enforced

    # Respect manual explicit countries (comparisons)
    if manual_countries:
        print(f"[Plan] Manual countries detected: {manual_countries}")
        plan.entities = manual_countries
        if is_compare_query(user_query):
            plan.retrieval_hints = plan.retrieval_hints or RetrievalHints()
            plan.retrieval_hints.scale = plan.retrieval_hints.scale or "small"
            plan.retrieval_hints.expected_entities = len(manual_countries)
            plan.region = plan.region or "N/A"

    print(f"[Plan] Final → region={plan.region} entity_type={plan.entity_type} chart={plan.chart_hint} entities={len(plan.entities)}")
    return plan


# --------- enumeration ---------
_LLM_REGION_CACHE: Dict[str, List[str]] = {}

def llm_region_countries(region: str) -> List[str]:
    """
    Ask the LLM for the exhaustive list of countries in a macro-region; cache results.
    Falls back to empty list if LLM unavailable.
    """
    canon = normalize_region(region) or region
    if canon in _LLM_REGION_CACHE:
        return _LLM_REGION_CACHE[canon]
    if not FEATURE_LLM_REGION_ENUM:
        return []

    msgs = build_region_enum_prompt(canon)
    raw = call_openai(msgs, max_tokens=RESP_TOKENS_ENUM)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("entities"):
            _LLM_REGION_CACHE[canon] = list(dict.fromkeys(data["entities"]))
            return _LLM_REGION_CACHE[canon]
    except Exception:
        pass
    return []


def enumerate_entities(user_query: str, plan: QueryPlan) -> EntityList:
    """
    Enumerate concrete entities. Priority:
      - Respect explicit plan.entities (manual mentions).
      - If entity_type='country' and region present → ask LLM for exhaustive list; fallback to empty.
      - Otherwise ask the LLM to list entities (e.g., stock markets).
    """
    print(f"[Enum] entity_type={plan.entity_type} region={plan.region}")

    # Respect manual entities
    if plan.entities:
        out = EntityList(entity_type=plan.entity_type or "entity", exhaustive=False, entities=plan.entities)
        print(f"[Enum] Using manual entities={len(out.entities)}")
        return out

    etype = (plan.entity_type or "").strip().lower()
    region = normalize_region(plan.region) if plan.region else None

    if etype in ("country", "countries"):
        ents: List[str] = []
        if region:
            ents = llm_region_countries(region)
        out = EntityList(entity_type="country", exhaustive=bool(ents), entities=ents)
        print(f"[Enum] Countries via LLM={len(out.entities)}")
        return out

    # Ask LLM for general entities (e.g., stock markets)
    msgs = build_entity_prompt(user_query, plan.model_dump())
    raw = call_openai(msgs, max_tokens=RESP_TOKENS_ENUM)
    try:
        lst = EntityList(**json.loads(raw))
        print(f"[Enum] LLM enumerated {len(lst.entities)}")
        return lst
    except Exception as e:
        print(f"[Enum][WARN] parse error: {e}")
        return EntityList(entity_type=plan.entity_type or "entity", exhaustive=False, entities=[])
