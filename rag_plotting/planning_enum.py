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
    from rag_plotting.llm_client import call_openai  # to be provided
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
    cleaned = re.sub(r"[^A-Za-z\s\.\-']", " ", key).strip()
    return cleaned.title() if cleaned else None

def extract_countries_from_text(text: str) -> List[str]:
    """
    Heurísticamente extrae países cuando el usuario compara algunos explícitamente.
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


def _region_from_user_query(user_query: str) -> Optional[str]:
    """
    Detecta la región explícita pedida por el usuario a partir de alias conocidos.
    Si se detecta, siempre debe prevalecer sobre la región propuesta por el LLM.
    """
    q = user_query.lower()
    # match por alias
    for canon, aliases in REGION_ALIASES.items():
        for token in aliases + [canon]:
            if token.lower() in q:
                return "Middle East" if canon == "middle east" else canon.title()
    # palabras genéricas
    if any(tok in q for tok in ["world", "global", "all countries"]):
        return "World"
    return None


# --------- planning ---------
def interpret_query(user_query: str) -> QueryPlan:
    """
    Produce un QueryPlan robusto. Si el LLM marca una región incorrecta
    pero el usuario explicitó otra, la del usuario PREVALECE.
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

    # === Región declarada por el USUARIO (debe ganar) ===
    region_user = _region_from_user_query(user_query)

    # Si el LLM no puso región, intenta inferir por alias (legacy)
    if plan.region is None and region_user:
        plan.region = region_user

    # Si el LLM puso una región pero el usuario pidió otra explícita → sobreescribe
    if region_user and plan.region and (normalize_region(plan.region) != region_user):
        print(f"[Plan] Overriding LLM region '{plan.region}' with user region '{region_user}'")
        plan.region = region_user

    # Si aún no hay región y el texto sugiere global
    if plan.region is None and any(tok in (user_query or "").lower() for tok in ["world","global","all countries"]):
        plan.region = "World"

    # Entity type guess
    qlow = user_query.lower()
    if plan.entity_type is None:
        if any(k in qlow for k in ["market cap","stock market","exchange","bourse"]):
            plan.entity_type = "stock_market"
            plan.task_kind = plan.task_kind or "market_bar"
        else:
            plan.entity_type = "country"
            plan.task_kind = plan.task_kind or "country_metric_map"

    # Respeta el pedido explícito de tipo de gráfico
    enforced = detect_requested_chart_type(user_query)
    if enforced:
        plan.chart_hint = enforced

    # Respeta países manuales (comparaciones)
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
    Pide al LLM la lista de países por macro-región; cachea el resultado.
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
    Enumeración de entidades. Respeta `plan.entities` si el usuario los dio.
    """
    print(f"[Enum] entity_type={plan.entity_type} region={plan.region}")

    # Respeta manual entities
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

    # LLM para otras entidades (e.g., stock markets)
    msgs = build_entity_prompt(user_query, plan.model_dump())
    raw = call_openai(msgs, max_tokens=RESP_TOKENS_ENUM)
    try:
        lst = EntityList(**json.loads(raw))
        print(f"[Enum] LLM enumerated {len(lst.entities)}")
        return lst
    except Exception as e:
        print(f"[Enum][WARN] parse error: {e}")
        return EntityList(entity_type=plan.entity_type or "entity", exhaustive=False, entities=[])
