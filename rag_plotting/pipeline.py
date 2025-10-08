"""
Module: pipeline.py
Purpose:
    Orchestrates the end-to-end RAG plotting flow:
      - interpret_query() → enumerate_entities()
      - build_search_queries() → Ray-parallel Qdrant retrieval (filtered+unfiltered)
      - RRF fusion + lexical bump
      - LLM (or fallback) → CombinedSpec
      - DataFrame build + chart rendering
      - Rescue pass (widen knobs) if chart sparse

Design:
    - Minimal imports; inner Ray remote task for retrieval.
    - Keeps behavior identical to monolith while using modular pieces.

Public API:
    - answer_and_plot(user_query: str) -> figure
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import re
import pandas as pd
import ray

from rag_plotting.features import (
    FEATURE_RETRY_IF_SPARSE, FEATURE_MULTI_SEARCH, FEATURE_HYBRID_FILTERED, FEATURE_LEXICAL_RERANK,
    MODES, CURRENT_MODE, _now
)
from rag_plotting.config import COLLECTION_NAME
from rag_plotting.aliases_filters import _region_filter, _text_filter, _any_field_filter, _merge_filters
from rag_plotting.qdrant_io import search_qdrant_with_query
from rag_plotting.schemas import QueryPlan
from rag_plotting.planning_enum import interpret_query, enumerate_entities
from rag_plotting.expansion import build_search_queries, _build_keywords
from rag_plotting.fuse import rrf_merge_rows
from rag_plotting.combined_build import call_model_for_combined_spec
from rag_plotting.dataframe_ops import build_dataframe_from_data_spec
from rag_plotting.render import render_chart_from_df, _chart_is_sparse


def _derive_knobs_from_hints(hints, enumerated_count: int):
    base = MODES[CURRENT_MODE].copy()
    if hints:
        if getattr(hints, "scale", None) == "small":
            base.update(MODES["tight"])
        if getattr(hints, "scale", None) == "large":
            base.update(MODES["wide"])
        if getattr(hints, "k_each", None):
            base["k_each"] = hints.k_each
        if getattr(hints, "k_total", None):
            base["k_total"] = hints.k_total
        if getattr(hints, "max_context", None):
            base["max_context"] = hints.max_context
        if getattr(hints, "max_entities", None):
            base["max_entities"] = hints.max_entities
        if getattr(hints, "max_subqueries", None):
            base["max_subq"] = hints.max_subqueries
        if getattr(hints, "expected_entities", None):
            enumerated_count = max(enumerated_count, hints.expected_entities)
    if enumerated_count >= 60:
        base.update(MODES["wide"])
    elif enumerated_count >= 35:
        base.update(MODES["balanced"])
    return (
        base["k_each"], base["k_total"], base["max_context"], base["max_entities"],
        base["max_syn"], base["max_constr"], base["max_subq"]
    )


@ray.remote(num_cpus=0.25, max_retries=1)
def _search_one_remote(collection: str, query: str, k_each: int, plan: Optional[QueryPlan]) -> List[Dict[str, Any]]:
    """
    Ray task: ejecuta 3 variantes y dedupea:
      1) Región + full-text(text)
      2) Solo región
      3) Sin filtro
    """
    try:
        rows_all: List[Dict[str, Any]] = []

        region_f = _region_filter(plan.region) if (plan and plan.region) else None
        text_f = _text_filter("text", query)  # requiere FT index para mejor calidad

        # filtros por metadata si el plan trae constraints obvios
        meta_f = None
        if plan and plan.constraints:
            mfilters = []
            for key in ["section", "subsection", "product", "datagroup_title", "market", "entity"]:
                v = plan.constraints.get(key)
                if v:
                    mfilters.append(_any_field_filter(key, [v] if isinstance(v, str) else v))
            if mfilters:
                meta_f = _merge_filters(*mfilters)

        f1 = _merge_filters(region_f, text_f, meta_f) if FEATURE_HYBRID_FILTERED else _merge_filters(region_f, meta_f)
        f2 = _merge_filters(region_f, meta_f)
        f3 = None  # sin filtro

        try:
            if f1:
                rows_all.extend(search_qdrant_with_query(collection, query, k=k_each, qfilter=f1))
        except Exception as e:
            rows_all.append({"__error__": f"filtered_text:{type(e).__name__}:{e}"})

        try:
            if f2 and (f1 != f2):
                rows_all.extend(search_qdrant_with_query(collection, query, k=k_each, qfilter=f2))
        except Exception as e:
            rows_all.append({"__error__": f"filtered_region:{type(e).__name__}:{e}"})

        try:
            rows_all.extend(search_qdrant_with_query(collection, query, k=k_each, qfilter=f3))
        except Exception as e:
            rows_all.append({"__error__": f"unfiltered:{type(e).__name__}:{e}"})

        # dedupe simple
        seen, uniq = set(), []
        for r in rows_all:
            sig = (r.get("text"), r.get("country"), r.get("metric"), r.get("committed_date"))
            if sig not in seen:
                uniq.append(r)
                seen.add(sig)
        return uniq[:k_each * 2]
    except Exception as e:
        return [{"__error__": f"fatal:{type(e).__name__}:{e}"}]


def _multi_search_qdrant(collection: str, queries: List[str], k_each: int, k_total: int, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
    if not FEATURE_MULTI_SEARCH:
        region_f = _region_filter(plan.region) if plan else None
        return search_qdrant_with_query(collection, queries[0] if queries else "", k=k_total, qfilter=region_f)

    seen, dedup = set(), []
    for q in queries:
        q2 = re.sub(r"\s+", " ", q.strip())
        if q2 and q2 not in seen:
            dedup.append(q2)
            seen.add(q2)
    if not dedup:
        print("[Multi] No queries after dedupe.")
        return []

    MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_QUERIES", "32"))
    print(f"[Multi] Launching {len(dedup)} Ray tasks (cap {MAX_PARALLEL}) k_each={k_each}, k_total={k_total}…")

    rankings = []
    errors = 0
    for i in range(0, len(dedup), MAX_PARALLEL):
        batch = dedup[i:i + MAX_PARALLEL]
        jobs = [_search_one_remote.remote(collection, q, k_each, plan) for q in batch]
        results = ray.get(jobs)
        for r in results:
            if r and isinstance(r, list) and len(r) and isinstance(r[0], dict) and "__error__" in r[0]:
                errors += 1
            else:
                rankings.append(r)
    print(f"[Multi] Collected {len(rankings)} rankings (errors={errors})")
    keywords = _build_keywords(plan, plan.normalized_question) if (plan and FEATURE_LEXICAL_RERANK) else None
    fused = rrf_merge_rows(rankings, k=k_total, c=60, keywords=keywords, lexical_weight=0.35)
    return fused


def _run_once(user_query: str, widen: bool = False):
    print(f"\n[Run] widen={widen}")
    plan = interpret_query(user_query)
    enumerated = enumerate_entities(user_query, plan)
    k_each, k_total, max_ctx, max_entities, max_syn, max_constr, max_subq = _derive_knobs_from_hints(
        plan.retrieval_hints, len(enumerated.entities)
    )
    if widen:
        k_each = int(k_each * 2.0)
        k_total = int(k_total * 2.0)
        max_ctx = int(k_each * 1.5 + max_ctx * 0.5)
        max_entities = max(int(max_entities * 1.5), len(enumerated.entities))
        max_subq = int(max_subq * 1.25)
        max_ctx = min(max_ctx, 140)
    print(f"[Knobs] final k_each={k_each} k_total={k_total} max_ctx={max_ctx} max_entities={max_entities} max_subq={max_subq}")

    queries = build_search_queries(plan, enumerated, max_entities, max_syn, max_constr, max_subq)
    context_struct = _multi_search_qdrant(COLLECTION_NAME, queries, k_each=k_each, k_total=k_total, plan=plan)[:max_ctx]
    print(f"[Context] rows={len(context_struct)}")
    combined = call_model_for_combined_spec(user_query, context_struct, chart_hint=plan.chart_hint, plan=plan)
    df = build_dataframe_from_data_spec(combined.data_spec)
    return df, combined.chart_spec


def answer_and_plot(user_query: str):
    print(f"\n[Main] === Start ===\nQuery: {user_query}\nMode: {CURRENT_MODE}\n")
    if not ray.is_initialized():
        os.environ.setdefault("RAY_LOG_TO_STDERR", "1")
        ray.init(ignore_reinit_error=True, log_to_driver=True)
        print("[Ray] initialized.")
        print("[Ray] cluster resources:", ray.cluster_resources())
        print("[Ray] available resources:", ray.available_resources())

    df, chart = _run_once(user_query, widen=False)

    if FEATURE_RETRY_IF_SPARSE and _chart_is_sparse(df, chart):
        print("[Main][Rescue] Sparse/blank chart detected. Retrying with widened retrieval...")
        df, chart = _run_once(user_query, widen=True)

    from rag_plotting.prompts import detect_requested_chart_type
    if detect_requested_chart_type(user_query) == "bar" and chart.chart_type != "bar":
        print("[Main] Enforcing BAR at render time.")
        enc = chart.encodings or {}
        str_cols = [c for c in df.columns if getattr(df[c], "dtype", None) == "string"]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        x = enc.get("x") or (str_cols[0] if str_cols else (df.columns[0] if len(df.columns) else "label"))
        y = enc.get("y") or (num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else "value"))
        chart.encodings = {"x": x, "y": y}
        chart.chart_type = "bar"
        if not chart.title.lower().startswith("bar"):
            chart.title = f"Bar — {chart.title}"

    fig = render_chart_from_df(df, chart)
    print("[Main] Done.")
    return fig

