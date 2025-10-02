"""
Module: combined_build.py
Purpose:
    Build a CombinedSpec (data + chart) from context:
      - call_model_for_combined_spec(): LLM JSON → CombinedSpec with robust repair/fallbacks
      - Fallback builders when LLM fails (markets → bar, Fitch/country → map/bar)
      - JSON soft repair helpers and token-budget slicing for context
      - Small encoding sanitizers and 'bar' coercion if explicitly requested by user

Fix notes:
    - Do NOT drop countries when numeric value is missing; fall back to presence=1.
    - If all rows would be dropped, rebuild rows from any detected countries (presence=1).
    - Markets fallback also supports presence=1 when market_cap is missing.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import math
import re

import numpy as np

from rag_plotting.schemas import (
    CombinedSpec, DataSpec, DataColumn, ChartSpec, QueryPlan
)
from rag_plotting.prompts import build_combined_prompt, detect_requested_chart_type
from rag_plotting.config import (
    MODEL_CONTEXT_WINDOW_TOKENS, RESP_TOKENS_COMBINED, SAFETY_MARGIN_TOKENS
)

# Optional LLM client import with graceful fallback
try:
    from rag_plotting.llm_client import call_openai  # to be provided
except Exception:  # pragma: no cover
    def call_openai(messages, max_tokens=None) -> str:  # type: ignore
        print("[LLM][WARN] llm_client not available, using fallback builder.")
        return ""


# --------- small utilities ----------
def _approx_tokens_from_text(s: str) -> int:
    # Very rough token estimate: ~4 chars/token for English.
    return max(1, math.ceil(len(s) / 4))

def _clean_encodings(enc: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Ensure encodings is Dict[str, str] with NO None/blank values."""
    out: Dict[str, str] = {}
    for k, v in (enc or {}).items():
        if isinstance(v, str) and v.strip():
            out[k] = v
    return out

def _select_context_under_budget(
    context_struct: List[Dict[str, Any]],
    base_prompt_len_tokens: int,
    response_budget_tokens: int
) -> List[Dict[str, Any]]:
    max_prompt_tokens = MODEL_CONTEXT_WINDOW_TOKENS - response_budget_tokens - SAFETY_MARGIN_TOKENS
    if max_prompt_tokens < 512:
        max_prompt_tokens = 512
    selected = []
    running = base_prompt_len_tokens
    for r in context_struct:
        row_s = json.dumps(r, ensure_ascii=False)
        tok = _approx_tokens_from_text(row_s)
        if running + tok > max_prompt_tokens:
            break
        selected.append(r)
        running += tok
    print(f"[Budget] base≈{base_prompt_len_tokens} rows={len(selected)} est_tok≈{running} max_prompt≈{max_prompt_tokens}")
    return selected


# --------- JSON soft repair ----------
def _strip_cc(s: str) -> str:
    return "".join(ch for ch in s if (ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 32))

def _balance_json_slice(s: str) -> Optional[str]:
    s = _strip_cc(s)
    try:
        start = s.index("{")
    except ValueError:
        return None
    depth, end = 0, None
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        last = s.rfind("}")
        if last > start:
            return s[start:last+1]
        return None
    return s[start:end]

def _json_soft_repair(s: str) -> str:
    s = _strip_cc(s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"}\s*{", "},{", s)
    s = re.sub(r'("rows"\s*:\s*\[\s*)(\{)', r'\1\2', s)
    s = re.sub(r'(\})\s*(\{)', r'\1,\2', s)
    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\bInfinity\b", "null", s)
    s = re.sub(r"\b-?Inf\b", "null", s)
    return s

def _safe_json_loads(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    blk = _balance_json_slice(raw)
    if blk is None:
        raise json.JSONDecodeError("No JSON object could be decoded", raw, 0)
    try:
        return json.loads(blk)
    except json.JSONDecodeError:
        fixed = _json_soft_repair(blk)
        return json.loads(fixed)


# --------- ratings helpers ----------
_FITCH_SCALE = {
    "AAA": 21, "AA+": 20, "AA": 19, "AA-": 18, "A+": 17, "A": 16, "A-": 15,
    "BBB+": 14, "BBB": 13, "BBB-": 12, "BB+": 11, "BB": 10, "BB-": 9,
    "B+": 8, "B": 7, "B-": 6, "CCC+": 5, "CCC": 4, "CCC-": 3, "CC": 2, "C": 1, "RD": 0, "D": 0
}

def _rating_to_score(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    return _FITCH_SCALE.get(s, None)


# --------- fallbacks ----------
def _build_fallback_markets(
    context_struct: List[Dict[str, Any]],
    region_hint: Optional[str]
) -> Optional[CombinedSpec]:
    """
    Try to build a bar chart from stock markets.
    If market_cap missing, fall back to presence=1 to avoid empty bars.
    """
    rows = []
    have_numeric = False
    for r in context_struct:
        mk = r.get("market") or r.get("stock_market")
        if not mk:
            continue
        cap = r.get("market_cap") or r.get("value")
        num: Optional[float] = None
        if cap is not None and str(cap).strip() != "":
            try:
                num = float(cap)
                have_numeric = True
            except Exception:
                num = None
        rows.append({"market": mk, "market_cap": num if num is not None else 1, "country": r.get("country")})
    if not rows:
        return None

    data_spec = DataSpec(
        columns=[
            DataColumn(name="market", type="string"),
            DataColumn(name="market_cap", type="number"),
            DataColumn(name="country", type="string"),
        ],
        rows=rows,
        notes="Fallback CombinedSpec from markets (presence=1 when missing)."
    )
    chart = ChartSpec(
        chart_type="bar",
        title=f"Market Cap by Stock Market — {region_hint or 'Selected'}",
        encodings=_clean_encodings({"x": "market", "y": "market_cap"})
    )
    return CombinedSpec(data_spec=data_spec, chart_spec=chart)


def _fallback_rows_from_countries(context_struct: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build presence=1 rows from any country fields in the context.
    """
    found = []
    seen = set()
    for r in context_struct:
        country = r.get("country") or r.get("entity")
        iso3 = r.get("iso3")
        if not country:
            continue
        key = (country, iso3)
        if key in seen:
            continue
        seen.add(key)
        found.append({"country": country, "iso3": iso3, "fitch_rating": r.get("fitch_rating"), "rating_score": 1})
    return found


def _build_fallback_fitch_or_country(
    user_query: str,
    context_struct: List[Dict[str, Any]],
    region_hint: Optional[str],
    plan: QueryPlan
) -> CombinedSpec:
    """
    Build a map (or bar if requested) of countries with rating_score; fall back to presence=1.
    """
    rows = []
    for r in context_struct:
        country = r.get("country") or r.get("entity")
        if not country:
            continue
        iso3 = r.get("iso3")
        fitch = r.get("fitch_rating")

        # Prefer Fitch → numeric; else try 'value'; else presence=1
        score = _rating_to_score(fitch) if fitch else None
        val = r.get("value")
        if score is None and val is not None and str(val).strip() != "":
            try:
                score = float(val)
            except Exception:
                score = None
        if score is None:
            score = 1.0  # <-- PRESENCE FALLBACK

        rows.append({
            "country": country,
            "iso3": iso3,
            "fitch_rating": fitch,
            "rating_score": score
        })

    # Dedup per country: keep the first with the highest info (prefer >1 over 1)
    best_by_country: Dict[str, Dict[str, Any]] = {}
    for rr in rows:
        c = rr["country"]
        prev = best_by_country.get(c)
        if prev is None:
            best_by_country[c] = rr
        else:
            # Prefer a row whose score != 1 (i.e., real numeric) over presence
            if prev.get("rating_score", 1) == 1 and rr.get("rating_score", 1) != 1:
                best_by_country[c] = rr
    rows = list(best_by_country.values())

    # If after all we have no rows, rebuild from countries list as presence=1
    if not rows:
        rows = _fallback_rows_from_countries(context_struct)

    # If only 1 row and we have more countries available in context, try to add a couple more presence rows
    if len(rows) == 1:
        extras = _fallback_rows_from_countries(context_struct)
        extras = [e for e in extras if e["country"] != rows[0]["country"]]
        rows += extras[:4]  # add up to 4 extra rows to avoid single-bar charts

    columns = [
        DataColumn(name="country", type="string"),
        DataColumn(name="iso3", type="string"),
        DataColumn(name="fitch_rating", type="string"),
        DataColumn(name="rating_score", type="number"),
    ]

    requested = detect_requested_chart_type(user_query)
    chart_type = "bar" if requested == "bar" else "map-choropleth"

    if chart_type == "bar":
        enc = _clean_encodings({"x": "country", "y": "rating_score"})
    else:
        enc = _clean_encodings({
            "location": "iso3" if any((r.get('iso3') or "") for r in rows) else "country",
            "value": "rating_score",
        })

    chart = ChartSpec(
        chart_type=chart_type,
        title=f"{'Bar' if chart_type == 'bar' else 'Map'} — {region_hint or (plan.region or 'Selected')}",
        region_hint=region_hint,
        color_legend="Rating score" if chart_type != "bar" else None,
        encodings=enc
    )
    data_spec = DataSpec(columns=columns, rows=rows, notes="Fallback CombinedSpec (presence=1 when missing).")
    return CombinedSpec(data_spec=data_spec, chart_spec=chart)


def _coerce_spec_to_bar(combined: CombinedSpec) -> CombinedSpec:
    """
    If the model produced a non-bar chart but user asked for bar, force a bar spec.
    """
    combined.chart_spec.chart_type = "bar"
    enc = combined.chart_spec.encodings or {}

    df_cols = [c.name for c in combined.data_spec.columns]
    str_cols = [c.name for c in combined.data_spec.columns if c.type == "string"]
    num_cols = [c.name for c in combined.data_spec.columns if c.type == "number"]

    # ensure market_cap present if appears in rows
    if "market_cap" not in df_cols:
        if combined.data_spec.rows and "market_cap" in combined.data_spec.rows[0]:
            combined.data_spec.columns.append(DataColumn(name="market_cap", type="number"))
            num_cols.append("market_cap")

    x = enc.get("x"); y = enc.get("y")
    if not (x and y):
        if "market" in df_cols and "market_cap" in df_cols:
            x, y = "market", "market_cap"
        elif "stock_market" in df_cols and ("market_cap" in df_cols or "value" in df_cols):
            x, y = "stock_market", ("market_cap" if "market_cap" in df_cols else "value")
        elif "market" in df_cols and "value" in df_cols:
            x, y = "market", "value"
        elif "country" in df_cols and ("value" in df_cols or "rating_score" in df_cols):
            x, y = "country", ("value" if "value" in df_cols else "rating_score")
        else:
            x = str_cols[0] if str_cols else (df_cols[0] if df_cols else "label")
            y = num_cols[0] if num_cols else (df_cols[1] if len(df_cols) > 1 else "value")

    combined.chart_spec.encodings = _clean_encodings({"x": x, "y": y})
    if not combined.chart_spec.title.lower().startswith("bar"):
        combined.chart_spec.title = f"Bar — {combined.chart_spec.title}"
    return combined


# --------- main builder ----------
def call_model_for_combined_spec(
    user_query: str,
    context_struct: List[Dict[str, Any]],
    chart_hint: Optional[str],
    plan: Optional[QueryPlan] = None
) -> CombinedSpec:
    """
    Ask the LLM for a CombinedSpec, then robustly parse/fix or fallback.
    """
    enforced = detect_requested_chart_type(user_query)
    print(f"[LLM] Building CombinedSpec… context_rows={len(context_struct)} chart_hint={chart_hint} enforced={enforced}")

    # Token-budget the context and form the prompt
    base_messages = build_combined_prompt(
        user_query=user_query,
        context_jsonl="",  # will be set after slicing
        chart_hint=chart_hint,
        enforced_chart_type=enforced
    )
    base_tokens = sum(_approx_tokens_from_text(m["content"]) for m in base_messages)
    selected_rows = _select_context_under_budget(context_struct, base_tokens, RESP_TOKENS_COMBINED)
    context_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in selected_rows)

    messages = build_combined_prompt(
        user_query=user_query,
        context_jsonl=context_jsonl,
        chart_hint=chart_hint,
        enforced_chart_type=enforced
    )

    raw = call_openai(messages, max_tokens=RESP_TOKENS_COMBINED) or ""
    candidate = raw
    m = re.search(r"\{.*", raw, flags=re.DOTALL)
    if m:
        candidate = m.group(0)

    try:
        parsed = _safe_json_loads(candidate)
        combined = CombinedSpec(**parsed)
        combined.chart_spec.encodings = _clean_encodings(combined.chart_spec.encodings)
        if enforced == "bar" and combined.chart_spec.chart_type != "bar":
            print("[LLM] Coercing chart to BAR as requested.")
            combined = _coerce_spec_to_bar(combined)
        print(f"[LLM] CombinedSpec OK: chart_type={combined.chart_spec.chart_type} title='{combined.chart_spec.title}' rows={len(combined.data_spec.rows)}")
        return combined
    except Exception as e:
        preview = candidate[:300].replace("\n", " ")
        print(f"[LLM][ERROR] JSON decode failed: {type(e).__name__}: {e}. Preview: {preview} ...")
        # Fallbacks
        region_hint = plan.region if plan else None
        fb = _build_fallback_markets(context_struct, region_hint)
        if fb is None:
            fb = _build_fallback_fitch_or_country(user_query, context_struct, region_hint, plan or QueryPlan(normalized_question=user_query))
        fb.chart_spec.encodings = _clean_encodings(fb.chart_spec.encodings)
        if enforced == "bar" and fb.chart_spec.chart_type != "bar":
            fb = _coerce_spec_to_bar(fb)
        print(f"[LLM][Fallback] rows={len(fb.data_spec.rows)} chart={fb.chart_spec.chart_type}")
        return fb
