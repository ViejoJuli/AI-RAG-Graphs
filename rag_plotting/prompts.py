"""
Module: prompts.py
Purpose:
    Centralizes prompt builders for the LLM:
      - build_plan_prompt(): ask for a QueryPlan JSON
      - build_entity_prompt(): ask to enumerate concrete entities
      - build_region_enum_prompt(): ask for macro-region country lists
      - build_combined_prompt(): ask for a CombinedSpec (data + chart)
      - detect_requested_chart_type(): parse explicit chart requests

Design:
    - Returns "messages" lists suitable for chat.completions.
    - Avoids heavy imports; only string composition.
    - Keep schemas inline as readable, reusable strings.

Usage:
    from rag_plotting.prompts import (
        build_plan_prompt, build_entity_prompt, build_region_enum_prompt,
        build_combined_prompt, detect_requested_chart_type
    )
"""

from __future__ import annotations
from typing import Dict, List, Optional
import json
import re

from rag_plotting.config import (
    MODEL_CONTEXT_WINDOW_TOKENS, RESP_TOKENS_COMBINED
)


# -----------------------
# Chart type detection
# -----------------------
def detect_requested_chart_type(user_query: str) -> Optional[str]:
    """
    Parse the user's literal request to enforce chart type (bar/line/pie/map/heatmap).
    """
    q = (user_query or "").lower()
    if "bar chart" in q or re.search(r"\bbar\b", q):
        return "bar"
    if "line chart" in q or re.search(r"\bline\b", q):
        return "line"
    if "pie chart" in q or re.search(r"\bpie\b", q):
        return "pie"
    if "heatmap" in q:
        return "heatmap"
    if "map" in q or "choropleth" in q:
        return "map-choropleth"
    return None


# -----------------------
# System messages
# -----------------------
SYSTEM_PLAN_ENFORCER = (
    "You are a retrieval planning assistant for a vector RAG system that answers data-visualization questions.\n"
    "Given a USER_QUERY, output ONLY a valid JSON object that matches the QueryPlan schema (no markdown, no backticks).\n"
    "Be explicit when a SET is implied (e.g., all European countries)."
)

PLAN_FS_USER = "please build a chart about policy X adoption by country"
PLAN_FS_ASSISTANT = {
    "normalized_question": "Chart of countries that adopted policy X.",
    "chart_hint": "map-choropleth",
    "region": "N/A",
    "task_kind": "country_metric_map",
    "entity_type": "country",
    "entities": [],
    "constraints": {"policy": "X"},
    "synonyms": ["adopted policy X","implemented policy X","policy X adoption"],
    "subqueries": ["policy X adoption countries","countries with policy X","policy X implemented states"],
    "retrieval_hints": {"scale": "small", "expected_entities": 20, "max_subqueries": 64}
}

def build_plan_prompt(user_query: str) -> List[Dict[str, str]]:
    """
    Build the planning prompt; the model should produce a QueryPlan JSON.
    """
    schema = (
        "Schema fields:\n"
        "- normalized_question (string)\n"
        "- chart_hint (optional string)\n"
        "- region (optional string: Europe|Americas|World|Africa|Asia|Oceania|Middle East)\n"
        "- task_kind (optional string)\n"
        "- entity_type (optional string)\n"
        "- entities (string array)\n"
        "- constraints (object)\n"
        "- synonyms (string array)\n"
        "- subqueries (string array)\n"
        "- retrieval_hints { scale, expected_entities, k_each, k_total, max_context, max_subqueries, max_entities }\n\n"
        "Guidance:\n"
        "- If the question implies a region set, set region and entity_type='country'.\n"
        "- 'market cap' or 'stock market' => task_kind='market_bar', entity_type='stock_market'.\n"
        "- 'settlement' or 'T+2' => task_kind='country_metric_map', add synonyms around 'settlement cycle'.\n"
        "- Include concise, retrieval-friendly subqueries with synonyms and constraints.\n"
        "- Europe ~45, Americas ~35, Africa ~54, Asia ~49, Oceania ~14; scale expected_entities accordingly.\n"
        "- Output JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PLAN_ENFORCER},
        {"role": "user", "content": PLAN_FS_USER},
        {"role": "assistant", "content": json.dumps(PLAN_FS_ASSISTANT)},
        {"role": "user", "content": f"USER_QUERY:\n{user_query}\n\n{schema}\nReturn JSON now:"},
    ]


SYSTEM_ENTITY_ENUM = (
    "You enumerate entities implied by a user request.\n"
    "Return ONLY JSON: {\"entity_type\": str, \"exhaustive\": bool, \"entities\": [str,...]} "
    "using standard English names. No extra text."
)

def build_entity_prompt(user_query: str, plan_model_dump: dict) -> List[Dict[str, str]]:
    """
    Ask the LLM to enumerate the concrete entities for the query (countries per region, markets per country, etc.).
    """
    hint = (
        "If a region is specified (Europe/Americas/Africa/Asia/Oceania/Middle East/World), return an exhaustive list of countries (English names).\n"
        "If entity_type is 'stock_market', list primary exchanges or stock markets per relevant country.\n"
        "Return JSON only."
    )
    demand = (
        "USER_QUERY:\n" + user_query + "\n\n" +
        "PLAN:\n" + json.dumps(plan_model_dump, ensure_ascii=False) + "\n\n" +
        "Return JSON now."
    )
    return [
        {"role": "system", "content": SYSTEM_ENTITY_ENUM},
        {"role": "user", "content": hint},
        {"role": "user", "content": demand},
    ]


SYSTEM_REGION_ENUM = (
    "Enumerate countries by macro-region. Return JSON ONLY in the shape:\n"
    "{\"entity_type\":\"country\",\"exhaustive\":true,\"entities\":[\"Country1\",\"Country2\",...]}\n"
    "Regions allowed: Europe, Americas, Africa, Asia, Oceania, Middle East, World."
)

def build_region_enum_prompt(region_name: str) -> List[Dict[str, str]]:
    """Prompt to enumerate all countries of a region."""
    return [
        {"role": "system", "content": SYSTEM_REGION_ENUM},
        {"role": "user", "content": f"Region: {region_name}\nReturn JSON now."},
    ]


SYSTEM_COMBINED_JSON = (
    "You are a data extraction and visualization planner. "
    "Your ONLY output must be one valid JSON object matching the CombinedSpec schema. "
    "Use ONLY the provided CONTEXT for facts; do not invent data. "
    "Prefer consistent identifiers (country names or ISO-3). "
    "If you compute derived metrics, define computed_fields (simple math). "
    "If aggregation/grouping is needed, specify group_by and aggregations. "
    "Do not include explanations, thought process, markdown or backticks."
)

def build_combined_prompt(
    user_query: str,
    context_jsonl: str,
    chart_hint: Optional[str],
    enforced_chart_type: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the prompt that asks the LLM to output a CombinedSpec given JSONL context.
    Token budgeting is handled by the caller; supply `context_jsonl` already trimmed.
    """
    schema_hint = (
        "CombinedSpec schema:\n"
        "{\n"
        ' "data_spec": {\n'
        '  "columns": [{"name":"string","type":"number|string"}],\n'
        '  "rows": [{"col": value, ...}],\n'
        '  "computed_fields": [{"name":"string","expression":"string"}],\n'
        '  "group_by": ["col", ...],\n'
        '  "aggregations": [{"op":"sum|avg|min|max|count","field":"col","alias":"string"}]\n'
        ' },\n'
        ' "chart_spec": {\n'
        '  "chart_type": "bar|line|pie|map-choropleth|map-heat|heatmap",\n'
        '  "title": "string",\n'
        '  "description": "string (optional)",\n'
        '  "region_hint": "string (optional)",\n'
        '  "color_legend": "string (optional)",\n'
        '  "encodings": {"x":"col","y":"col","label":"col","value":"col","location":"col","color":"col"}\n'
        ' }\n'
        "}\n"
        "Rules:\n"
        "- Build as many rows as the CONTEXT supports (avoid empty charts). If too many rows, keep at most 120 most relevant.\n"
        "- If Fitch ratings appear as letters (AAA..D with +/-), add a numeric score column (21..0) for color.\n"
        "- Ensure numeric y/value columns; omit unsupported rows.\n"
        "- For maps, set encodings.location as ISO-3 or country names; include region_hint if obvious.\n"
        f"- If the user requested a specific chart type, prefer that type: {enforced_chart_type or 'N/A'}.\n"
        "- For bar charts about stock markets and market cap, prefer encodings: x='market' or 'stock_market', y='market_cap' or 'value'.\n"
    )

    user_block = (
        f"USER_QUERY:\n{user_query}\n\n"
        "CONTEXT (JSON lines from Qdrant; each line may include keys like text, country, iso3, region, market, entity, "
        "product, section, subsection, datagroup_title, committed_date, metric, value, unit, fitch_rating, settlement_cycle, stock_market, market_cap, source):\n"
        f"{context_jsonl}\n\n"
        f"Chart hint (optional): {chart_hint or 'N/A'} | Enforced chart type: {enforced_chart_type or 'N/A'}\n\n"
        f"{schema_hint}\n"
        "Return JSON now:"
    )

    return [
        {"role": "system", "content": SYSTEM_COMBINED_JSON},
        {"role": "user", "content": user_block},
    ]
