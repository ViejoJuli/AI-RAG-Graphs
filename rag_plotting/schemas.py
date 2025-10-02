"""
Module: schemas.py
Purpose:
    Central Pydantic models and type aliases used by planning, LLM I/O, and rendering:
    - ChartType, DataColumn, ComputedField, Aggregation
    - DataSpec, ChartSpec, RetrievalHints, QueryPlan, EntityList, CombinedSpec

Design:
    - Mirrors your monolith’s field names to avoid refactors.
    - No business logic here—pure data contracts.
    - Keep imports minimal (typing + pydantic).

Usage:
    from rag_plotting.schemas import (
        ChartType, DataColumn, ComputedField, Aggregation,
        DataSpec, ChartSpec, RetrievalHints, QueryPlan,
        EntityList, CombinedSpec
    )
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ChartType = Literal["bar", "line", "pie", "map-choropleth", "map-heat", "heatmap"]


class DataColumn(BaseModel):
    name: str
    type: Literal["number", "string"]


class ComputedField(BaseModel):
    name: str
    expression: str


class Aggregation(BaseModel):
    op: Literal["sum", "avg", "min", "max", "count"]
    field: str
    alias: str


class DataSpec(BaseModel):
    columns: List[DataColumn]
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    computed_fields: List[ComputedField] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    aggregations: List[Aggregation] = Field(default_factory=list)
    notes: Optional[str] = None


class ChartSpec(BaseModel):
    chart_type: ChartType
    title: str
    description: Optional[str] = None
    region_hint: Optional[str] = None
    color_legend: Optional[str] = None
    encodings: Dict[str, str] = Field(default_factory=dict)
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None


class RetrievalHints(BaseModel):
    scale: Optional[Literal["small", "medium", "large"]] = None
    expected_entities: Optional[int] = None
    k_each: Optional[int] = None
    k_total: Optional[int] = None
    max_context: Optional[int] = None
    max_subqueries: Optional[int] = None
    max_entities: Optional[int] = None


class QueryPlan(BaseModel):
    normalized_question: str
    chart_hint: Optional[str] = None
    region: Optional[str] = None  # Europe|Americas|Africa|Asia|Oceania|Middle East|World
    task_kind: Optional[str] = None  # e.g., "country_metric_map", "market_bar"
    entity_type: Optional[str] = None  # "country", "stock_market", ...
    entities: List[str] = []
    constraints: Dict[str, str] = {}
    synonyms: List[str] = []
    subqueries: List[str] = []
    retrieval_hints: Optional[RetrievalHints] = None


class EntityList(BaseModel):
    entity_type: str
    exhaustive: bool
    entities: List[str] = []


class CombinedSpec(BaseModel):
    data_spec: DataSpec
    chart_spec: ChartSpec
