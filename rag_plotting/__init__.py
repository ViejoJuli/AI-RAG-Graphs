"""
Package: rag_plotting
Purpose:
    Convenience exports for primary pipeline pieces so downstream code can:
        from rag_plotting import interpret_query, enumerate_entities, build_search_queries, rrf_merge_rows,
                               call_model_for_combined_spec, build_dataframe_from_data_spec

    Keeps import paths short and stable.

Notes:
    Export only the core, performance-critical APIs.
"""

from .config import *
from .features import *
from .aliases_filters import *
from .embeddings import get_embedding
from .qdrant_io import search_qdrant_with_query
from .schemas import (
    ChartType, DataColumn, ComputedField, Aggregation,
    DataSpec, ChartSpec, RetrievalHints, QueryPlan,
    EntityList, CombinedSpec
)
from .prompts import (
    build_plan_prompt, build_entity_prompt, build_region_enum_prompt,
    build_combined_prompt, detect_requested_chart_type
)
from .planning_enum import interpret_query, enumerate_entities
from .expansion import build_search_queries, _build_keywords
from .fuse import rrf_merge_rows
from .combined_build import call_model_for_combined_spec
from .dataframe_ops import build_dataframe_from_data_spec
