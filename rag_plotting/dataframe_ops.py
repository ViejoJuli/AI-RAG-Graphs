"""
Module: dataframe_ops.py
Purpose:
    Build/transform pandas DataFrames from DataSpec with safe coercions:
      - build_dataframe_from_data_spec(): types, computed fields, groupby/aggregations
      - _maybe_add_fitch_numeric(): map Fitch letter rating → numeric column
      - _apply_computed_fields(): SAFE expr eval per row (whitelisted AST)
      - _apply_groupby_aggs(): standard aggregations with predictable column names

Design:
    - Avoids heavy logic: pure pandas ops with small helpers.
    - SAFE eval forbids arbitrary code by whitelisting AST nodes/functions.

Usage:
    from rag_plotting.dataframe_ops import build_dataframe_from_data_spec
"""

from __future__ import annotations
from typing import Any, Dict, List, Set
import ast
import re
import pandas as pd
import numpy as np

from rag_plotting.schemas import DataSpec, DataColumn, ComputedField, Aggregation


# --------- type coercions ----------
def _coerce_types(df: pd.DataFrame, columns: List[DataColumn]) -> pd.DataFrame:
    for col in columns:
        if col.name not in df.columns:
            df[col.name] = pd.Series(dtype=("float64" if col.type == "number" else "string"))
        if col.type == "number":
            df[col.name] = pd.to_numeric(df[col.name], errors="coerce")
        else:
            df[col.name] = df[col.name].astype("string")
    return df


# --------- Fitch letter → numeric ----------
_FITCH_SCALE = {
    "AAA": 21, "AA+": 20, "AA": 19, "AA-": 18, "A+": 17, "A": 16, "A-": 15,
    "BBB+": 14, "BBB": 13, "BBB-": 12, "BB+": 11, "BB": 10, "BB-": 9,
    "B+": 8, "B": 7, "B-": 6, "CCC+": 5, "CCC": 4, "CCC-": 3, "CC": 2, "C": 1, "RD": 0, "D": 0
}

def _maybe_add_fitch_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cand_cols = [c for c in df.columns if re.search(r"fitch.*rating", c, flags=re.I)]
    if not cand_cols:
        return df
    col = cand_cols[0]

    def _map(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v).strip().upper().replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", "", s)
        return _FITCH_SCALE.get(s, np.nan)

    new_col = re.sub(r"(?i)rating", "rating_score", col)
    df[new_col] = df[col].map(_map)
    return df


# --------- SAFE row expr eval ----------
_ALLOWED_AST_NODES: Set[type] = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Load, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Call
}
_ALLOWED_FUNC_NAMES = {"abs", "round", "min", "max"}

def _safe_eval_expr(expr: str, scope: Dict[str, Any]) -> Any:
    def _check(node: ast.AST):
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            _check(child)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNC_NAMES:
                raise ValueError("Disallowed function call.")
    tree = ast.parse(expr, mode="eval"); _check(tree)
    code = compile(tree, "<expr>", "eval")
    return eval(code, {"__builtins__": {}}, scope)


def _apply_computed_fields(df: pd.DataFrame, computed_fields: List[ComputedField]) -> pd.DataFrame:
    if not computed_fields:
        return df
    for cf in computed_fields:
        def _row_eval(row):
            scope = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            try:
                return _safe_eval_expr(cf.expression, scope)
            except Exception:
                return None
        df[cf.name] = df.apply(_row_eval, axis=1)
    return df


# --------- aggregations ----------
def _apply_groupby_aggs(df: pd.DataFrame, group_by: List[str], aggs: List[Aggregation]) -> pd.DataFrame:
    if not group_by and not aggs:
        return df
    if not aggs:
        return df.drop_duplicates(subset=group_by)

    gb = df.groupby(group_by, dropna=False) if group_by else df.assign(_grp_all=0).groupby("_grp_all", dropna=False)
    agg_map: Dict[str, List[str]] = {}
    for a in aggs:
        agg_map.setdefault(a.field, []).append({"sum":"sum","avg":"mean","min":"min","max":"max","count":"count"}[a.op])

    res = gb.agg(agg_map)
    res.columns = ["__".join([c for c in col if c]) for col in res.columns.values]
    res = res.reset_index()

    rename_map: Dict[str, str] = {}
    for a in aggs:
        key = f"{a.field}__" + {"sum":"sum","avg":"mean","min":"min","max":"max","count":"count"}[a.op]
        if key in res.columns:
            rename_map[key] = a.alias
    res = res.rename(columns=rename_map)

    if "_grp_all" in res.columns:
        res = res.drop(columns=["_grp_all"])
    return res


# --------- public API ----------
def build_dataframe_from_data_spec(data_spec: DataSpec) -> pd.DataFrame:
    """
    Build a DataFrame from DataSpec with safe coercions, computed fields, and aggregations.
    """
    print(f"[DF] Building DataFrame… columns={len(data_spec.columns)} rows={len(data_spec.rows)}")
    df = pd.DataFrame(data_spec.rows or [])
    df = _coerce_types(df, data_spec.columns)
    df = _maybe_add_fitch_numeric(df)
    df = _apply_computed_fields(df, data_spec.computed_fields)
    df = _apply_groupby_aggs(df, data_spec.group_by, data_spec.aggregations)
    print(f"[DF] Done df.shape={df.shape}")
    return df
