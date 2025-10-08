"""
Module: render.py
Purpose:
    Chart rendering + validations using Plotly Express:
      - render_chart_from_df(): renders bar/line/pie/map/heatmap with sensible defaults
      - _chart_is_sparse(): signals blank/sparse specs for rescue runs
      - Small helpers to resolve columns and coerce heatmap→map-heat when axis is countries

Fix notes (empty-map hotfix):
    - For map charts, if no numeric 'value' is provided or all values become NaN after filtering,
      we fallback to a presence indicator (=1) so the map is never blank.
    - Keeps BAR explicit behavior (never auto-convert to map).

Usage:
    from rag_plotting.render import render_chart_from_df, _chart_is_sparse
"""
from __future__ import annotations
from typing import Optional, Dict
import pandas as pd
import numpy as np
import plotly.express as px
import re

from rag_plotting.schemas import ChartSpec

def _resolve_col(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    if name in df.columns:
        return name
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(name.lower(), None)

def _filter_valid_rows(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[metric_col], errors="coerce")
    keep = s.notna() & np.isfinite(s)
    return df.loc[keep].copy()

def _chart_is_sparse(df: pd.DataFrame, chart: ChartSpec) -> bool:
    t = chart.chart_type
    enc = chart.encodings or {}

    if t == "pie":
        names = _resolve_col(df, enc.get("label") or enc.get("x"))
        values = _resolve_col(df, enc.get("value") or enc.get("y"))
        if not names:
            return True
        if values:
            dff = _filter_valid_rows(df, values)
        else:
            dff = df.copy()
        return dff[names].nunique(dropna=True) < 2

    if t == "bar":
        x = _resolve_col(df, enc.get("x") or enc.get("label") or chart.x_axis)
        y = _resolve_col(df, enc.get("y") or enc.get("value") or chart.y_axis)
        if not x or not y:
            return True
        dff = _filter_valid_rows(df, y)
        return dff[x].nunique(dropna=True) < 2

    if t in ("map-choropleth", "map-heat"):
        location = _resolve_col(df, enc.get("location") or enc.get("country") or enc.get("label") or enc.get("x"))
        return (not location) or (len(df.dropna(subset=[location])) < 2)

    if t == "heatmap":
        xcol = _resolve_col(df, enc.get("x")); ycol = _resolve_col(df, enc.get("y")); vcol = _resolve_col(df, enc.get("value"))
        return (not xcol) or (not ycol) or (not vcol) or df.empty

    return False

def _clean_encodings(enc: Dict[str, Optional[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (enc or {}).items():
        if isinstance(v, str) and v.strip():
            out[k] = v
    return out

def _coerce_country_heatmap_to_map(chart: ChartSpec, df: pd.DataFrame) -> ChartSpec:
    if chart.chart_type != "heatmap":
        return chart
    enc = chart.encodings or {}
    candidates = [enc.get("location"), enc.get("country"), enc.get("label"), enc.get("x"), "country", "Country"]
    country_col = next((c for c in candidates if c and c in df.columns), None)
    if country_col:
        value_col = enc.get("value") or enc.get("y")
        if not value_col:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            value_col = num_cols[0] if num_cols else None
        if value_col:
            chart.chart_type = "map-heat"
            chart.encodings = _clean_encodings({"location": country_col, "value": value_col})
    return chart

def render_chart_from_df(df: pd.DataFrame, chart: ChartSpec):
    print(f"[Render] chart_type={chart.chart_type} title='{chart.title}'")
    if chart.chart_type == "heatmap":
        chart = _coerce_country_heatmap_to_map(chart, df)

    enc = chart.encodings or {}

    if chart.chart_type == "pie":
        names = _resolve_col(df, enc.get("label") or enc.get("x"))
        values = _resolve_col(df, enc.get("value") or enc.get("y"))
        if not names:
            # elige primera string como names
            str_cols = [c for c in df.columns if getattr(df[c], "dtype", None) == "string"]
            names = str_cols[0] if str_cols else df.columns[0]
        dff = df.copy()
        if values is None:
            dff["_pie_value"] = 1
            values = "_pie_value"
        else:
            dff = _filter_valid_rows(df, values)
            if dff.empty:
                dff = df.copy()
                dff["_pie_value_fallback"] = 1
                values = "_pie_value_fallback"
        fig = px.pie(dff, names=names, values=values, title=chart.title)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        fig.show()
        return fig

    if chart.chart_type == "bar":
        x = _resolve_col(df, enc.get("x") or enc.get("label") or chart.x_axis)
        y = _resolve_col(df, enc.get("y") or enc.get("value") or chart.y_axis)
        if not x:
            str_cols = [c for c in df.columns if getattr(df[c], "dtype", None) == "string"]
            x = (str_cols[0] if str_cols else (df.columns[0] if len(df.columns) else None))
        if (not y) or (y not in df.columns):
            df = df.copy()
            df["_bar_value"] = 1
            y = "_bar_value"
        dff = _filter_valid_rows(df, y)
        if dff.empty:
            dff = df.copy()
            dff["_bar_value_fallback"] = 1
            y = "_bar_value_fallback"
        if not x:
            x = next(iter(dff.columns), "label")
            if x == y and len(dff.columns) > 1:
                x = [c for c in dff.columns if c != y][0]
        fig = px.bar(dff, x=x, y=y, title=chart.title, labels={x: x, y: y})
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        fig.show()
        return fig

    if chart.chart_type in ("map-choropleth", "map-heat"):
        location = _resolve_col(df, enc.get("location") or enc.get("country") or enc.get("label") or enc.get("x"))
        value = _resolve_col(df, enc.get("value") or enc.get("y"))
        if not location:
            raise ValueError("[Render][map] 'location' is required.")
        if value is None:
            df = df.copy()
            df["_map_value"] = 1
            value = "_map_value"
        dff = _filter_valid_rows(df, value)
        if dff.empty:
            dff = df.copy()
            dff["_map_value_fallback"] = 1
            value = "_map_value_fallback"
        sample = dff[location].dropna().astype(str).head(5)
        loc_mode = "ISO-3" if (len(sample) > 0 and sample.str.fullmatch(r"[A-Z]{3}").all()) else "country names"
        fig = px.choropleth(dff, locations=location, color=value, color_continuous_scale="Blues",
                            title=chart.title, locationmode=loc_mode)
        rh = (chart.region_hint or "").lower()
        if rh == "europe": fig.update_geos(scope="europe")
        if rh == "africa": fig.update_geos(scope="africa")
        if rh == "asia": fig.update_geos(scope="asia")
        if rh == "north america": fig.update_geos(scope="north america")
        if rh == "south america": fig.update_geos(scope="south america")
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), coloraxis_colorbar_title=chart.color_legend or "")
        fig.show()
        return fig

    if chart.chart_type == "heatmap":
        xcol = _resolve_col(df, enc.get("x")); ycol = _resolve_col(df, enc.get("y")); vcol = _resolve_col(df, enc.get("value"))
        if not (xcol and ycol and vcol):
            raise ValueError("[Render][heatmap] Need x,y,value.")
        df2 = _filter_valid_rows(df, vcol)
        pivot = df2.pivot_table(index=ycol, columns=xcol, values=vcol, aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", title=chart.title)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), xaxis_title=xcol, yaxis_title=ycol)
        fig.show()
        return fig

    raise NotImplementedError(f"[Render] Unknown chart type: {chart.chart_type}")
