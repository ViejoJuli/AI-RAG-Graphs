"""
Module: aliases_filters.py
Purpose:
    Houses lightweight domain constants and small helpers used everywhere:
    - Canonical region aliases and subregions
    - Country aliases (normalization)
    - Domain synonyms
    - Region normalization and Qdrant filter builders
    - Payload standardization helpers (flatten qdrant payload → row)

Why separate?
    These are pure-Python tables + tiny functions, hot on the call path.
    Keeping them in one module avoids import sprawl and preserves speed.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from qdrant_client.http import models as qmodels  # type: ignore
except Exception:  # pragma: no cover
    qmodels = None  # type: ignore

# =========================
# Canonical region aliases
# =========================
REGION_ALIASES = {
    "europe": ["Europe", "EUROPE", "europe", "UE", "EU"],
    "americas": [
        "Americas","AMERICAS","america","America","LATAM","LatAm",
        "latin america","américa","north america","south america",
        "caribbean","central america"
    ],
    "africa": ["Africa","AFRICA","africa","África"],
    "asia": ["Asia","ASIA","asia"],
    "oceania": ["Oceania","oceania","Pacific","pacific","Oceanía"],
    "middle east": ["Middle East","MIDDLE EAST","middle east","MENA","mena","Gulf","gulf","oriente medio","medio oriente"],
    "world": ["World","Global","global","earth","all countries","the world"],
}

AMERICAS_SUBREGIONS = ["Americas","North America","South America","Latin America","Caribbean","Central America"]

COUNTRY_ALIASES = {
    "us":"United States","usa":"United States","u.s.":"United States","u.s.a.":"United States","united states of america":"United States",
    "uk":"United Kingdom","u.k.":"United Kingdom","britain":"United Kingdom","great britain":"United Kingdom",
    "ivory coast":"Ivory Coast","cote d'ivoire":"Ivory Coast","côte d’ivoire":"Ivory Coast",
    "drc":"Democratic Republic of the Congo","congo drc":"Democratic Republic of the Congo",
    "uae":"United Arab Emirates","u.a.e.":"United Arab Emirates",
    "south korea":"South Korea","north korea":"North Korea",
    "czech republic":"Czechia",
}

SYN_DEFAULT = ["definition", "status", "list", "overview"]
SYN_COUNTRY = ["country policy", "regulation", "statute", "requirement"]
SYN_FITCH   = ["Fitch rating", "Fitch Ratings", "credit rating Fitch", "sovereign rating Fitch", "sovereign credit score"]
SYN_SETTLE  = ["settlement cycle T+2", "T+2 settlement", "settlement period", "post-trade cycle"]
SYN_MARKET  = ["stock market", "stock exchange", "primary exchange", "bourse", "market cap", "capitalization USD", "market capitalization"]

# =========================
# Normalization helpers
# =========================
def normalize_region(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    for canon, aliases in REGION_ALIASES.items():
        if key == canon or key in [a.lower() for a in aliases]:
            return "Middle East" if canon == "middle east" else canon.title()
    if key in ["world","global","all","earth"]:
        return "World"
    return None

# =========================
# Qdrant filters
# =========================
def _match_any(values: List[str]):
    if qmodels is None:
        return None
    clean = list(dict.fromkeys([v for v in values if v]))
    return qmodels.MatchAny(any=clean)

def _region_filter(region: Optional[str]):
    if qmodels is None or not region:
        return None
    canon = normalize_region(region) or region
    candidates = [canon, canon.title(), canon.upper(), canon.lower()]
    if canon == "Americas":
        candidates += AMERICAS_SUBREGIONS + [c.upper() for c in AMERICAS_SUBREGIONS] + [c.lower() for c in AMERICAS_SUBREGIONS]
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="region", match=_match_any(list(dict.fromkeys(candidates))))]  # type: ignore
    )

def _any_field_filter(key: str, values: List[str]):
    if qmodels is None:
        return None
    values = [v for v in values if v]
    if not values:
        return None
    return qmodels.Filter(must=[qmodels.FieldCondition(key=key, match=_match_any(values))])  # type: ignore

def _text_filter(key: str, text: str):
    """
    Full-text filter (Qdrant FT). Si FT no está disponible, cae a igualdad suave.
    """
    if qmodels is None or not text:
        return None
    try:
        mt = qmodels.MatchText(text=text)  # requiere FT index en el campo
    except Exception:
        mt = _match_any([text])  # fallback
    return qmodels.Filter(must=[qmodels.FieldCondition(key=key, match=mt)])  # type: ignore

def _merge_filters(*filters):
    if qmodels is None:
        return None
    must = []
    for f in filters:
        if not f:
            continue
        must.extend(f.must or [])
    return qmodels.Filter(must=must) if must else None

# =========================
# Payload → row standardizer
# =========================
def _extract_payload_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text") or payload.get("content") or payload.get("chunk") or ""
    return {
        "text": text,
        "country": payload.get("country") or payload.get("Country"),
        "iso3": payload.get("iso3") or payload.get("ISO3") or payload.get("iso"),
        "region": payload.get("region") or payload.get("Region"),
        "market": payload.get("market") or payload.get("Market"),
        "entity": payload.get("entity") or payload.get("Entity"),
        "product": payload.get("product") or payload.get("Product"),
        "section": payload.get("section") or payload.get("Section"),
        "subsection": payload.get("subsection") or payload.get("Subsection"),
        "datagroup_title": payload.get("datagroup_title") or payload.get("data_group_title"),
        "committed_date": payload.get("committed_date") or payload.get("date") or payload.get("Date"),
        "metric": payload.get("metric") or payload.get("Metric") or payload.get("field"),
        "value": payload.get("value") or payload.get("Value") or payload.get("numeric_value"),
        "unit": payload.get("unit") or payload.get("Unit"),
        "fitch_rating": payload.get("fitch_rating") or payload.get("Fitch") or payload.get("rating"),
        "settlement_cycle": payload.get("settlement_cycle") or payload.get("settlement") or payload.get("t_plus"),
        "stock_market": payload.get("stock_market") or payload.get("exchange") or payload.get("Exchange"),
        "market_cap": payload.get("market_cap") or payload.get("marketcap") or payload.get("MarketCap"),
        "source": payload.get("source") or payload.get("Source") or payload.get("url"),
        "_score": payload.get("_score"),
    }
