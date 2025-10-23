"""
Crash ADK — Analysis Tools

These tools operate over your crash dataset and are designed to be LLM-callable
(via Google ADK). They return small, JSON-like dicts that agents can summarize.

Data backends
-------------
You can use either of two backends:
  1) Pandas (in-memory DataFrame) — easiest for prototyping.
  2) BigQuery (table or view reference) — scalable.

Choose one backend below by setting the DataStore (see the `DataStore` class).

Schema (columns used)
---------------------
X, Y, OBJECTID, Incidentid, DateTime, Year, StreetName, CrossStreet, Distance,
JunctionRelation, Totalinjuries, Totalfatalities, Injuryseverity, Collisionmanner,
Lightcondition, Weather, SurfaceCondition, Unittype_One, Age_Drv1, Gender_Drv1,
Traveldirection_One, Unitaction_One, Violation1_Drv1, AlcoholUse_Drv1,
DrugUse_Drv1, Unittype_Two, Age_Drv2, Gender_Drv2, Traveldirection_Two,
Unitaction_Two, Violation1_Drv2, AlcoholUse_Drv2, DrugUse_Drv2, Latitude,
Longitude, time_of_day_category, environmental_risk_level, crash_narrative_summary,
_data_source, _load_time, Crash_Date, Crash_Time

Notes
-----
- Year is FLOAT64 in the raw data; we cast to int when filtering.
- Crash_Date is DATE, Crash_Time is TIME; prefer these over the DateTime STRING.
- Influence and condition fields can have inconsistent casing/values.
  Normalize conservatively where needed when computing summaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import os
import math

# Optional imports; only required for the chosen backend
try:
    import pandas as pd
except Exception:
    pd = None  # Pandas mode disabled if not installed

try:
    from google.cloud import bigquery  # type: ignore
except Exception:
    bigquery = None  # BigQuery mode disabled if not installed


# ---------------------------------------------------------------------------
# Data access layer
# ---------------------------------------------------------------------------
@dataclass
class DataStore:
    """Configurable data access layer for the tools.

    Exactly one of `df` (pandas) or `bq_table` (string) should be provided.
    If both are provided, pandas takes precedence.
    """

    df: Optional["pd.DataFrame"] = None
    bq_table: Optional[str] = None  # e.g., "project.dataset.table"
    bq_client: Optional["bigquery.Client"] = None

    # ---------------------
    # Pandas helpers
    # ---------------------
    def has_pandas(self) -> bool:
        return self.df is not None and pd is not None

    # ---------------------
    # BigQuery helpers
    # ---------------------
    def _ensure_bq(self) -> None:
        if self.bq_table is None:
            raise RuntimeError("BigQuery backend not configured: bq_table is None")
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed")
        if self.bq_client is None:
            self.bq_client = bigquery.Client()

    def _bq_query(self, sql: str) -> List[dict]:
        self._ensure_bq()
        job = self.bq_client.query(sql)
        rows = list(job.result())
        return [dict(r) for r in rows]


# Global store (set this from your app / notebook)
DATA_STORE = DataStore()


# ---------------------------------------------------------------------------
# Utility functions (filters & normalization) — used by both backends
# ---------------------------------------------------------------------------

def _coerce_int_year(val) -> Optional[int]:
    try:
        if pd is not None and pd.isna(val):
            return None
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _apply_common_filters_df(
    df: "pd.DataFrame",
    *,
    year: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    time_of_day: Optional[str] = None,
    junction_relation: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> "pd.DataFrame":
    filt = df.copy()

    # cast Year -> int for filtering
    if "Year" in filt.columns:
        filt["Year_int"] = filt["Year"].apply(_coerce_int_year)
    else:
        filt["Year_int"] = None

    if year is not None:
        filt = filt[filt["Year_int"] == int(year)]

    if date_start is not None:
        filt = filt[filt["Crash_Date"] >= date_start]
    if date_end is not None:
        filt = filt[filt["Crash_Date"] <= date_end]

    if time_start is not None:
        filt = filt[filt["Crash_Time"] >= time_start]
    if time_end is not None:
        filt = filt[filt["Crash_Time"] <= time_end]

    if time_of_day is not None and "time_of_day_category" in filt.columns:
        filt = filt[filt["time_of_day_category"].str.lower() == str(time_of_day).lower()]

    if junction_relation is not None and "JunctionRelation" in filt.columns:
        filt = filt[filt["JunctionRelation"].str.lower() == str(junction_relation).lower()]

    if bbox is not None and all(col in filt.columns for col in ["Latitude", "Longitude"]):
        min_lat, min_lon, max_lat, max_lon = bbox
        filt = filt[(filt["Latitude"] >= min_lat) & (filt["Latitude"] <= max_lat) &
                    (filt["Longitude"] >= min_lon) & (filt["Longitude"] <= max_lon)]

    return filt


def _sum_safe(series) -> int:
    try:
        return int(series.fillna(0).astype(float).sum())
    except Exception:
        try:
            return int(series.sum())
        except Exception:
            return 0


def _top_counts(df: "pd.DataFrame", col: str, n: int) -> List[Dict]:
    if col not in df.columns:
        return []
    s = (df[col].fillna("Unknown").astype(str).str.strip()
                    .replace({"": "Unknown"}))
    counts = s.value_counts().head(n)
    return [{col: k, "crashes": int(v)} for k, v in counts.items()]


def _rates(injuries: int, fatalities: int, crashes: int) -> Tuple[float, float]:
    if crashes <= 0:
        return (0.0, 0.0)
    return (round(100.0 * injuries / crashes, 2), round(100.0 * fatalities / crashes, 3))


def _normalize_flag(val: Optional[str]) -> str:
    if val is None:
        return "unknown"
    v = str(val).strip().lower()
    if v in ("", "\\n", "\\t", "\\r"):
        return "unknown"
    # Common mappings
    if v in ("alcohol", "alcohol use", "pos", "positive", "+"):
        return "positive"
    if v in ("no apparent influence", "none", "neg", "negative", "-", "no"):
        return "negative"
    if v in ("drugs", "drug", "drug use"):
        return "positive"  # for drug fields, presence implies positive
    return v


# ---------------------------------------------------------------------------
# Public API — Tool functions (Pandas backend first; BQ fallbacks at end)
# ---------------------------------------------------------------------------

def get_crash_summary(
    year: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    time_of_day: Optional[str] = None,
    junction_relation: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    top_n: int = 5,
) -> Dict:
    """Summarize overall crash statistics for the given filters.

    Returns dict with totals and top categories (collision manner, violations, light).
    """
    if DATA_STORE.has_pandas():
        df = _apply_common_filters_df(
            DATA_STORE.df,
            year=year,
            date_start=date_start,
            date_end=date_end,
            time_start=time_start,
            time_end=time_end,
            time_of_day=time_of_day,
            junction_relation=junction_relation,
            bbox=bbox,
        )
        total_crashes = len(df)
        total_injuries = _sum_safe(df.get("Totalinjuries", 0))
        total_fatalities = _sum_safe(df.get("Totalfatalities", 0))
        top_collision = _top_counts(