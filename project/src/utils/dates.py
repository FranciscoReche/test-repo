"""Date/time normalization utilities."""
from __future__ import annotations

import pandas as pd


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def event_timestamp(date_col: pd.Series, time_bucket: pd.Series | None = None) -> pd.Series:
    """Build a UTC event timestamp using conservative BMO/AMC conventions."""
    base = pd.to_datetime(date_col, errors="coerce", utc=True)
    if time_bucket is None:
        return base
    offsets = time_bucket.astype(str).str.upper().map({"BMO": 12, "AMC": 21, "INTRADAY": 16}).fillna(16)
    return base + pd.to_timedelta(offsets, unit="h")
