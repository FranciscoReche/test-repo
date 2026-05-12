"""CSV loaders with schema normalization and validation-friendly defaults."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.dates import event_timestamp, to_utc


POLYMARKET_MARKET_COLUMNS = {
    "market_id", "title", "ticker", "market_type", "resolution_rules", "resolution_source",
    "created_ts", "close_ts", "resolved_ts", "current_yes_probability", "current_no_probability",
    "total_volume", "recent_volume", "metadata", "comments", "resolution_threshold",
}
POLYMARKET_TS_COLUMNS = {"market_id", "timestamp", "yes_probability", "no_probability", "volume"}
EARNINGS_COLUMNS = {
    "ticker", "earnings_date", "earnings_time", "consensus_eps", "reported_eps",
    "consensus_revenue", "reported_revenue", "guidance", "sector", "market_cap", "market_cap_bucket",
}
EQUITY_COLUMNS = {
    "ticker", "date", "close", "open", "volume", "return_1d", "return_3d", "return_5d",
    "return_10d", "realized_volatility", "gap_open", "after_hours_return", "premarket_return",
    "post_return_1d", "post_return_3d", "post_return_5d", "expected_move",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def _ensure_columns(df: pd.DataFrame, required: set[str], source: str) -> pd.DataFrame:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    return df


def load_polymarket_markets(path: str | Path) -> pd.DataFrame:
    df = _ensure_columns(_read_csv(path), POLYMARKET_MARKET_COLUMNS, "polymarket_markets")
    for col in ("created_ts", "close_ts", "resolved_ts"):
        df[col] = to_utc(df[col])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    numeric = ["current_yes_probability", "current_no_probability", "total_volume", "recent_volume", "resolution_threshold"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def load_polymarket_timeseries(path: str | Path) -> pd.DataFrame:
    df = _ensure_columns(_read_csv(path), POLYMARKET_TS_COLUMNS, "polymarket_timeseries")
    df["timestamp"] = to_utc(df["timestamp"])
    df[["yes_probability", "no_probability", "volume"]] = df[["yes_probability", "no_probability", "volume"]].apply(pd.to_numeric, errors="coerce")
    return df.sort_values(["market_id", "timestamp"])


def load_earnings(path: str | Path) -> pd.DataFrame:
    df = _ensure_columns(_read_csv(path), EARNINGS_COLUMNS, "earnings")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["event_ts"] = event_timestamp(df["earnings_date"], df["earnings_time"])
    numeric = ["consensus_eps", "reported_eps", "consensus_revenue", "reported_revenue", "market_cap"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def load_equity_prices(path: str | Path) -> pd.DataFrame:
    df = _ensure_columns(_read_csv(path), EQUITY_COLUMNS, "equity_prices")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    numeric = sorted(EQUITY_COLUMNS - {"ticker", "date"})
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df
