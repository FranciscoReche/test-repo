"""Ticker-event mapping and duplicate handling."""
from __future__ import annotations

import pandas as pd


def _empty_earnings_fields(earnings: pd.DataFrame) -> dict[str, object]:
    """Return NA placeholders for all prefixed earnings columns."""
    return {f"earnings_{col}": pd.NA for col in earnings.columns}


def map_markets_to_events(markets: pd.DataFrame, earnings: pd.DataFrame, tolerance_days: int = 14) -> pd.DataFrame:
    """Map each Polymarket market to the nearest plausible earnings event for the same ticker.

    Preference is given to earnings events after market creation, because mapping to an event that
    occurred before the market existed is usually a data error. Bad or ambiguous mappings are flagged
    rather than dropped so downstream quality scoring can compare filtered vs unfiltered universes.
    """
    rows: list[dict[str, object]] = []
    tol = pd.Timedelta(days=tolerance_days)
    for _, market in markets.iterrows():
        market_dict = market.to_dict()
        candidates = earnings[earnings["ticker"] == market["ticker"]].copy()
        if candidates.empty:
            rows.append({**market_dict, **_empty_earnings_fields(earnings), "event_id": pd.NA, "mapping_status": "ticker_not_found", "mapping_distance_days": pd.NA})
            continue

        created_ts = market["created_ts"]
        close_ts = market["close_ts"] if pd.notna(market["close_ts"]) else created_ts
        future_candidates = candidates[candidates["event_ts"] >= created_ts].copy()
        pool = future_candidates if not future_candidates.empty else candidates
        pool["distance"] = (pool["event_ts"] - close_ts).abs()
        best = pool.sort_values("distance").iloc[0]
        distance = best["distance"]
        status = "mapped" if distance <= tol and best["event_ts"] >= created_ts else "distant_event"
        if best["event_ts"] < created_ts:
            status = "event_before_market_creation"
        rows.append({
            **market_dict,
            **best.drop(labels=["distance"]).add_prefix("earnings_").to_dict(),
            "event_id": f"{best['ticker']}_{best['event_ts'].date()}",
            "mapping_status": status,
            "mapping_distance_days": distance / pd.Timedelta(days=1),
        })
    mapped = pd.DataFrame(rows)
    if not mapped.empty:
        mapped["duplicate_event"] = mapped.duplicated(["event_id", "market_type"], keep=False) & mapped["event_id"].notna()
    return mapped
