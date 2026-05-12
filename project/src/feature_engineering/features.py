"""Feature engineering for raw signal, liquidity, quality, divergence and context."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering.snapshots import build_probability_snapshots


def _safe_log_odds(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").clip(1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _pre_event_signal(base: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    """Return latest probability observed at or before the earnings timestamp.

    This prevents using post-event/resolution probability as the raw Polymarket signal when a market
    closes after the earnings announcement or when the raw market snapshot was exported later.
    """
    rows: list[dict[str, object]] = []
    for _, market in base.iterrows():
        event_ts = market.get("earnings_event_ts")
        market_id = market["market_id"]
        ts = timeseries[timeseries["market_id"] == market_id].dropna(subset=["timestamp"]).sort_values("timestamp")
        pre = ts[ts["timestamp"] <= event_ts] if pd.notna(event_ts) else ts.iloc[0:0]
        if pre.empty:
            rows.append({"market_id": market_id, "pre_event_yes_probability": pd.NA, "raw_signal_timestamp": pd.NaT, "cutoff_lag_hours": pd.NA})
            continue
        last = pre.iloc[-1]
        rows.append({
            "market_id": market_id,
            "pre_event_yes_probability": last["yes_probability"],
            "raw_signal_timestamp": last["timestamp"],
            "cutoff_lag_hours": (event_ts - last["timestamp"]) / pd.Timedelta(hours=1),
        })
    return pd.DataFrame(rows)


def _timeseries_features(mapped: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, market in mapped.iterrows():
        market_id = market["market_id"]
        event_ts = market.get("earnings_event_ts")
        ts = timeseries[timeseries["market_id"] == market_id].dropna(subset=["timestamp"]).sort_values("timestamp")
        if pd.notna(event_ts):
            ts = ts[ts["timestamp"] <= event_ts]
        out = {"market_id": market_id, "n_probability_updates": len(ts)}
        if ts.empty:
            rows.append(out)
            continue
        probs = ts["yes_probability"].astype(float)
        deltas = probs.diff().abs().fillna(0)
        gap_hours = ts["timestamp"].diff().dt.total_seconds().dropna() / 3600
        out.update({
            "pricing_stability": float(1 - min(deltas.std(ddof=0) if len(deltas) > 1 else 0, 1)),
            "prob_total_movement": float(deltas.sum()),
            "prob_net_movement": float(abs(probs.iloc[-1] - probs.iloc[0])) if len(probs) > 1 else 0.0,
            "movement_persistence": float(abs(probs.iloc[-1] - probs.iloc[0]) / deltas.sum()) if deltas.sum() > 0 else 0.0,
            "max_probability_jump": float(deltas.max()),
            "time_gap_hours_max": float(gap_hours.max()) if not gap_hours.empty else 0.0,
            "activity_regularity": float(1 / (1 + gap_hours.std())) if len(gap_hours) > 1 and pd.notna(gap_hours.std()) else 0.0,
            "volume_concentration": float(ts["volume"].max() / ts["volume"].sum()) if ts["volume"].sum() > 0 else 0.0,
        })
        if pd.notna(event_ts):
            last_24h = ts[(ts["timestamp"] >= event_ts - pd.Timedelta(hours=24)) & (ts["timestamp"] <= event_ts)]
            out["late_move_intensity"] = float(last_24h["yes_probability"].diff().abs().sum()) if len(last_24h) > 1 else 0.0
            out["late_volume"] = float(last_24h["volume"].sum()) if not last_24h.empty else 0.0
            out["freshness_hours"] = float((event_ts - ts["timestamp"].max()) / pd.Timedelta(hours=1))
        rows.append(out)
    return pd.DataFrame(rows)


def _merge_equity_at_event(events: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Attach the equity row for the event date, not the latest row in the dataset.

    Using the latest row by ticker leaks future ACME-style events into earlier quarters. This function
    picks the most recent equity row with date <= earnings event date for each market/event.
    """
    rows: list[pd.Series] = []
    for _, event in events.iterrows():
        ticker = event["ticker"]
        event_ts = event.get("earnings_event_ts")
        candidates = equity[equity["ticker"] == ticker].copy()
        if pd.notna(event_ts):
            candidates = candidates[candidates["date"] <= event_ts]
        if candidates.empty:
            rows.append(pd.Series(dtype="object", name=event.name))
        else:
            rows.append(candidates.sort_values("date").iloc[-1].add_prefix("equity_"))
    equity_at_event = pd.DataFrame(rows).reset_index(drop=True)
    for col in equity.add_prefix("equity_").columns:
        if col not in equity_at_event.columns:
            equity_at_event[col] = pd.NA
    return pd.concat([events.reset_index(drop=True), equity_at_event.reset_index(drop=True)], axis=1)


def build_features(mapped: pd.DataFrame, timeseries: pd.DataFrame, equity: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    base = mapped.copy()
    pre_signal = _pre_event_signal(base, timeseries)
    base = base.merge(pre_signal, on="market_id", how="left")
    p = pd.to_numeric(base["pre_event_yes_probability"], errors="coerce").combine_first(pd.to_numeric(base["current_yes_probability"], errors="coerce"))
    base["raw_polymarket_probability"] = p
    base["current_no_probability_clean"] = 1 - p
    base["distance_to_50"] = (p - 0.5).abs()
    base["log_odds"] = _safe_log_odds(p)
    base["market_age_hours"] = (base["close_ts"] - base["created_ts"]).dt.total_seconds() / 3600
    base["time_remaining_hours"] = (base["earnings_event_ts"] - base["created_ts"]).dt.total_seconds() / 3600
    base["volume_per_hour_remaining"] = base["total_volume"] / base["time_remaining_hours"].clip(lower=1)
    base["rule_clarity_score"] = np.where(base["resolution_rules"].fillna("").str.len() > 25, 1.0, 0.4)
    base["resolution_source_score"] = np.where(base["resolution_source"].fillna("").str.len() > 3, 1.0, 0.3)
    ts_feats = _timeseries_features(base, timeseries)
    snaps = build_probability_snapshots(base, timeseries, windows)
    df = base.merge(ts_feats, on="market_id", how="left").merge(snaps, on="market_id", how="left")
    df = _merge_equity_at_event(df, equity)
    df["consensus_implied_beat_prob"] = np.where(df["earnings_consensus_eps"].notna(), 0.5, np.nan)
    df["divergence_vs_consensus"] = df["raw_polymarket_probability"] - df["consensus_implied_beat_prob"]
    drift = df[["equity_return_1d", "equity_return_3d", "equity_return_5d", "equity_return_10d"]].mean(axis=1)
    df["recent_price_drift"] = drift
    df["divergence_vs_price_drift"] = df["raw_polymarket_probability"] - (0.5 + drift.fillna(0)).clip(0, 1)
    df["prob_price_direction_divergence"] = np.sign(df["raw_polymarket_probability"] - 0.5) != np.sign(drift.fillna(0))
    df["volume_expected_move_ratio"] = df["total_volume"] / df["equity_expected_move"].abs().replace(0, np.nan)
    return df
