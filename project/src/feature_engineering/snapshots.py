"""Temporal snapshot extraction for pre-event Polymarket probabilities."""
from __future__ import annotations

import pandas as pd


def parse_window(window: str) -> pd.Timedelta:
    unit = window[-1].lower()
    value = int(window[:-1])
    return pd.Timedelta(days=value) if unit == "d" else pd.Timedelta(hours=value) if unit == "h" else pd.Timedelta(minutes=value)


def build_probability_snapshots(mapped: pd.DataFrame, timeseries: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for _, market in mapped.iterrows():
        market_id = market["market_id"]
        event_ts = market.get("earnings_event_ts")
        ts = timeseries[timeseries["market_id"] == market_id].dropna(subset=["timestamp"]).sort_values("timestamp")
        out = {"market_id": market_id}
        if pd.isna(event_ts) or ts.empty:
            rows.append(out)
            continue
        pre = ts[ts["timestamp"] <= event_ts]
        for w in windows:
            cutoff = event_ts - parse_window(w)
            eligible = pre[pre["timestamp"] <= cutoff]
            label = w
            if eligible.empty:
                out[f"prob_t_minus_{label}"] = pd.NA
                out[f"vol_t_minus_{label}"] = pd.NA
            else:
                last = eligible.iloc[-1]
                out[f"prob_t_minus_{label}"] = last["yes_probability"]
                out[f"vol_t_minus_{label}"] = last["volume"]
        rows.append(out)
    df = pd.DataFrame(rows)
    # window-to-window changes in chronological order from far to near
    prob_cols = [f"prob_t_minus_{w}" for w in windows if f"prob_t_minus_{w}" in df.columns]
    for prev, cur in zip(prob_cols, prob_cols[1:]):
        df[f"chg_{prev}_to_{cur}"] = pd.to_numeric(df[cur], errors="coerce") - pd.to_numeric(df[prev], errors="coerce")
    return df
