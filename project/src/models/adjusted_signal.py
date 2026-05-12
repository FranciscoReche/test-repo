"""Adjusted predictive signals separated from raw sentiment and quality."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_adjusted_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reliability_weight = out["market_reliability_score"].fillna(0).clip(0, 100) / 100
    raw = out["raw_polymarket_probability"].fillna(0.5).clip(0, 1)
    consensus = out["consensus_implied_beat_prob"].fillna(0.5)
    divergence_boost = out["divergence_vs_consensus"].fillna(0) * reliability_weight * 0.25
    out["adjusted_beat_probability"] = (consensus * (1 - reliability_weight) + raw * reliability_weight + divergence_boost).clip(0, 1)
    price_component = (out["adjusted_beat_probability"] - 0.5) + 0.5 * out["divergence_vs_price_drift"].fillna(0)
    anomaly_penalty = np.where(out["market_reliability_score"] < 30, 0.25, 1.0)
    out["adjusted_price_reaction_signal"] = (price_component * reliability_weight * anomaly_penalty).clip(-1, 1)
    return out
