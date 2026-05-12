"""Quantitative anomaly detection without making insider-trading claims."""
from __future__ import annotations

import numpy as np
import pandas as pd

def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def robust_rank(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce").fillna(0).rank(pct=True)
    out["liquidity_adjusted_jump"] = out["max_probability_jump"].fillna(0) / np.log1p(out["total_volume"].fillna(0)).replace(0, np.nan)
    out["abnormal_jump_score"] = 100 * robust_rank("liquidity_adjusted_jump")
    out["abnormal_activity_score"] = 100 * robust_rank("volume_concentration")
    out["divergence_score"] = 100 * pd.concat([
        out["divergence_vs_consensus"].abs().fillna(0),
        out["divergence_vs_price_drift"].abs().fillna(0),
    ], axis=1).mean(axis=1).rank(pct=True)
    out["late_move_score"] = 100 * robust_rank("late_move_intensity") * out["liquidity_score"].fillna(0).clip(0, 1)
    out["surprise_to_liquidity_ratio"] = out.get("eps_surprise_abs", 0).abs() / np.log1p(out["total_volume"].fillna(0)).replace(0, np.nan)
    out["anomaly_score"] = (0.30 * out["late_move_score"] + 0.25 * out["abnormal_jump_score"] + 0.20 * out["abnormal_activity_score"] + 0.25 * out["divergence_score"]).fillna(0).clip(0, 100).round(2)
    conditions = [
        out["market_reliability_score"] < 30,
        out["anomaly_score"] >= 85,
        out["anomaly_score"] >= 65,
        out["late_move_score"] >= 50,
    ]
    choices = ["mercado ilíquido poco fiable", "anomalía fuerte", "posible flujo informado", "flujo direccional normal"]
    out["flow_classification"] = np.select(conditions, choices, default="ruido normal")
    return out
