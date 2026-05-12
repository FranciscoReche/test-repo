"""Report generation for model, cohort and incremental-value analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def cohort_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = {
        "reliability_bucket": pd.cut(df["market_reliability_score"], [-1, 20, 40, 60, 80, 100]),
        "liquidity_bucket": pd.qcut(df["total_volume"].rank(method="first"), 4, duplicates="drop"),
        "earnings_time": df.get("earnings_earnings_time"),
        "sector": df.get("earnings_sector"),
    }
    for name, groups in buckets.items():
        if groups is None:
            continue
        tmp = df.assign(_group=groups).dropna(subset=["target_eps_beat"])
        for g, part in tmp.groupby("_group", observed=True):
            if len(part) == 0:
                continue
            pred = (part["adjusted_beat_probability"] >= 0.5).astype(int)
            rows.append({"cohort_type": name, "cohort": str(g), "n": len(part), "hit_rate": (pred == part["target_eps_beat"].astype(int)).mean(), "avg_reliability": part["market_reliability_score"].mean()})
    return pd.DataFrame(rows)


def filter_comparison(df: pd.DataFrame, thresholds: list[int | float]) -> pd.DataFrame:
    """Compare all markets vs reliability, late-move and divergence filters."""
    rows = []
    base = df.dropna(subset=["target_eps_beat"])
    scenarios = {"all_markets": base}
    for threshold in thresholds:
        scenarios[f"reliability_gt_{threshold}"] = base[base["market_reliability_score"] > threshold]
    scenarios["late_move_relevant"] = base[base["late_move_score"] >= 50] if "late_move_score" in base else base.iloc[0:0]
    scenarios["extreme_divergence"] = base[base["divergence_score"] >= 75] if "divergence_score" in base else base.iloc[0:0]
    for name, part in scenarios.items():
        if part.empty:
            rows.append({"scenario": name, "n": 0, "adjusted_hit_rate": pd.NA, "avg_reliability": pd.NA, "avg_total_volume": pd.NA})
            continue
        pred = (part["adjusted_beat_probability"] >= 0.5).astype(int)
        rows.append({
            "scenario": name,
            "n": len(part),
            "adjusted_hit_rate": (pred == part["target_eps_beat"].astype(int)).mean(),
            "avg_reliability": part["market_reliability_score"].mean(),
            "avg_total_volume": part["total_volume"].mean(),
        })
    return pd.DataFrame(rows)


def write_model_report(classification: pd.DataFrame, regression: pd.DataFrame, cohorts: pd.DataFrame, filters: pd.DataFrame, path: str | Path) -> None:
    lines = [
        "# Modeling Report",
        "",
        "## Incremental value experiments",
        "",
        "Compare feature sets: consensus_only, price_only, polymarket_raw, polymarket_liquidity_temporal, full.",
        "",
        "### Classification",
        "",
        classification.to_markdown(index=False) if not classification.empty else "No classification results.",
        "",
        "### Regression",
        "",
        regression.to_markdown(index=False) if not regression.empty else "No regression results.",
        "",
        "## Filter comparison",
        "",
        filters.to_markdown(index=False) if not filters.empty else "No filter comparison.",
        "",
        "## Cohort performance",
        "",
        cohorts.to_markdown(index=False) if not cohorts.empty else "No cohorts.",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
