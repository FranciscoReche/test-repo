"""Market quality flags, soft penalties and 0-100 reliability score."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip01(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(0).clip(0, 1)


def add_quality_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    min_vol = float(cfg.get("min_total_volume", 1000))
    min_recent = float(cfg.get("min_recent_volume", 100))
    min_obs = float(cfg.get("min_timeseries_observations", 5))
    max_gap = float(cfg.get("max_gap_hours", 48))
    stale = float(cfg.get("stale_hours_before_event", 24))
    jump = float(cfg.get("large_jump_probability", 0.15))
    tiny_vol = float(cfg.get("tiny_jump_volume", 250))
    late_hours = float(cfg.get("late_creation_hours", 12))
    out["flag_low_total_volume"] = out["total_volume"].fillna(0) < min_vol
    out["flag_low_recent_activity"] = out["recent_volume"].fillna(0) < min_recent
    out["flag_frozen_market"] = out["prob_total_movement"].fillna(0) <= float(cfg.get("frozen_probability_epsilon", 0.01))
    out["flag_few_observations"] = out["n_probability_updates"].fillna(0) < min_obs
    out["flag_ambiguous_rules"] = out["resolution_rules"].fillna("").str.lower().str.contains("tbd|unclear|unknown|ambiguous", regex=True)
    out["flag_unclear_source"] = out["resolution_source"].fillna("").str.len() < 3
    out["flag_consensus_misaligned"] = (out["resolution_threshold"].notna() & out["earnings_consensus_eps"].notna() & ((out["resolution_threshold"] - out["earnings_consensus_eps"]).abs() > float(cfg.get("consensus_misalignment_tolerance", 0.05))))
    out["flag_bad_mapping"] = out["mapping_status"].fillna("") != "mapped"
    out["flag_duplicate_event"] = out["duplicate_event"].fillna(False).astype(bool)
    out["flag_created_too_late"] = ((out["earnings_event_ts"] - out["created_ts"]).dt.total_seconds() / 3600) < late_hours
    out["flag_no_pre_event_signal"] = out.get("pre_event_yes_probability", pd.Series(index=out.index, dtype="float64")).isna()
    out["flag_stale_before_event"] = out.get("cutoff_lag_hours", pd.Series(index=out.index, dtype="float64")).fillna(stale * 2) > stale
    out["flag_spurious_price_variation"] = (out["max_probability_jump"].fillna(0) > jump) & (out["total_volume"].fillna(0) < tiny_vol)
    out["flag_insufficient_depth"] = out["volume_per_hour_remaining"].fillna(0) <= 0
    out["flag_large_jump_tiny_market"] = out["flag_spurious_price_variation"]
    out["flag_unusable_design"] = ~out["market_type"].fillna("").str.lower().str.contains("eps|revenue|beat|miss|guidance")
    out["liquidity_score"] = _clip01(np.log1p(out["total_volume"]) / np.log1p(min_vol * 10))
    out["recent_activity_score"] = _clip01(np.log1p(out["recent_volume"]) / np.log1p(min_recent * 10))
    out["data_sufficiency_score"] = _clip01(out["n_probability_updates"] / (min_obs * 3))
    out["pricing_stability_score"] = _clip01(out["pricing_stability"])
    out["rule_quality_score"] = _clip01((out["rule_clarity_score"] + out["resolution_source_score"]) / 2)
    out["consensus_alignment_score"] = (~out["flag_consensus_misaligned"]).astype(float)
    out["freshness_score"] = _clip01(1 - out["freshness_hours"].fillna(stale * 2) / stale)
    out["alive_score"] = (~out["flag_frozen_market"]).astype(float) * _clip01(1 - out["time_gap_hours_max"].fillna(max_gap * 2) / max_gap)
    out["anomaly_cleanliness_score"] = (~out["flag_spurious_price_variation"]).astype(float)
    weights = cfg.get("reliability_weights", {})
    components = {
        "liquidity": "liquidity_score", "recent_activity": "recent_activity_score", "data_sufficiency": "data_sufficiency_score",
        "pricing_stability": "pricing_stability_score", "rule_clarity": "rule_quality_score", "consensus_alignment": "consensus_alignment_score",
        "freshness": "freshness_score", "alive": "alive_score", "anomaly_cleanliness": "anomaly_cleanliness_score",
    }
    score = sum(float(weights.get(k, 0)) * out[v] for k, v in components.items())
    hard_flags = [c for c in out.columns if c.startswith("flag_")]
    out["hard_exclusion_flags"] = out[hard_flags].apply(lambda r: ",".join([c for c, v in r.items() if bool(v)]), axis=1)
    penalty = out[hard_flags].sum(axis=1).clip(0, 8) * 3
    out["market_reliability_score"] = (100 * score - penalty).clip(0, 100).round(2)
    bins = [-1, 20, 40, 60, 80, 100]
    labels = ["inutilizable", "muy débil", "usable con mucha cautela", "señal razonable", "mercado de alta calidad"]
    out["market_classification"] = pd.cut(out["market_reliability_score"], bins=bins, labels=labels).astype(str)
    return out
