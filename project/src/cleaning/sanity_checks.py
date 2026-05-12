"""Pre-modeling sanity checks. Modeling should not run before this report exists."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SanityResult:
    checks: pd.DataFrame
    passed: bool


def run_sanity_checks(mapped: pd.DataFrame, timeseries: pd.DataFrame, quality_cfg: dict) -> SanityResult:
    rows: list[dict] = []
    min_vol = float(quality_cfg.get("min_total_volume", 1000))
    jump = float(quality_cfg.get("large_jump_probability", 0.15))
    tiny_vol = float(quality_cfg.get("tiny_jump_volume", 250))
    max_gap = pd.Timedelta(hours=float(quality_cfg.get("max_gap_hours", 48)))
    stale = pd.Timedelta(hours=float(quality_cfg.get("stale_hours_before_event", 24)))
    for _, row in mapped.iterrows():
        market_id = row["market_id"]
        ts = timeseries[timeseries["market_id"] == market_id].sort_values("timestamp")
        event_ts = row.get("earnings_event_ts")
        flags: list[str] = []
        if row.get("mapping_status") != "mapped":
            flags.append("ticker_event_mapping_problem")
        if bool(row.get("duplicate_event", False)):
            flags.append("duplicate_ticker_event_market_type")
        if pd.notna(row.get("resolution_threshold")) and pd.notna(row.get("earnings_consensus_eps")):
            if abs(float(row["resolution_threshold"]) - float(row["earnings_consensus_eps"])) > float(quality_cfg.get("consensus_misalignment_tolerance", 0.05)):
                flags.append("resolution_threshold_misaligned_with_consensus")
        if pd.notna(row.get("total_volume")) and float(row["total_volume"]) < min_vol:
            flags.append("low_total_volume")
        if ts.empty:
            flags.append("no_probability_timeseries")
        else:
            deltas = ts["yes_probability"].diff().abs()
            if deltas.max(skipna=True) >= jump and float(row.get("total_volume") or 0) <= tiny_vol:
                flags.append("large_price_jump_tiny_market")
            if len(ts) > 1 and ts["timestamp"].diff().max() > max_gap:
                flags.append("excessive_temporal_gaps")
            if pd.notna(event_ts):
                recent = ts[ts["timestamp"] >= event_ts - stale]
                if recent.empty or recent["volume"].fillna(0).sum() <= 0:
                    flags.append("no_relevant_activity_near_event")
                if pd.notna(row.get("close_ts")) and abs(row["close_ts"] - event_ts) > pd.Timedelta(days=2):
                    flags.append("market_close_event_timing_misaligned")
        rules = " ".join(str(row.get(c, "")) for c in ["resolution_rules", "resolution_source"]).lower()
        if any(token in rules for token in ["tbd", "unclear", "unknown", "ambiguous"]):
            flags.append("ambiguous_or_unclear_resolution_rules")
        rows.append({"market_id": market_id, "event_id": row.get("event_id"), "flags": ",".join(flags), "n_flags": len(flags), "passes_sanity": len(flags) == 0})
    checks = pd.DataFrame(rows)
    return SanityResult(checks=checks, passed=checks["passes_sanity"].any() if not checks.empty else False)


def write_quality_report(result: SanityResult, path: str | Path) -> None:
    path = Path(path)
    counts = result.checks["flags"].str.get_dummies(sep=",").sum().sort_values(ascending=False) if not result.checks.empty else pd.Series(dtype=int)
    lines = ["# Data Quality Report", "", f"Passed any market: {result.passed}", "", "## Flag counts", ""]
    if counts.empty:
        lines.append("No flags generated.")
    else:
        lines.extend(f"- {flag}: {int(count)}" for flag, count in counts.items() if flag)
    lines += ["", "## Market-level checks", "", result.checks.to_markdown(index=False) if not result.checks.empty else "No markets."]
    path.write_text("\n".join(lines), encoding="utf-8")
