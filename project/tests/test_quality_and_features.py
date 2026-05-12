from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.pipeline import build_master_dataset


def test_master_dataset_contains_required_outputs():
    settings = load_settings("project/src/config/default.yaml")
    df = build_master_dataset(settings)
    required = {
        "raw_polymarket_probability",
        "market_reliability_score",
        "adjusted_beat_probability",
        "adjusted_price_reaction_signal",
        "divergence_score",
        "late_move_score",
        "anomaly_score",
        "market_classification",
        "flow_classification",
    }
    assert required.issubset(df.columns)
    assert df["market_reliability_score"].between(0, 100).all()


def test_low_quality_market_penalized():
    settings = load_settings("project/src/config/default.yaml")
    df = build_master_dataset(settings)
    omega = df.loc[df["market_id"] == "m5"].iloc[0]
    assert omega["market_reliability_score"] <= 20
    assert "flag_ambiguous_rules" in omega["hard_exclusion_flags"]
