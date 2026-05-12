from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_feature_engineering_uses_event_time_equity_join():
    source = (ROOT / "project/src/feature_engineering/features.py").read_text(encoding="utf-8")
    assert "_merge_equity_at_event" in source
    assert "candidates[\"date\"] <= event_ts" in source
    assert 'drop_duplicates("ticker", keep="last")' not in source


def test_targets_do_not_convert_missing_outcomes_to_misses():
    source = (ROOT / "project/src/labels/targets.py").read_text(encoding="utf-8")
    assert "_binary_target" in source
    assert "pd.NA" in source
