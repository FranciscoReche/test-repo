"""End-to-end reproducible pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.anomaly_detection.anomalies import add_anomaly_scores
from src.backtest.strategies import run_backtests, write_backtest_report
from src.cleaning.sanity_checks import run_sanity_checks, write_quality_report
from src.config.settings import Settings, load_settings
from src.evaluation.reporting import cohort_performance, filter_comparison, write_model_report
from src.feature_engineering.features import build_features
from src.ingestion.loaders import load_earnings, load_equity_prices, load_polymarket_markets, load_polymarket_timeseries
from src.labels.targets import add_targets
from src.mapping.event_mapping import map_markets_to_events
from src.models.adjusted_signal import add_adjusted_signals
from src.models.train import evaluate_classification, evaluate_regression
from src.quality.reliability import add_quality_scores
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_master_dataset(settings: Settings) -> pd.DataFrame:
    logger.info("Loading raw data")
    markets = load_polymarket_markets(settings.path("polymarket_markets"))
    timeseries = load_polymarket_timeseries(settings.path("polymarket_timeseries"))
    earnings = load_earnings(settings.path("earnings"))
    equity = load_equity_prices(settings.path("equity_prices"))
    logger.info("Mapping markets to earnings events")
    mapped = map_markets_to_events(markets, earnings)
    sanity = run_sanity_checks(mapped, timeseries, settings.quality)
    write_quality_report(sanity, settings.path("quality_report"))
    if mapped.empty:
        raise ValueError("No mapped markets available for dataset construction")
    logger.info("Building features, labels, quality, anomaly and adjusted signals")
    df = build_features(mapped, timeseries, equity, settings.cutoffs.get("snapshot_windows", []))
    df = add_targets(df)
    df = add_quality_scores(df, settings.quality)
    df = add_anomaly_scores(df)
    df = add_adjusted_signals(df)
    settings.path("master_dataset").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.path("master_dataset"), index=False)
    return df


def run_modeling(settings: Settings, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = pd.read_csv(settings.path("master_dataset"), parse_dates=["earnings_event_ts"])
    logger.info("Running strict temporal validation")
    cls_eps = evaluate_classification(df, "target_eps_beat", settings.modeling)
    cls_price = evaluate_classification(df, "target_positive_price_reaction", settings.modeling)
    classification = pd.concat([cls_eps, cls_price], ignore_index=True)
    reg_eps = evaluate_regression(df, "target_eps_surprise_magnitude", settings.modeling)
    reg_price = evaluate_regression(df, "target_price_reaction_magnitude", settings.modeling)
    regression = pd.concat([reg_eps, reg_price], ignore_index=True)
    cohorts = cohort_performance(df)
    filters = filter_comparison(df, settings.modeling.get("reliability_thresholds", []))
    backtests = run_backtests(df, settings.backtest)
    write_model_report(classification, regression, cohorts, filters, settings.path("model_report"))
    write_backtest_report(backtests, settings.path("backtest_report"))
    return classification, regression, cohorts, backtests


def run_pipeline(config_path: str | Path = "project/src/config/default.yaml") -> None:
    settings = load_settings(config_path)
    df = build_master_dataset(settings)
    run_modeling(settings, df)


if __name__ == "__main__":
    run_pipeline()
