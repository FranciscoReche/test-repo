"""Simple event-driven backtests for signal usefulness, not over-optimized trading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"n": 0, "hit_rate": np.nan, "average_return": np.nan, "median_return": np.nan, "sharpe_simple": np.nan, "max_drawdown": np.nan, "win_loss_ratio": np.nan}
    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return {"n": len(returns), "hit_rate": float((returns > 0).mean()), "average_return": returns.mean(), "median_return": returns.median(), "sharpe_simple": returns.mean() / returns.std(ddof=0) if returns.std(ddof=0) > 0 else np.nan, "max_drawdown": dd.min(), "win_loss_ratio": wins.mean() / abs(losses.mean()) if not losses.empty else np.nan}


def run_backtests(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cost = float(cfg.get("transaction_cost_bps", 5)) / 10000
    min_sig = float(cfg.get("min_abs_signal", 0.05))
    high_rel = float(cfg.get("high_reliability_threshold", 60))
    anom = float(cfg.get("anomaly_threshold", 70))
    ret = df["equity_post_return_1d"].fillna(0)
    strategies = {
        "follow_raw_polymarket": np.sign(df["raw_polymarket_probability"].fillna(0.5) - 0.5),
        "follow_adjusted_signal": np.where(df["adjusted_price_reaction_signal"].abs() >= min_sig, np.sign(df["adjusted_price_reaction_signal"]), 0),
        "high_reliability_only": np.where(df["market_reliability_score"] >= high_rel, np.sign(df["adjusted_price_reaction_signal"]), 0),
        "anomalous_late_moves_only": np.where(df["anomaly_score"] >= anom, np.sign(df["adjusted_price_reaction_signal"]), 0),
        "divergence_vs_consensus": np.where(df["divergence_vs_consensus"].abs() >= min_sig, np.sign(df["divergence_vs_consensus"]), 0),
    }
    rows = []
    for name, side in strategies.items():
        pnl = pd.Series(side, index=df.index).replace(0, np.nan) * ret - cost
        rows.append({"strategy": name, **_metrics(pnl)})
    return pd.DataFrame(rows)


def write_backtest_report(results: pd.DataFrame, path: str | Path) -> None:
    Path(path).write_text("# Backtest Report\n\n" + (results.to_markdown(index=False) if not results.empty else "No backtest results."), encoding="utf-8")
