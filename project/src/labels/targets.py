"""Target construction. All labels use reported outcomes after the prediction cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _binary_target(series: pd.Series) -> pd.Series:
    """Return nullable binary target without turning missing outcomes into misses."""
    return pd.Series(np.where(series.notna(), series > 0, pd.NA), index=series.index).astype("Int64")


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["eps_surprise_abs"] = out["earnings_reported_eps"] - out["earnings_consensus_eps"]
    out["eps_surprise_rel"] = out["eps_surprise_abs"] / out["earnings_consensus_eps"].abs().replace(0, np.nan)
    out["revenue_surprise_abs"] = out["earnings_reported_revenue"] - out["earnings_consensus_revenue"]
    out["revenue_surprise_rel"] = out["revenue_surprise_abs"] / out["earnings_consensus_revenue"].abs().replace(0, np.nan)
    out["target_eps_beat"] = _binary_target(out["eps_surprise_abs"])
    out["target_revenue_beat"] = _binary_target(out["revenue_surprise_abs"])
    out["target_positive_price_reaction"] = _binary_target(out["equity_post_return_1d"])
    out["target_eps_surprise_magnitude"] = out["eps_surprise_rel"].replace([np.inf, -np.inf], np.nan)
    out["target_price_reaction_magnitude"] = out["equity_post_return_3d"]
    return out
