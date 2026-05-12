"""Temporal modeling, baselines and incremental-value experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_SETS = {
    "consensus_only": ["earnings_consensus_eps", "earnings_consensus_revenue"],
    "price_only": ["recent_price_drift", "equity_return_1d", "equity_return_3d", "equity_return_5d", "equity_realized_volatility"],
    "polymarket_raw": ["raw_polymarket_probability", "log_odds", "distance_to_50"],
    "polymarket_liquidity_temporal": ["raw_polymarket_probability", "log_odds", "total_volume", "recent_volume", "late_move_intensity", "n_probability_updates", "prob_total_movement", "movement_persistence"],
    "full": ["raw_polymarket_probability", "log_odds", "total_volume", "recent_volume", "late_move_intensity", "n_probability_updates", "prob_total_movement", "movement_persistence", "market_reliability_score", "divergence_vs_consensus", "divergence_vs_price_drift", "anomaly_score", "recent_price_drift", "equity_realized_volatility", "earnings_market_cap", "earnings_sector", "earnings_earnings_time"],
}


def temporal_splits(df: pd.DataFrame, n_splits: int, min_train_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window temporal splits with no random shuffle."""
    n = len(df.sort_values("earnings_event_ts"))
    if n < max(6, min_train_size + n_splits):
        split = max(1, int(n * 0.7))
        return [(np.arange(split), np.arange(split, n))] if split < n else []
    test_size = max(1, (n - min_train_size) // n_splits)
    splits = []
    for end in range(min_train_size, n, test_size):
        test_end = min(n, end + test_size)
        if end < test_end:
            splits.append((np.arange(0, end), np.arange(end, test_end)))
    return splits


def _preprocessor(cols: list[str], df: pd.DataFrame) -> ColumnTransformer:
    cat = [c for c in cols if c in df.columns and (df[c].dtype == "object" or str(df[c].dtype) == "category")]
    num = [c for c in cols if c in df.columns and c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _classification_models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def _regression_models(seed: int) -> dict[str, object]:
    return {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def _safe_auc(y: list[int], p: list[float]) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def classification_metrics(y: list[int] | np.ndarray, pred: list[int] | np.ndarray, prob: list[float] | np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "n": 0}
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": _safe_auc(y, prob),
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, prob),
        "n": len(y),
    }


def evaluate_classification(df: pd.DataFrame, target: str, cfg: dict) -> pd.DataFrame:
    data = df.dropna(subset=[target, "earnings_event_ts"]).sort_values("earnings_event_ts").reset_index(drop=True)
    results = []
    splits = temporal_splits(data, int(cfg.get("n_splits", 4)), int(cfg.get("min_train_size", 20)))
    seed = int(cfg.get("random_seed", 42))
    for feature_set, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in data.columns]
        if not cols or not splits:
            continue
        for model_name, model in _classification_models(seed).items():
            preds: list[int] = []
            probs: list[float] = []
            actuals: list[int] = []
            for tr, te in splits:
                train, test = data.iloc[tr], data.iloc[te]
                if train[target].nunique() < 2:
                    continue
                pipe = Pipeline([("prep", _preprocessor(cols, data)), ("model", model)])
                pipe.fit(train[cols], train[target].astype(int))
                prob = pipe.predict_proba(test[cols])[:, 1]
                pred = (prob >= 0.5).astype(int)
                preds.extend(pred.tolist())
                probs.extend(prob.tolist())
                actuals.extend(test[target].astype(int).tolist())
            if actuals:
                results.append({"target": target, "feature_set": feature_set, "model": model_name, **classification_metrics(actuals, preds, probs)})
    results.extend(naive_classification_baselines(data, target, splits))
    return pd.DataFrame(results)


def naive_classification_baselines(data: pd.DataFrame, target: str, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Evaluate naive baselines using train-window base rates to avoid future leakage."""
    if not splits:
        return []
    rows = []
    collectors = {name: {"y": [], "pred": [], "prob": []} for name in ["always_beat", "always_follow_consensus", "coin_flip_calibrated", "follow_recent_price_drift"]}
    for tr, te in splits:
        train, test = data.iloc[tr], data.iloc[te]
        if train.empty or test.empty:
            continue
        train_rate = float(train[target].astype(int).mean())
        y = test[target].astype(int).to_numpy()
        baseline_preds = {
            "always_beat": np.ones(len(test), dtype=int),
            "always_follow_consensus": np.ones(len(test), dtype=int),
            "coin_flip_calibrated": np.full(len(test), int(train_rate >= 0.5)),
            "follow_recent_price_drift": (test["recent_price_drift"].fillna(0).to_numpy() > 0).astype(int) if "recent_price_drift" in test else np.full(len(test), int(train_rate >= 0.5)),
        }
        for name, pred in baseline_preds.items():
            collectors[name]["y"].extend(y.tolist())
            collectors[name]["pred"].extend(pred.tolist())
            collectors[name]["prob"].extend(np.full(len(test), train_rate).tolist())
    for name, values in collectors.items():
        if values["y"]:
            rows.append({"target": target, "feature_set": "naive", "model": name, **classification_metrics(values["y"], values["pred"], values["prob"])})
    return rows


def evaluate_regression(df: pd.DataFrame, target: str, cfg: dict) -> pd.DataFrame:
    data = df.dropna(subset=[target, "earnings_event_ts"]).sort_values("earnings_event_ts").reset_index(drop=True)
    results = []
    splits = temporal_splits(data, int(cfg.get("n_splits", 4)), int(cfg.get("min_train_size", 20)))
    seed = int(cfg.get("random_seed", 42))
    for feature_set, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in data.columns]
        if not cols or not splits:
            continue
        for model_name, model in _regression_models(seed).items():
            preds: list[float] = []
            actuals: list[float] = []
            for tr, te in splits:
                train, test = data.iloc[tr], data.iloc[te]
                pipe = Pipeline([("prep", _preprocessor(cols, data)), ("model", model)])
                pipe.fit(train[cols], train[target])
                preds.extend(pipe.predict(test[cols]).tolist())
                actuals.extend(test[target].tolist())
            if actuals:
                rmse = mean_squared_error(actuals, preds) ** 0.5
                results.append({"target": target, "feature_set": feature_set, "model": model_name, "mae": mean_absolute_error(actuals, preds), "rmse": rmse, "r2": r2_score(actuals, preds) if len(actuals) > 1 else np.nan, "n": len(actuals)})
    return pd.DataFrame(results)
