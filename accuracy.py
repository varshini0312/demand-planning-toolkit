"""Forecast accuracy and bias measurement.

The four metrics a planning organisation is actually governed on:

    MAPE   mean absolute percentage error  - readable, but blows up near zero
    WMAPE  weighted MAPE                   - volume-weighted, the reporting default
    Bias%  signed error                    - tells you *which way* you are wrong
    MAD    mean absolute deviation         - error in units, for stock conversations

MAPE and WMAPE tell you how big the error is. Bias tells you whether you are
persistently over- or under-forecasting, which is the one that quietly builds
excess stock or stockouts — a 20% MAPE with 0% bias is a very different
problem from 20% MAPE with -18% bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return np.nan
    return float(np.abs(actual - forecast).sum() / denominator * 100)


def bias_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return np.nan
    return float((forecast - actual).sum() / denominator * 100)


def mad(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


METRICS = {"mape": mape, "wmape": wmape, "bias_pct": bias_pct, "mad": mad}


def score(df: pd.DataFrame, by: list[str] | None = None,
          actual_col: str = "actual", forecast_col: str = "forecast") -> pd.DataFrame:
    """Compute all four metrics, optionally grouped (by item, by segment, by lag)."""
    def _score(group: pd.DataFrame) -> pd.Series:
        actual = group[actual_col].to_numpy(dtype=float)
        forecast = group[forecast_col].to_numpy(dtype=float)
        result = {name: fn(actual, forecast) for name, fn in METRICS.items()}
        result["volume"] = actual.sum()
        result["periods"] = len(group)
        return pd.Series(result)

    if not by:
        return _score(df).to_frame().T.round(2)
    return df.groupby(by).apply(_score, include_groups=False).reset_index().round(2)


def accuracy_summary(scored: pd.DataFrame, target_wmape: float = 30.0) -> pd.DataFrame:
    """Flag series breaching the accuracy target or showing persistent bias."""
    out = scored.copy()
    out["meets_target"] = out["wmape"] <= target_wmape
    out["bias_direction"] = np.select(
        [out["bias_pct"] > 5, out["bias_pct"] < -5],
        ["over-forecast", "under-forecast"],
        default="balanced",
    )
    return out
