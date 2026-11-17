"""Statistical baseline forecasting models and best-fit selection.

Deliberately implemented in numpy/pandas rather than a forecasting library:
these are the baselines a planner overrides, so being able to explain exactly
what the number is — and why it moved — matters more than model sophistication.

Every model takes a history array and a horizon, and returns the forecast.
Best-fit selection backtests each model on a holdout and picks the winner per
series on weighted MAPE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_LENGTH = 12


def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Last observed value, carried forward. The benchmark every model must beat."""
    return np.repeat(history[-1], horizon)


def moving_average(history: np.ndarray, horizon: int, window: int = 3) -> np.ndarray:
    """Mean of the last `window` periods — the standard planner baseline."""
    window = min(window, len(history))
    return np.repeat(history[-window:].mean(), horizon)


def seasonal_naive(history: np.ndarray, horizon: int, season: int = SEASON_LENGTH) -> np.ndarray:
    """Same month last year. Strong where seasonality dominates."""
    if len(history) < season:
        return moving_average(history, horizon)
    last_season = history[-season:]
    return np.array([last_season[i % season] for i in range(horizon)])


def simple_exponential_smoothing(history: np.ndarray, horizon: int, alpha: float = 0.3) -> np.ndarray:
    """Exponentially weighted level. Reacts faster than a moving average."""
    level = history[0]
    for value in history[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.repeat(level, horizon)


def holt_linear(history: np.ndarray, horizon: int, alpha: float = 0.3, beta: float = 0.1) -> np.ndarray:
    """Level plus trend (Holt). Used where a series is clearly growing or declining."""
    if len(history) < 2:
        return naive(history, horizon)
    level, trend = history[0], history[1] - history[0]
    for value in history[1:]:
        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
    return np.clip(np.array([level + (h + 1) * trend for h in range(horizon)]), 0, None)


MODELS = {
    "naive": naive,
    "moving_average_3": lambda h, n: moving_average(h, n, window=3),
    "moving_average_6": lambda h, n: moving_average(h, n, window=6),
    "seasonal_naive": seasonal_naive,
    "ses": simple_exponential_smoothing,
    "holt": holt_linear,
}


def _wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return np.inf
    return np.abs(actual - forecast).sum() / denominator


def best_fit(history: np.ndarray, holdout: int = 6) -> tuple[str, float]:
    """Backtest every model on the last `holdout` periods; return the winner.

    Selection is on weighted MAPE rather than plain MAPE so that high-volume
    periods carry proportionate weight and near-zero months cannot dominate.
    """
    if len(history) <= holdout + 2:
        return "moving_average_3", np.inf

    train, test = history[:-holdout], history[-holdout:]
    scores = {name: _wmape(test, model(train, holdout)) for name, model in MODELS.items()}
    winner = min(scores, key=scores.get)
    return winner, scores[winner]


def generate_forecasts(df: pd.DataFrame, value_col: str = "quantity_clean", horizon: int = 6,
                       holdout: int = 6) -> pd.DataFrame:
    """Fit the best model per Item x Location series and forecast `horizon` months."""
    records = []
    months = sorted(df["month"].unique())
    future = pd.period_range(pd.Period(months[-1], freq="M") + 1, periods=horizon, freq="M").astype(str)

    for (item, location), group in df.groupby(["item_id", "location_id"]):
        series = group.sort_values("month")[value_col].to_numpy(dtype=float)
        model_name, backtest_wmape = best_fit(series, holdout=holdout)
        forecast = MODELS[model_name](series, horizon)

        for month, value in zip(future, forecast):
            records.append(
                {
                    "item_id": item,
                    "location_id": location,
                    "month": month,
                    "forecast": round(float(max(value, 0)), 1),
                    "model": model_name,
                    "backtest_wmape": round(float(backtest_wmape), 4),
                }
            )

    return pd.DataFrame(records)
