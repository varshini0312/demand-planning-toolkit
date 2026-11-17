"""Generate synthetic monthly sales history for the demo.

Builds a realistic-looking history at Item x Location x Month grain with
trend, annual seasonality, promotional uplifts, intermittent demand and a
handful of outliers, so the rest of the pipeline has something to work on.
All data is synthetic; no client data is used anywhere in this repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ITEMS = 60
N_LOCATIONS = 8
START = "2023-01"
PERIODS = 36  # three years of monthly history


def _series(base: float, trend: float, seasonality: float, noise: float, n: int) -> np.ndarray:
    t = np.arange(n)
    level = base + trend * t
    season = 1 + seasonality * np.sin(2 * np.pi * (t % 12) / 12)
    values = level * season * RNG.normal(1.0, noise, n)
    return np.clip(values, 0, None)


def build() -> pd.DataFrame:
    months = pd.period_range(START, periods=PERIODS, freq="M").astype(str)
    rows = []

    for i in range(1, N_ITEMS + 1):
        item = f"ITM{i:04d}"
        # a mix of profiles: fast movers, slow movers, seasonal, intermittent
        profile = RNG.choice(["fast", "medium", "slow", "intermittent"], p=[0.2, 0.4, 0.3, 0.1])
        base = {"fast": 900, "medium": 300, "slow": 80, "intermittent": 40}[profile]
        noise = {"fast": 0.10, "medium": 0.18, "slow": 0.30, "intermittent": 0.60}[profile]
        seasonality = RNG.uniform(0.05, 0.35)
        trend = RNG.normal(base * 0.004, base * 0.004)

        for j in RNG.choice(np.arange(1, N_LOCATIONS + 1), size=RNG.integers(3, N_LOCATIONS + 1), replace=False):
            location = f"LOC{j:03d}"
            values = _series(base * RNG.uniform(0.5, 1.5), trend, seasonality, noise, PERIODS)

            if profile == "intermittent":  # zero out random months
                values[RNG.random(PERIODS) < 0.35] = 0

            promo = RNG.random(PERIODS) < 0.08  # promotional uplift
            values[promo] *= RNG.uniform(1.4, 2.2, promo.sum())

            spikes = RNG.random(PERIODS) < 0.02  # data-entry outliers
            values[spikes] *= RNG.uniform(4, 8, spikes.sum())

            for month, qty, is_promo in zip(months, values, promo):
                rows.append(
                    {
                        "item_id": item,
                        "location_id": location,
                        "month": month,
                        "quantity": round(float(qty), 1),
                        "unit_price": round(float(RNG.uniform(50, 800)), 2),
                        "promotion_flag": int(is_promo),
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv("data/sales_history.csv", index=False)
    print(
        f"wrote data/sales_history.csv: {len(df):,} rows, "
        f"{df['item_id'].nunique()} items, {df['location_id'].nunique()} locations, "
        f"{df['month'].nunique()} months"
    )
