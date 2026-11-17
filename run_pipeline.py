"""End-to-end demand planning pipeline.

    sales history -> outlier cleansing -> ABC/XYZ segmentation
                  -> best-fit statistical baseline -> accuracy and bias scoring

Backtests on a holdout so the accuracy numbers are measured, not asserted,
then writes three reports: forecasts, accuracy by segment, and exceptions
for planner review.

    python run_pipeline.py
    python run_pipeline.py --holdout 6 --horizon 6 --target-wmape 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import accuracy
import cleansing
import forecasting
import segmentation

GRAIN = ["item_id", "location_id"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"loaded {len(df):,} rows | {df['item_id'].nunique()} items | "
          f"{df['location_id'].nunique()} locations | {df['month'].nunique()} months")
    return df


def backtest(df: pd.DataFrame, holdout: int, target_wmape: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the final months, forecast them, and score against actuals."""
    months = sorted(df["month"].unique())
    train_months, test_months = months[:-holdout], months[-holdout:]

    train = df[df["month"].isin(train_months)]
    test = df[df["month"].isin(test_months)]

    forecasts = forecasting.generate_forecasts(train, horizon=holdout, holdout=holdout)
    actuals = test.rename(columns={"quantity_clean": "actual"})[GRAIN + ["month", "actual"]]

    comparison = forecasts.merge(actuals, on=GRAIN + ["month"], how="inner")
    print(f"backtested {len(comparison):,} item-location-month combinations "
          f"over {holdout} months")
    return comparison, forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sales_history.csv"))
    parser.add_argument("--holdout", type=int, default=6, help="months held out for accuracy measurement")
    parser.add_argument("--horizon", type=int, default=6, help="months to forecast forward")
    parser.add_argument("--target-wmape", type=float, default=30.0)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)

    print("\n1. load")
    history = load(args.input)

    print("\n2. outlier cleansing")
    cleaned = cleansing.cleanse(history)
    flagged = int(cleaned["is_outlier"].sum())
    print(f"flagged and replaced {flagged:,} outliers "
          f"({flagged / len(cleaned):.1%} of rows) using modified z-score")

    print("\n3. ABC / XYZ segmentation")
    segments = segmentation.segment(cleaned, value_col="quantity_clean")
    print(segments["segment"].value_counts().sort_index().to_string())

    print("\n4. best-fit baseline and backtest")
    comparison, _ = backtest(cleaned, args.holdout, args.target_wmape)
    comparison = comparison.merge(segments[GRAIN + ["abc_class", "xyz_class", "segment"]], on=GRAIN)

    print("\n5. accuracy and bias")
    overall = accuracy.score(comparison)
    by_segment = accuracy.accuracy_summary(
        accuracy.score(comparison, by=["segment"]), target_wmape=args.target_wmape
    )
    by_series = accuracy.accuracy_summary(
        accuracy.score(comparison, by=GRAIN + ["segment"]), target_wmape=args.target_wmape
    )

    print("\noverall")
    print(overall.to_string(index=False))
    print("\nby segment")
    print(by_segment.sort_values("segment").to_string(index=False))

    model_mix = comparison.groupby("model").size().sort_values(ascending=False)
    print("\nbest-fit model selected per series")
    print(model_mix.to_string())

    exceptions = by_series[(~by_series["meets_target"]) & (by_series["bias_direction"] != "balanced")]
    exceptions = exceptions.sort_values("volume", ascending=False)

    print("\n6. forward forecast")
    forward = forecasting.generate_forecasts(cleaned, horizon=args.horizon, holdout=args.holdout)

    forward.to_csv(args.outdir / "forecast.csv", index=False)
    by_segment.to_csv(args.outdir / "accuracy_by_segment.csv", index=False)
    exceptions.to_csv(args.outdir / "exceptions_for_review.csv", index=False)

    hit_rate = by_series["meets_target"].mean()
    print(f"\n{len(forward):,} forecast rows for the next {args.horizon} months")
    print(f"{hit_rate:.0%} of series meet the {args.target_wmape:.0f}% WMAPE target")
    print(f"{len(exceptions):,} series flagged for planner review "
          f"(missed target and persistently biased)")
    print(f"\nreports written to {args.outdir}/")


if __name__ == "__main__":
    main()
