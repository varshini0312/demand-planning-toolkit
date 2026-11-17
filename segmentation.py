"""ABC / XYZ demand segmentation.

ABC ranks what matters commercially (cumulative revenue contribution).
XYZ ranks how forecastable it is (coefficient of variation of demand).
The 3x3 grid tells a planner where to spend attention: AX gets a tight
statistical baseline and a high accuracy target, CZ gets a simple rule and
a loose one. Accuracy targets are meaningless without this split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAIN = ["item_id", "location_id"]

ABC_CUTS = {"A": 0.80, "B": 0.95}   # cumulative revenue share
XYZ_CUTS = {"X": 0.50, "Y": 1.00}   # coefficient of variation


def abc(df: pd.DataFrame, value_col: str = "revenue") -> pd.DataFrame:
    totals = df.groupby(GRAIN, as_index=False)[value_col].sum()
    totals = totals.sort_values(value_col, ascending=False)
    totals["cumulative_share"] = totals[value_col].cumsum() / totals[value_col].sum()
    totals["abc_class"] = np.select(
        [totals["cumulative_share"] <= ABC_CUTS["A"], totals["cumulative_share"] <= ABC_CUTS["B"]],
        ["A", "B"],
        default="C",
    )
    return totals[GRAIN + [value_col, "cumulative_share", "abc_class"]]


def xyz(df: pd.DataFrame, value_col: str = "quantity") -> pd.DataFrame:
    stats = df.groupby(GRAIN)[value_col].agg(["mean", "std"]).reset_index()
    stats["cv"] = np.where(stats["mean"] > 0, stats["std"] / stats["mean"], np.inf)
    stats["xyz_class"] = np.select(
        [stats["cv"] <= XYZ_CUTS["X"], stats["cv"] <= XYZ_CUTS["Y"]],
        ["X", "Y"],
        default="Z",
    )
    return stats[GRAIN + ["mean", "cv", "xyz_class"]]


def segment(df: pd.DataFrame, value_col: str = "quantity") -> pd.DataFrame:
    work = df.copy()
    work["revenue"] = work[value_col] * work["unit_price"]
    merged = abc(work).merge(xyz(work, value_col=value_col), on=GRAIN)
    merged["segment"] = merged["abc_class"] + merged["xyz_class"]
    return merged
