"""Outlier detection and cleansing for demand history.

Planner-adjusted history is the input to every statistical baseline, so a
single unscrubbed spike (a data-entry error, a one-off bulk order) drags the
baseline up for months. This flags outliers per series and replaces them with
a robust estimate rather than deleting the row, so history stays complete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAIN = ["item_id", "location_id"]


def flag_outliers(df: pd.DataFrame, value_col: str = "quantity", threshold: float = 3.5) -> pd.DataFrame:
    """Flag outliers using the modified z-score (median absolute deviation).

    MAD is used instead of mean/standard deviation because the spikes we are
    hunting for distort the mean and inflate the standard deviation, which
    hides the very outliers we want to catch.
    """

    def _flag(group: pd.DataFrame) -> pd.Series:
        values = group[value_col].to_numpy(dtype=float)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:  # flat or highly intermittent series
            return pd.Series(False, index=group.index)
        modified_z = 0.6745 * (values - median) / mad
        return pd.Series(np.abs(modified_z) > threshold, index=group.index)

    out = df.copy()
    out["is_outlier"] = (
        out.groupby(GRAIN, group_keys=False)[[value_col]].apply(_flag).reindex(out.index).fillna(False)
    )
    return out


def cleanse(df: pd.DataFrame, value_col: str = "quantity", threshold: float = 3.5) -> pd.DataFrame:
    """Replace flagged outliers with the series median; keep the original value."""
    out = flag_outliers(df, value_col=value_col, threshold=threshold)
    medians = out.groupby(GRAIN)[value_col].transform("median")
    out[f"{value_col}_raw"] = out[value_col]
    out[f"{value_col}_clean"] = np.where(out["is_outlier"], medians, out[value_col])
    return out
