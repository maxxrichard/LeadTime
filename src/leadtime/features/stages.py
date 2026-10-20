"""Derive the per-stage durations (targets) from the order timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadtime.data.schema import CARRIER_ESTIMATE_COL, STAGES, target_col


def _days(a: pd.Series, b: pd.Series) -> pd.Series:
    return (b - a).dt.total_seconds() / 86_400.0


def add_stage_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add `<stage>_days` for every stage plus the carrier estimate horizon (known at purchase)."""
    df = df.copy()
    for s in STAGES:
        df[target_col(s.name)] = _days(df[s.start], df[s.end])
    df[CARRIER_ESTIMATE_COL] = _days(df["purchase_ts"], df["estimated_ts"])
    df["shipping_limit_days"] = _days(df["purchase_ts"], df["shipping_limit_ts"])
    return df


def filter_valid_orders(df: pd.DataFrame, max_total_days: float = 90.0) -> pd.DataFrame:
    """Keep delivered orders with a complete, monotone timestamp path.

    Olist contains a few hundred rows with out-of-order timestamps (e.g. delivered before hand-over)
    and a handful of >150-day deliveries; both are removed so the targets are well-defined.
    """
    needed = ["purchase_ts", "approved_ts", "carrier_ts", "delivered_ts", "estimated_ts"]
    m = (df["order_status"] == "delivered") & df[needed].notna().all(axis=1)
    out = add_stage_targets(df.loc[m])
    stage_cols = [target_col(s.name) for s in STAGES]
    ok = (out[stage_cols] >= 0).all(axis=1) & (out[target_col("total")] <= max_total_days)
    out = out.loc[ok].copy()
    # avoid exact zeros for log targets
    for c in stage_cols:
        out[c] = np.maximum(out[c].values, 1.0 / 1440.0)
    return out.sort_values("purchase_ts").reset_index(drop=True)
