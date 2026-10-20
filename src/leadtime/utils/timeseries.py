"""Vectorised as-of (point-in-time) helpers used by both the feature builder and the synthetic generator.

Every function here only looks at events that *completed strictly before* the query timestamp,
which is what makes the derived features leakage-safe for a model that scores an order at
purchase time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NS_PER_DAY = 86_400 * 1_000_000_000


def to_int_ns(ts: pd.Series | np.ndarray) -> np.ndarray:
    """Datetime64 -> int64 nanoseconds (NaT becomes the int64 minimum, callers must mask it)."""
    return pd.to_datetime(pd.Series(np.asarray(ts))).values.astype("datetime64[ns]").astype("int64")


def asof_window_stats(
    event_ts: np.ndarray,
    values: np.ndarray,
    query_ts: np.ndarray,
    window_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count and mean of `values` whose `event_ts` lies in [query - window, query).

    Parameters are int64 nanosecond arrays. Rows with NaN values are ignored.
    Returns (n, mean) aligned with `query_ts`; mean is NaN where n == 0.
    """
    mask = ~np.isnan(values)
    event_ts = event_ts[mask]
    values = values[mask]
    order = np.argsort(event_ts, kind="stable")
    e = event_ts[order]
    v = values[order]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    hi = np.searchsorted(e, query_ts, side="left")
    if window_days is None:
        lo = np.zeros_like(hi)
    else:
        lo = np.searchsorted(e, query_ts - int(window_days * NS_PER_DAY), side="left")
    n = hi - lo
    s = cs[hi] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return n.astype(np.int64), mean


def rolling_count(ts: np.ndarray, query_ts: np.ndarray, window_days: float) -> np.ndarray:
    """Number of `ts` events in [query - window, query)."""
    s = np.sort(ts)
    hi = np.searchsorted(s, query_ts, side="left")
    lo = np.searchsorted(s, query_ts - int(window_days * NS_PER_DAY), side="left")
    return (hi - lo).astype(np.int64)


def count_before(ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Number of `ts` events strictly before each query timestamp."""
    s = np.sort(ts[~np.isnan(ts.astype("float64"))]) if ts.dtype.kind == "f" else np.sort(ts)
    return np.searchsorted(s, query_ts, side="left").astype(np.int64)
