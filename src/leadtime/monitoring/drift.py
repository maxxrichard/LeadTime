"""Monitoring: feature drift (PSI) and forecast-quality drift over time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadtime.evaluation.metrics import coverage, mae


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index on quantile bins of the reference distribution (NaNs dropped)."""
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.maximum(r, eps), np.maximum(c, eps)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference: pd.Series, current: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    r = reference.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    c = current.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    r, c = np.maximum(r, eps), np.maximum(c, eps)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str],
                 categorical: list[str] | None = None, warn: float = 0.1, alert: float = 0.25) -> pd.DataFrame:  # fmt: skip
    categorical = set(categorical or [])
    rows = []
    for c in columns:
        if c in categorical:
            v = categorical_psi(reference[c], current[c])
        else:
            v = psi(reference[c].to_numpy(dtype="float64"), current[c].to_numpy(dtype="float64"))
        status = "alert" if v >= alert else "warn" if v >= warn else "ok"
        rows.append({"feature": c, "psi": v, "status": status,
                     "ref_mean": reference[c].mean() if c not in categorical else np.nan,
                     "cur_mean": current[c].mean() if c not in categorical else np.nan})  # fmt: skip
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def performance_drift(preds: pd.DataFrame, median_col: str, upper_col: str, nominal: float,
                      actual_col: str = "total_days", ts_col: str = "purchase_ts", freq: str = "W",
                      tolerance: float = 0.05) -> pd.DataFrame:  # fmt: skip
    """Weekly MAE and upper-quantile coverage; flags weeks whose coverage drifts from the nominal level."""
    g = preds.groupby(pd.Grouper(key=ts_col, freq=freq))
    rows = []
    for period, d in g:
        if len(d) == 0:
            continue
        y = d[actual_col].to_numpy()
        cov = coverage(y, d[upper_col].to_numpy())
        rows.append({"period": period, "n": len(d), "mae": mae(y, d[median_col].to_numpy()),
                     "coverage": cov, "coverage_gap": cov - nominal,
                     "status": "alert" if abs(cov - nominal) > tolerance else "ok"})  # fmt: skip
    return pd.DataFrame(rows)
