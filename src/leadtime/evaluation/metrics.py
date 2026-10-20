"""Point, quantile and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def bias(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean (pred - actual); positive means the forecast is too long."""
    return float(np.mean(np.asarray(pred) - np.asarray(y)))


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of actuals at or below the predicted quantile (should equal the nominal level)."""
    return float(np.mean(np.asarray(y) <= np.asarray(pred)))


def summarize_quantiles(y: np.ndarray, preds: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    """Metrics for an (n, n_q) quantile matrix: MAE/bias of the median, pinball and coverage per q."""
    out: dict[str, float] = {}
    qi = {q: i for i, q in enumerate(quantiles)}
    mid = qi.get(0.5, len(quantiles) // 2)
    out["mae"] = mae(y, preds[:, mid])
    out["bias"] = bias(y, preds[:, mid])
    pl = []
    for q, i in qi.items():
        out[f"pinball_q{int(q * 100):02d}"] = pinball_loss(y, preds[:, i], q)
        out[f"coverage_q{int(q * 100):02d}"] = coverage(y, preds[:, i])
        pl.append(out[f"pinball_q{int(q * 100):02d}"])
    out["pinball_mean"] = float(np.mean(pl))
    if 0.1 in qi and 0.9 in qi:
        out["width_q10_q90"] = float(np.mean(preds[:, qi[0.9]] - preds[:, qi[0.1]]))
    return out


def calibration_table(y: np.ndarray, preds: np.ndarray, quantiles: list[float]) -> pd.DataFrame:
    rows = [{"quantile": q, "empirical_coverage": coverage(y, preds[:, i]), "gap": coverage(y, preds[:, i]) - q}
            for i, q in enumerate(quantiles)]  # fmt: skip
    return pd.DataFrame(rows)
