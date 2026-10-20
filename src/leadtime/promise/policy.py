"""Delivery-promise policies: turn quantile forecasts into a promised lead time, and score them.

A promise policy maps a scored order (its quantile forecasts) to `promised_days`. Policies are
compared on the *same* out-of-sample orders along two axes: the on-time rate (share of orders
delivered at or before the promise) and the mean promised lead time (shorter promises sell better).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from leadtime.data.schema import CARRIER_ESTIMATE_COL, PATH_STAGES
from leadtime.models.quantile import quantile_col


@dataclass(frozen=True)
class PromisePolicy:
    name: str = "base"
    buffer_days: float = 0.0
    round_up: bool = True

    def raw_days(self, preds: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def promised_days(self, preds: pd.DataFrame) -> np.ndarray:
        d = self.raw_days(preds) + self.buffer_days
        return np.ceil(d) if self.round_up else d

    def with_buffer(self, buffer_days: float) -> PromisePolicy:
        return replace(self, buffer_days=buffer_days)


def available_quantiles(preds: pd.DataFrame, stage: str = "total") -> list[float]:
    """Quantile levels present in a prediction frame for `stage` (from columns like `total_q80`)."""
    pre = f"{stage}_q"
    return sorted(
        int(c[len(pre) :]) / 100 for c in preds.columns if c.startswith(pre) and c[len(pre) :].isdigit()
    )


def interpolated_quantile(preds: pd.DataFrame, q: float, stage: str = "total") -> np.ndarray:
    """P_q per row; levels between the fitted grid are linearly interpolated, outside it are clipped."""
    grid = available_quantiles(preds, stage)
    if not grid:
        raise KeyError(f"no quantile columns for stage {stage!r}")
    q = float(np.clip(q, grid[0], grid[-1]))
    for g in grid:
        if abs(g - q) < 1e-9:
            return preds[quantile_col(stage, g)].to_numpy(dtype="float64")
    hi = next(g for g in grid if g > q)
    lo = grid[grid.index(hi) - 1]
    w = (q - lo) / (hi - lo)
    return (1 - w) * preds[quantile_col(stage, lo)].to_numpy(dtype="float64") + w * preds[
        quantile_col(stage, hi)
    ].to_numpy(dtype="float64")


@dataclass(frozen=True)
class QuantilePromise(PromisePolicy):
    """Promise the model's P_q of the end-to-end lead time (plus an optional buffer).

    `quantile` may be any level inside the fitted grid; off-grid levels are interpolated.
    """

    quantile: float = 0.8
    stage: str = "total"
    name: str = "model_quantile"

    def raw_days(self, preds: pd.DataFrame) -> np.ndarray:
        return interpolated_quantile(preds, self.quantile, self.stage)

    def with_quantile(self, quantile: float) -> QuantilePromise:
        return replace(self, quantile=quantile)


@dataclass(frozen=True)
class CarrierEstimatePromise(PromisePolicy):
    """Control policy: the platform/carrier estimate shown today (shifted by a constant buffer)."""

    name: str = "carrier_estimate"

    def raw_days(self, preds: pd.DataFrame) -> np.ndarray:
        return preds[CARRIER_ESTIMATE_COL].to_numpy(dtype="float64")


@dataclass(frozen=True)
class FixedLeadTimePromise(PromisePolicy):
    days: float = 10.0
    name: str = "fixed"

    def raw_days(self, preds: pd.DataFrame) -> np.ndarray:
        return np.full(len(preds), float(self.days))


@dataclass(frozen=True)
class ComposedQuantilePromise(PromisePolicy):
    """Compose the per-stage forecasts (approval + handling + transit) into a total-lead-time quantile.

    Quantiles do not add, so each stage's quantile grid is turned into a piecewise-linear inverse
    CDF, stage durations are sampled independently, summed, and the requested quantile of the sum
    is taken. This is the "every path of the network" view: the same stage models can be recombined
    for any route, e.g. when a stage is skipped (pickup) or a different carrier leg is used.
    """

    quantile: float = 0.8
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.8, 0.9)
    stages: tuple[str, ...] = tuple(PATH_STAGES)
    n_samples: int = 256
    seed: int = 0
    name: str = "composed_quantile"

    def raw_days(self, preds: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        qs = np.asarray(self.quantiles)
        # extend the grid to the open tails so sampling covers (0, 1)
        grid = np.concatenate([[0.0], qs, [1.0]])
        u = rng.random((self.n_samples, len(preds)))
        total = np.zeros((self.n_samples, len(preds)))
        for s in self.stages:
            qv = np.column_stack([preds[quantile_col(s, q)].to_numpy(dtype="float64") for q in qs])
            lo = np.maximum(qv[:, [0]] - (qv[:, [1]] - qv[:, [0]]), 0.0)
            hi = qv[:, [-1]] + (qv[:, [-1]] - qv[:, [-2]])
            vals = np.hstack([lo, qv, hi])  # (n, len(grid))
            for j in range(len(preds)):
                total[:, j] += np.interp(u[:, j], grid, vals[j])
        return np.quantile(total, self.quantile, axis=0)


def evaluate_promise(
    preds: pd.DataFrame, promised_days: np.ndarray, actual_col: str = "total_days"
) -> dict[str, float]:
    y = preds[actual_col].to_numpy(dtype="float64")
    late = y > promised_days
    return {
        "n": int(len(y)),
        "on_time_rate": float(1 - late.mean()),
        "late_rate": float(late.mean()),
        "mean_promised_days": float(np.mean(promised_days)),
        "median_promised_days": float(np.median(promised_days)),
        "mean_slack_days": float(np.mean(promised_days - y)),
        "promise_mae": float(np.mean(np.abs(promised_days - y))),
        "mean_days_late_when_late": float(np.mean((y - promised_days)[late])) if late.any() else 0.0,
    }


def calibrate_buffer(preds: pd.DataFrame, policy: PromisePolicy, target_on_time: float,
                     lo: float = -10.0, hi: float = 20.0, tol: float = 1e-3) -> PromisePolicy:  # fmt: skip
    """Bisection on the buffer so the policy hits `target_on_time` on a calibration set."""
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = evaluate_promise(preds, policy.with_buffer(mid).promised_days(preds))["on_time_rate"]
        if rate < target_on_time:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return policy.with_buffer(round(hi, 3))


def policy_frontier(
    preds: pd.DataFrame, policies: list[PromisePolicy], buffers: np.ndarray | None = None
) -> pd.DataFrame:
    """On-time rate vs mean promised lead time per policy: a constant-buffer sweep for every policy and,
    for quantile policies, a sweep over the quantile level (the efficient way to trade lead time for reliability)."""
    buffers = np.arange(-6, 8.5, 0.5) if buffers is None else buffers
    rows = []
    for p in policies:
        for b in buffers:
            m = evaluate_promise(preds, p.with_buffer(float(b)).promised_days(preds))
            rows.append(
                {
                    "policy": p.name,
                    "sweep": "buffer",
                    "buffer_days": float(b),
                    "quantile": getattr(p, "quantile", np.nan),
                    **m,
                }
            )
        if isinstance(p, QuantilePromise):
            grid = available_quantiles(preds, p.stage)
            for q in np.linspace(grid[0], grid[-1], 25):
                m = evaluate_promise(preds, p.with_quantile(float(q)).promised_days(preds))
                rows.append(
                    {
                        "policy": f"{p.name}_quantile_sweep",
                        "sweep": "quantile",
                        "buffer_days": p.buffer_days,
                        "quantile": float(q),
                        **m,
                    }
                )
    return pd.DataFrame(rows)
