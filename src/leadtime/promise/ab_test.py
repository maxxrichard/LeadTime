"""A/B evaluation design for the promise policy.

Offline simulation on out-of-sample orders: each order is randomly assigned to control (the current
carrier estimate) or treatment (the model promise). Treatment is first *lead-time matched* - its
quantile level (and a residual buffer) is tuned so the mean promised lead time equals control's - so
the comparison isolates the accuracy of the promise rather than simply promising later. The primary metric is the late-delivery
rate; a two-proportion z-test and a bootstrap CI on the relative lift are reported, together with the
sample size an online test would need.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from leadtime.promise.policy import PromisePolicy, QuantilePromise, available_quantiles, evaluate_promise


@dataclass
class ABResult:
    control: str
    treatment: str
    n_control: int
    n_treatment: int
    late_rate_control: float
    late_rate_treatment: float
    abs_diff: float
    rel_lift: float
    rel_lift_ci_low: float
    rel_lift_ci_high: float
    z: float
    p_value: float
    mean_promised_control: float
    mean_promised_treatment: float
    treatment_buffer_days: float
    treatment_quantile: float
    required_n_per_arm: int

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.treatment} vs {self.control}: late rate {self.late_rate_treatment:.3%} vs {self.late_rate_control:.3%} "
            f"({self.rel_lift:+.1%}, 95% CI [{self.rel_lift_ci_low:+.1%}, {self.rel_lift_ci_high:+.1%}], p={self.p_value:.2e}) "
            f"at mean promised lead time {self.mean_promised_treatment:.2f} vs {self.mean_promised_control:.2f} days "
            f"(treatment: P{self.treatment_quantile * 100:.0f} {self.treatment_buffer_days:+.2f} d)"
        )


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def required_sample_size(p_control: float, rel_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size to detect a relative change `rel_effect` in a proportion (two-sided)."""
    p_t = p_control * (1 + rel_effect)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p_control + p_t) / 2
    num = (
        z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p_control * (1 - p_control) + p_t * (1 - p_t))
    ) ** 2
    return int(np.ceil(num / (p_t - p_control) ** 2))


def match_lead_time(
    preds: pd.DataFrame,
    treatment: PromisePolicy,
    target_mean_days: float,
    mode: str = "quantile",
    lo: float = -15.0,
    hi: float = 15.0,
) -> PromisePolicy:
    """Tune the treatment so its mean promised lead time equals `target_mean_days`.

    mode="quantile": for quantile policies, move the quantile level inside the fitted grid (the
    efficient trade-off); a residual constant buffer then closes any remaining gap (e.g. from whole-day
    rounding or when the target lies outside the grid). mode="buffer": constant shift only.
    """

    def mean_days(pol: PromisePolicy) -> float:
        return float(np.mean(pol.promised_days(preds)))

    if mode == "quantile" and isinstance(treatment, QuantilePromise):
        grid = available_quantiles(preds, treatment.stage)
        q_lo, q_hi = grid[0], grid[-1]
        for _ in range(50):
            mid = (q_lo + q_hi) / 2
            if mean_days(treatment.with_quantile(mid)) < target_mean_days:
                q_lo = mid
            else:
                q_hi = mid
            if q_hi - q_lo < 1e-4:
                break
        treatment = treatment.with_quantile(round((q_lo + q_hi) / 2, 4))
        lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_days(treatment.with_buffer(mid)) < target_mean_days:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return treatment.with_buffer(round((lo + hi) / 2, 3))


def simulate_ab(
    preds: pd.DataFrame,
    control: PromisePolicy,
    treatment: PromisePolicy,
    seed: int = 7,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    power: float = 0.8,
    match: bool = True,
    match_mode: str = "quantile",
    actual_col: str = "total_days",
) -> ABResult:
    rng = np.random.default_rng(seed)
    arm = rng.random(len(preds)) < 0.5
    c, t = preds.loc[~arm], preds.loc[arm]

    ctrl_eval = evaluate_promise(c, control.promised_days(c), actual_col)
    if match:
        treatment = match_lead_time(preds, treatment, ctrl_eval["mean_promised_days"], mode=match_mode)
    treat_eval = evaluate_promise(t, treatment.promised_days(t), actual_col)

    late_c = (c[actual_col].to_numpy() > control.promised_days(c)).astype(int)
    late_t = (t[actual_col].to_numpy() > treatment.promised_days(t)).astype(int)
    z, p = two_proportion_ztest(int(late_t.sum()), len(late_t), int(late_c.sum()), len(late_c))

    lifts = []
    for _ in range(n_bootstrap):
        bc = late_c[rng.integers(0, len(late_c), len(late_c))].mean()
        bt = late_t[rng.integers(0, len(late_t), len(late_t))].mean()
        lifts.append((bt - bc) / bc if bc > 0 else 0.0)
    lo, hi = np.quantile(lifts, [alpha / 2, 1 - alpha / 2])
    rel = (
        (treat_eval["late_rate"] - ctrl_eval["late_rate"]) / ctrl_eval["late_rate"]
        if ctrl_eval["late_rate"] > 0
        else 0.0
    )
    req = required_sample_size(ctrl_eval["late_rate"], rel, alpha, power) if abs(rel) > 1e-6 else 0

    return ABResult(
        control=control.name, treatment=treatment.name, n_control=len(c), n_treatment=len(t),
        late_rate_control=ctrl_eval["late_rate"], late_rate_treatment=treat_eval["late_rate"],
        abs_diff=treat_eval["late_rate"] - ctrl_eval["late_rate"], rel_lift=float(rel),
        rel_lift_ci_low=float(lo), rel_lift_ci_high=float(hi), z=z, p_value=p,
        mean_promised_control=ctrl_eval["mean_promised_days"], mean_promised_treatment=treat_eval["mean_promised_days"],
        treatment_buffer_days=treatment.buffer_days, treatment_quantile=float(getattr(treatment, "quantile", float("nan"))),
        required_n_per_arm=req,
    )  # fmt: skip
