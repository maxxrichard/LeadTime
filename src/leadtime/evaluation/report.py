"""Markdown + PNG reporting for a backtest / promise / A-B run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from leadtime.evaluation.backtest import BacktestResult
from leadtime.evaluation.metrics import calibration_table
from leadtime.models.quantile import quantile_col
from leadtime.promise.ab_test import ABResult


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _try_plots(
    out_dir: Path,
    bt: BacktestResult,
    quantiles: list[float],
    frontier: pd.DataFrame | None,
    promise_quantile: float,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return []
    files = []
    fm = bt.fold_metrics
    tot = fm[fm.stage == "total"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m, d in tot.groupby("model"):
        if m == "sum_of_stage_medians":
            continue
        ax.plot(d["fold"], d["mae"], marker="o", label=m)
    ax.set_xlabel("backtest fold (rolling origin)")
    ax.set_ylabel("MAE of delivery date [days]")
    ax.set_title("End-to-end lead time: MAE per fold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "mae_by_fold.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)

    oos = bt.oos_predictions
    fig, ax = plt.subplots(figsize=(4.2, 4))
    cal = calibration_table(
        oos["total_days"].to_numpy(), oos[[quantile_col("total", q) for q in quantiles]].to_numpy(), quantiles
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(cal["quantile"], cal["empirical_coverage"], marker="o", label="model (total)")
    rb = calibration_table(
        oos["total_days"].to_numpy(),
        oos[[f"route_{quantile_col('total', q)}" for q in quantiles]].to_numpy(),
        quantiles,
    )
    ax.plot(rb["quantile"], rb["empirical_coverage"], marker="s", label="route baseline")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration (out-of-sample)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "calibration.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)

    if frontier is not None and not frontier.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, d in frontier.groupby("policy"):
            ax.plot(d["mean_promised_days"], d["on_time_rate"], marker=".", label=name)
        ax.set_xlabel("mean promised lead time [days]")
        ax.set_ylabel("on-time rate")
        ax.set_title("Promise frontier: on-time rate vs promised lead time")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / "promise_frontier.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p.name)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp = bt.feature_importance.head(20)[::-1]
    ax.barh(imp.index, imp.values)
    ax.set_title("Feature importance (total lead time, median model)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    p = out_dir / "feature_importance.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)
    return files


def write_report(
    out_dir: str | Path,
    bt: BacktestResult,
    quantiles: list[float],
    promise_eval: pd.DataFrame | None = None,
    ab: ABResult | None = None,
    frontier: pd.DataFrame | None = None,
    promise_quantile: float = 0.8,
    meta: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    bt.oos_predictions.to_parquet(out_dir / "oos_predictions.parquet", index=False)
    bt.feature_importance.to_csv(out_dir / "feature_importance.csv", header=["importance"])
    summary = bt.summary().reset_index()
    pngs = _try_plots(out_dir, bt, quantiles, frontier, promise_quantile)

    tot = summary[summary.stage == "total"].set_index("model")
    model_name = next((m for m in tot.index if m.endswith("_segmented")), None)
    lines = ["# LeadTime backtest report", ""]
    if meta:
        lines += ["```json", json.dumps(meta, indent=2, default=str), "```", ""]
    if model_name and "carrier_estimate" in tot.index:
        m_mae, c_mae = tot.loc[model_name, "mae"], tot.loc["carrier_estimate", "mae"]
        lines += [
            f"**Delivery-date MAE (median forecast): {m_mae:.2f} days vs carrier estimate {c_mae:.2f} days "
            f"({(m_mae / c_mae - 1):+.0%}).**",
            "",
        ]
    cols = ["stage", "model", "mae", "bias", "pinball_mean"] + [
        f"coverage_q{int(q * 100):02d}" for q in quantiles
    ]
    lines += [
        "## Rolling-origin backtest (mean over folds)",
        "",
        _md_table(summary[[c for c in cols if c in summary.columns]]),
        "",
    ]
    oos = bt.oos_predictions
    cal = calibration_table(
        oos["total_days"].to_numpy(), oos[[quantile_col("total", q) for q in quantiles]].to_numpy(), quantiles
    )
    lines += ["## Calibration of the end-to-end forecast (pooled out-of-sample)", "", _md_table(cal), ""]
    if promise_eval is not None:
        lines += ["## Promise policies (same out-of-sample orders)", "", _md_table(promise_eval), ""]
    if ab is not None:
        lines += ["## A/B simulation (lead-time matched)", "", f"{ab}", "",
                  f"Required sample size per arm for this effect at alpha=0.05, power=0.8: **{ab.required_n_per_arm:,}** orders.", ""]  # fmt: skip
    lines += [
        "## Feature importance (top 15)",
        "",
        _md_table(
            bt.feature_importance.head(15)
            .reset_index()
            .rename(columns={"index": "feature", 0: "importance"}),
            "{:.1f}",
        ),
        "",
    ]
    if pngs:
        lines += ["## Figures", ""] + [f"![{p}]({p})" for p in pngs] + [""]
    path = out_dir / "backtest_report.md"
    path.write_text("\n".join(lines))
    return path
