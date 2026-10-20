"""Rolling-origin backtest.

For each cutoff T_k the model is trained on orders *delivered* before T_k (so every training label
was observable at T_k) and *purchased* at least `label_gap_days` before T_k (orders purchased later are
right-censored: only the fast ones have a label yet, which would bias the model low) and scored on orders *purchased* in (T_k, T_k + horizon]. The out-of-sample
predictions of all folds are concatenated and reused by the promise-policy and A/B analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leadtime.config import BacktestConfig, ModelConfig
from leadtime.data.schema import CARRIER_ESTIMATE_COL, PATH_STAGES, SEGMENT_COL, STAGE_NAMES, target_col
from leadtime.evaluation.metrics import mae, summarize_quantiles
from leadtime.models.baselines import RouteQuantileBaseline
from leadtime.models.quantile import SegmentedQuantileModel, quantile_col


@dataclass
class Fold:
    fold: int
    cutoff: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class BacktestResult:
    fold_metrics: pd.DataFrame  # one row per (fold, stage, model)
    oos_predictions: pd.DataFrame  # one row per test order with quantile columns
    feature_importance: pd.Series

    def summary(self) -> pd.DataFrame:
        """Average metrics over folds per (stage, model)."""
        num = self.fold_metrics.select_dtypes("number").columns.difference(["fold"])
        return self.fold_metrics.groupby(["stage", "model"])[list(num)].mean().round(4)


def make_folds(df: pd.DataFrame, cfg: BacktestConfig) -> list[Fold]:
    purchase = df["purchase_ts"]
    delivered = df["delivered_ts"]
    end = purchase.max().normalize()
    start = purchase.min().normalize()
    folds: list[Fold] = []
    for k in range(cfg.n_folds, 0, -1):
        cutoff = end - pd.Timedelta(days=cfg.horizon_days + cfg.step_days * (k - 1))
        if (cutoff - start).days < cfg.min_train_days:
            continue
        mature = purchase <= cutoff - pd.Timedelta(days=cfg.label_gap_days)
        train_idx = np.where(((delivered <= cutoff) & mature).values)[0]
        test_idx = np.where(
            ((purchase > cutoff) & (purchase <= cutoff + pd.Timedelta(days=cfg.horizon_days))).values
        )[0]
        if len(test_idx) == 0:
            continue
        folds.append(Fold(fold=len(folds) + 1, cutoff=cutoff, train_idx=train_idx, test_idx=test_idx))
    if not folds:
        raise ValueError("No usable folds: increase the data span or lower min_train_days")
    return folds


def run_backtest(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    bt_cfg: BacktestConfig,
    feature_cols: list[str],
    categorical_cols: list[str],
    stages: list[str] | None = None,
    verbose: bool = True,
) -> BacktestResult:
    stages = stages or STAGE_NAMES
    quantiles = list(model_cfg.quantiles)
    folds = make_folds(df, bt_cfg)
    rows: list[dict] = []
    oos: list[pd.DataFrame] = []
    importance = None
    for f in folds:
        train, test = df.iloc[f.train_idx], df.iloc[f.test_idx]
        if verbose:
            print(
                f"[fold {f.fold}] cutoff={f.cutoff.date()} train={len(train):,} test={len(test):,}",
                flush=True,
            )
        model = SegmentedQuantileModel(model_cfg, feature_cols, categorical_cols, stages=stages).fit(train)
        pred = model.predict(test)
        base = {s: RouteQuantileBaseline(quantiles, target=target_col(s)).fit(train) for s in stages}

        keep = ["order_id", "purchase_ts", "delivered_ts", SEGMENT_COL, "seller_state", "customer_state",
                CARRIER_ESTIMATE_COL] + [target_col(s) for s in stages]  # fmt: skip
        out = test[keep].copy()
        out["fold"] = f.fold
        out["cutoff"] = f.cutoff
        out = out.join(pred)
        for s in stages:
            bp = base[s].predict(test)
            for i, q in enumerate(quantiles):
                out[f"route_{quantile_col(s, q)}"] = bp[:, i]

        for s in stages:
            y = test[target_col(s)].to_numpy()
            qm = pred[[quantile_col(s, q) for q in quantiles]].to_numpy()
            rows.append({"fold": f.fold, "cutoff": f.cutoff, "stage": s, "model": f"{model_cfg.backend}_segmented", "n_test": len(test),
                         **summarize_quantiles(y, qm, quantiles)})  # fmt: skip
            rows.append({"fold": f.fold, "cutoff": f.cutoff, "stage": s, "model": "route_quantile_baseline", "n_test": len(test),
                         **summarize_quantiles(y, base[s].predict(test), quantiles)})  # fmt: skip
            if s == "total":
                est = test[CARRIER_ESTIMATE_COL].to_numpy()
                rows.append({"fold": f.fold, "cutoff": f.cutoff, "stage": s, "model": "carrier_estimate", "n_test": len(test),
                             "mae": mae(y, est), "bias": float(np.mean(est - y)),
                             "coverage_point": float(np.mean(y <= est))})  # fmt: skip
        # composed path forecast: sum of stage medians vs. the direct total model (diagnostic)
        if all(s in stages for s in PATH_STAGES) and "total" in stages:
            summed = (
                sum(pred[quantile_col(s, 0.5)].to_numpy() for s in PATH_STAGES) if 0.5 in quantiles else None
            )
            if summed is not None:
                rows.append({"fold": f.fold, "cutoff": f.cutoff, "stage": "total", "model": "sum_of_stage_medians",
                             "n_test": len(test), "mae": mae(test[target_col("total")].to_numpy(), summed)})  # fmt: skip
        oos.append(out)
        importance = (
            model.feature_importance("total")
            if importance is None
            else importance.add(model.feature_importance("total"), fill_value=0)
        )

    fm = pd.DataFrame(rows)
    return BacktestResult(
        fold_metrics=fm,
        oos_predictions=pd.concat(oos, ignore_index=True),
        feature_importance=(importance / len(folds)).sort_values(ascending=False),
    )
