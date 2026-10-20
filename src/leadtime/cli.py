"""Command-line entry point: `leadtime --help`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from leadtime import __version__
from leadtime.config import Config
from leadtime.data.schema import CARRIER_ESTIMATE_COL

app = typer.Typer(help="LeadTime: processing-time & delivery-promise forecasting.", no_args_is_help=True)
console = Console()
_CFG: dict[str, Config] = {}


def cfg() -> Config:
    return _CFG.setdefault("cfg", Config())


@app.callback()
def main(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config (defaults to built-ins)."),
):
    _CFG["cfg"] = Config.load(config)


@app.command()
def version():
    rprint(f"leadtime {__version__}")


@app.command()
def synth(
    n_orders: int | None = typer.Option(None),
    seed: int | None = typer.Option(None),
    out: Path | None = typer.Option(None),
):
    """Write synthetic Olist-shaped CSVs (offline stand-in for the Kaggle download)."""
    from leadtime.data.synthetic import generate_olist_like

    c = cfg().data
    out = out or Path(c.raw_dir)
    t = time.time()
    tables = generate_olist_like(n_orders or c.n_synthetic_orders, seed or c.seed, out)
    rprint(
        f"wrote {len(tables)} tables with {len(tables['orders']):,} orders to {out} in {time.time() - t:.1f}s"
    )


@app.command()
def prepare(raw: Path | None = typer.Option(None), out: Path | None = typer.Option(None)):
    """Join the raw Olist CSVs into one canonical, validated order table (parquet)."""
    from leadtime.data.olist import build_orders, load_raw
    from leadtime.features.stages import filter_valid_orders

    c = cfg().data
    raw = raw or Path(c.raw_dir)
    out = out or Path(c.processed_dir) / "orders.parquet"
    orders = build_orders(load_raw(raw))
    valid = filter_valid_orders(orders, c.max_total_days)
    out.parent.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(out, index=False)
    rprint(f"{len(orders):,} orders loaded, {len(valid):,} valid delivered orders written to {out}")


@app.command()
def features(orders: Path | None = typer.Option(None), out: Path | None = typer.Option(None)):
    """Build the leakage-safe feature table."""
    from leadtime.features.build import build_features

    c = cfg().data
    orders = orders or Path(c.processed_dir) / "orders.parquet"
    out = out or Path(c.processed_dir) / "features.parquet"
    t = time.time()
    f = build_features(pd.read_parquet(orders))
    f.to_parquet(out, index=False)
    rprint(f"{len(f):,} rows x {f.shape[1]} columns written to {out} in {time.time() - t:.1f}s")


def _promise_policies(conf: Config):
    from leadtime.promise.policy import CarrierEstimatePromise, ComposedQuantilePromise, QuantilePromise

    q = conf.promise.quantile
    return [
        CarrierEstimatePromise(buffer_days=0.0, round_up=False),
        QuantilePromise(quantile=0.5, name="model_p50", round_up=conf.promise.round_up),
        QuantilePromise(
            quantile=q,
            name=f"model_p{int(q * 100)}",
            buffer_days=conf.promise.buffer_days,
            round_up=conf.promise.round_up,
        ),
        ComposedQuantilePromise(
            quantile=q,
            quantiles=tuple(conf.model.quantiles),
            name=f"composed_p{int(q * 100)}",
            round_up=conf.promise.round_up,
        ),
    ]


@app.command()
def backtest(
    features: Path | None = typer.Option(None),
    out: Path = typer.Option(Path("reports")),
    backend: str | None = typer.Option(None, help="lightgbm | xgboost"),
    n_folds: int | None = typer.Option(None),
):
    """Rolling-origin backtest + promise-policy evaluation + A/B simulation + report."""
    from leadtime.evaluation.backtest import run_backtest
    from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

    conf = cfg()
    if backend:
        conf.model.backend = backend
    if n_folds:
        conf.backtest.n_folds = n_folds
    features = features or Path(conf.data.processed_dir) / "features.parquet"
    df = pd.read_parquet(features)
    t = time.time()
    bt = run_backtest(df, conf.model, conf.backtest, FEATURE_COLUMNS, CATEGORICAL_COLUMNS)
    elapsed = time.time() - t
    meta = {
        "backend": conf.model.backend,
        "quantiles": conf.model.quantiles,
        "n_orders": len(df),
        "folds": int(bt.oos_predictions["fold"].max()),
        "horizon_days": conf.backtest.horizon_days,
        "label_gap_days": conf.backtest.label_gap_days,
        "train_seconds": round(elapsed, 1),
        "features": len(FEATURE_COLUMNS),
    }
    _promise_and_report(bt, conf, out, meta)
    rprint(f"({elapsed:.0f}s of training)")


@app.command()
def promise(reports: Path = typer.Option(Path("reports"), help="Folder holding a previous `backtest` run.")):
    """Re-score promise policies and the A/B simulation on saved out-of-sample predictions (no retraining)."""
    from leadtime.evaluation.backtest import BacktestResult

    conf = cfg()
    reports = Path(reports)
    bt = BacktestResult(
        fold_metrics=pd.read_csv(reports / "fold_metrics.csv", parse_dates=["cutoff"]),
        oos_predictions=pd.read_parquet(reports / "oos_predictions.parquet"),
        feature_importance=pd.read_csv(reports / "feature_importance.csv", index_col=0)["importance"],
    )
    meta = json.loads((reports / "meta.json").read_text()) if (reports / "meta.json").exists() else None
    _promise_and_report(bt, conf, reports, meta)


def _promise_and_report(bt, conf: Config, out: Path, meta: dict | None) -> None:
    from leadtime.evaluation.report import write_report
    from leadtime.promise.ab_test import simulate_ab
    from leadtime.promise.policy import calibrate_buffer, evaluate_promise, policy_frontier

    oos = bt.oos_predictions
    policies = _promise_policies(conf)
    # a calibrated variant: buffer tuned on the earlier half of the folds, scored on all folds
    first_half = oos["fold"] <= oos["fold"].median()
    calibrated = calibrate_buffer(oos[first_half], policies[2], target_on_time=conf.promise.quantile)
    calibrated = calibrated.__class__(**{**calibrated.__dict__, "name": f"{policies[2].name}_calibrated"})
    rows = []
    for p in policies + [calibrated]:
        rows.append(
            {
                "policy": p.name,
                "quantile": getattr(p, "quantile", None),
                "buffer_days": p.buffer_days,
                **evaluate_promise(oos, p.promised_days(oos)),
            }
        )
    promise_eval = pd.DataFrame(rows)
    frontier = policy_frontier(oos, [policies[0], policies[2]])
    ab = simulate_ab(
        oos, policies[0], policies[2], seed=conf.ab.seed, n_bootstrap=conf.ab.n_bootstrap,
        alpha=conf.ab.alpha, power=conf.ab.power, match=conf.ab.match_lead_time, match_mode=conf.ab.match_mode,
    )  # fmt: skip

    path = write_report(
        out, bt, conf.model.quantiles, promise_eval, ab, frontier, conf.promise.quantile, meta
    )
    (Path(out) / "ab_result.json").write_text(json.dumps(ab.to_dict(), indent=2))
    if meta:
        (Path(out) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    promise_eval.to_csv(Path(out) / "promise_policies.csv", index=False)
    frontier.to_csv(Path(out) / "promise_frontier.csv", index=False)
    _print_summary(bt, promise_eval, ab)
    rprint(f"\nreport written to [bold]{path}[/bold]")


def _print_summary(bt, promise_eval, ab):
    s = bt.summary().reset_index()
    tab = Table(title="Backtest (mean over folds)")
    for c in ["stage", "model", "mae", "bias", "pinball_mean", "coverage_q80"]:
        tab.add_column(c)
    for _, r in s.iterrows():
        tab.add_row(
            *[
                f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                for c in ["stage", "model", "mae", "bias", "pinball_mean", "coverage_q80"]
            ]
        )
    console.print(tab)
    tab = Table(title="Promise policies (out-of-sample)")
    for c in ["policy", "buffer_days", "on_time_rate", "mean_promised_days", "promise_mae"]:
        tab.add_column(c)
    for _, r in promise_eval.iterrows():
        tab.add_row(
            *[
                f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                for c in ["policy", "buffer_days", "on_time_rate", "mean_promised_days", "promise_mae"]
            ]
        )
    console.print(tab)
    rprint(f"[bold]A/B:[/bold] {ab}")


@app.command()
def train(features: Path | None = typer.Option(None), out: Path = typer.Option(Path("models/leadtime_model.joblib")),
          mlflow_uri: str | None = typer.Option(None, help="If set, log params/metrics/artifact to MLflow.")):  # fmt: skip
    """Train the production bundle on all data (optionally logged to MLflow)."""
    from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
    from leadtime.models.quantile import SegmentedQuantileModel

    conf = cfg()
    features = features or Path(conf.data.processed_dir) / "features.parquet"
    df = pd.read_parquet(features)
    t = time.time()
    model = SegmentedQuantileModel(conf.model, FEATURE_COLUMNS, CATEGORICAL_COLUMNS).fit(df)
    model.save(out)
    rprint(f"trained {model.n_models()} boosters on {len(df):,} orders in {time.time() - t:.0f}s -> {out}")
    if mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("leadtime")
        with mlflow.start_run(run_name=f"train-{conf.model.backend}"):
            mlflow.log_params(
                {
                    "backend": conf.model.backend,
                    "quantiles": conf.model.quantiles,
                    "n_rows": len(df),
                    **conf.model.params,
                }
            )
            mlflow.log_metric("n_models", model.n_models())
            mlflow.log_artifact(str(out))
            rprint("logged to MLflow")


@app.command()
def predict(model: Path = typer.Option(Path("models/leadtime_model.joblib")), features: Path | None = typer.Option(None),
            out: Path = typer.Option(Path("reports/predictions.parquet"))):  # fmt: skip
    """Score a feature table with a trained bundle and attach the promise."""
    from leadtime.models.quantile import SegmentedQuantileModel
    from leadtime.promise.policy import QuantilePromise

    conf = cfg()
    features = features or Path(conf.data.processed_dir) / "features.parquet"
    df = pd.read_parquet(features)
    m = SegmentedQuantileModel.load(model)
    preds = m.predict(df)
    preds.insert(0, "order_id", df["order_id"].values)
    preds["purchase_ts"] = df["purchase_ts"].values
    preds[CARRIER_ESTIMATE_COL] = df[CARRIER_ESTIMATE_COL].values
    pol = QuantilePromise(
        quantile=conf.promise.quantile, buffer_days=conf.promise.buffer_days, round_up=conf.promise.round_up
    )
    preds["promised_days"] = pol.promised_days(preds)
    preds["promised_date"] = (
        preds["purchase_ts"] + pd.to_timedelta(preds["promised_days"], "D")
    ).dt.normalize()
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    rprint(f"{len(preds):,} orders scored -> {out}")


@app.command()
def monitor(features: Path | None = typer.Option(None), predictions: Path = typer.Option(Path("reports/oos_predictions.parquet")),
            reference_end: str = typer.Option(..., help="Reference window ends here (YYYY-MM-DD); later rows are 'current'."),
            out: Path = typer.Option(Path("reports"))):  # fmt: skip
    """Feature drift (PSI) and weekly forecast-quality drift."""
    from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
    from leadtime.models.quantile import quantile_col
    from leadtime.monitoring.drift import drift_report, performance_drift

    conf = cfg()
    features = features or Path(conf.data.processed_dir) / "features.parquet"
    f = pd.read_parquet(features)
    cut = pd.Timestamp(reference_end)
    ref, cur = f[f.purchase_ts <= cut], f[f.purchase_ts > cut]
    d = drift_report(
        ref, cur, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, conf.monitoring.psi_warn, conf.monitoring.psi_alert
    )
    d.to_csv(Path(out) / "feature_drift.csv", index=False)
    console.print(d.head(12).to_string(index=False))
    if Path(predictions).exists():
        p = pd.read_parquet(predictions)
        q = conf.promise.quantile
        pdft = performance_drift(
            p,
            quantile_col("total", 0.5),
            quantile_col("total", q),
            q,
            tolerance=conf.monitoring.coverage_tolerance,
        )
        pdft.to_csv(Path(out) / "performance_drift.csv", index=False)
        console.print(pdft.tail(10).to_string(index=False))
    rprint(f"drift tables written to {out}")


@app.command("run-all")
def run_all(
    n_orders: int | None = typer.Option(None),
    backend: str | None = typer.Option(None),
    out: Path = typer.Option(Path("reports")),
):
    """synth -> prepare -> features -> backtest -> train (the full offline prototype)."""
    synth(n_orders=n_orders, seed=None, out=None)
    prepare(raw=None, out=None)
    features(orders=None, out=None)
    backtest(features=None, out=out, backend=backend, n_folds=None)
    train(features=None, out=Path("models/leadtime_model.joblib"), mlflow_uri=None)


if __name__ == "__main__":
    app()
