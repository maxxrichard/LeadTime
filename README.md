# LeadTime — processing-time & delivery-promise forecasting for an e-commerce fulfillment network

[![ci](https://github.com/maxxrichard/LeadTime/actions/workflows/ci.yml/badge.svg)](https://github.com/maxxrichard/LeadTime/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

LeadTime forecasts, per order and at purchase time, how long every leg of the fulfillment path will
take — **approval → picking/packing → carrier hand-over → last mile** — and turns the forecast into a
calibrated **delivery promise with quantile bounds**. It is built on the public
[Olist e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100k orders,
eight timestamps each) and ships with a synthetic Olist-shaped generator so the whole pipeline runs
offline in a few minutes.

```
Python · pandas · LightGBM / XGBoost · scikit-learn · PySpark · Delta Lake · MLflow · Databricks
```

## What is inside

| capability | where |
|---|---|
| canonical order table with 8 timestamps from the 9 Olist CSVs | `leadtime.data.olist` |
| offline synthetic stand-in with the same schema | `leadtime.data.synthetic` |
| leakage-safe seller / product / route / calendar / load features (point-in-time) | `leadtime.features.build` |
| per-segment LightGBM **quantile** models for every stage + the total (XGBoost drop-in) | `leadtime.models.quantile` |
| **rolling-origin backtest** against the carrier estimate and a route-quantile baseline | `leadtime.evaluation.backtest` |
| delivery-promise **policies** (model P80, composed per-stage, carrier estimate), calibration, frontier | `leadtime.promise.policy` |
| **A/B evaluation design**: lead-time-matched simulation, z-test, bootstrap CI, sample size | `leadtime.promise.ab_test` |
| monitoring: PSI feature drift + weekly MAE / coverage drift | `leadtime.monitoring.drift` |
| **PySpark on Databricks** pipeline (bronze → silver → gold → train → evaluate → monitor) on Delta + MLflow | `pipelines/`, `databricks/` |
| CLI + markdown/PNG report | `leadtime.cli`, `leadtime.evaluation.report` |

## Quick start

```bash
git clone https://github.com/maxxrichard/LeadTime.git && cd LeadTime
make venv                      # python -m venv .venv && pip install -e ".[dev]"
make prototype                 # synth -> prepare -> features -> backtest -> train  (~5 min on a laptop)
open reports/backtest_report.md
```

Step by step (every command takes `--help`):

```bash
leadtime synth                               # 40k synthetic Olist-shaped orders -> data/raw/*.csv
leadtime prepare                             # join 9 CSVs -> data/processed/orders.parquet
leadtime features                            # -> data/processed/features.parquet (46 features)
leadtime backtest --backend lightgbm         # rolling-origin backtest + promise policies + A/B -> reports/
leadtime train --mlflow-uri ./mlruns         # production bundle -> models/leadtime_model.joblib
leadtime predict                             # score a feature table and attach the promised date
leadtime monitor --reference-end 2018-06-01  # PSI drift + weekly coverage drift
```

To run on the **real Olist data**: `make olist` (Kaggle API token required), then the same commands
from `prepare` onwards. On **Databricks**: `python -m build && databricks bundle deploy -t dev`
(see [pipelines/README.md](pipelines/README.md)).

## Results (synthetic 40k-order run, 6 rolling-origin folds)

| metric (out of sample, 14,965 orders) | carrier estimate | LeadTime |
|---|---|---|
| delivery-date MAE (median forecast) | 2.86 d | **1.99 d (-31 %)** |
| on-time rate of the promise | 66.0 % | **81.6 %** at the P80 promise (80.0 % after buffer calibration) |
| late deliveries at *equal* mean promised lead time (A/B simulation) | 33.3 % | **31.2 % (-6.5 %, 95 % CI -10.8 … -2.3 %, p = 0.005)** |
| per-stage MAE (approval / handling / transit) | – | 0.08 / 1.23 / 1.24 d |

![Promise frontier](docs/results/promise_frontier.png)

Full tables and figures: [docs/results.md](docs/results.md). Numbers are from the synthetic stand-in
shipped with the repo; re-run `leadtime backtest` on the Kaggle files for the real-data figures.

## How it works

1. **Stages & targets.** Each order yields four durations: approval, handling, transit and total
   (purchase → delivered). Orders with incomplete or non-monotone timestamps are dropped.
2. **Point-in-time features.** Seller handling history, SLA-miss rate, backlog and load, route
   transit history, product and calendar attributes — every statistic only uses events completed
   *before* the purchase timestamp.
3. **Quantile GBMs per segment.** One LightGBM booster per (segment, stage, quantile) on the log
   target; thin segments fall back to a global model; crossing quantiles are repaired.
4. **Backtest.** Train on orders *delivered* before the cutoff, score orders *purchased* in the next
   30 days, roll forward. Baselines: the carrier estimate and route-level empirical quantiles.
5. **Promise.** The P80 of the total lead time (rounded up to whole days) is the promise; a
   calibrated buffer hits any target on-time rate; the composed policy rebuilds the total from the
   per-stage quantile functions for arbitrary paths.
6. **A/B design.** Treatment is lead-time matched to control so the test measures promise *accuracy*,
   not just promising later; the late rate is the primary metric.

Read more: [docs/architecture.md](docs/architecture.md) · [docs/modelling.md](docs/modelling.md) ·
[data/README.md](data/README.md).

## Repository layout

```
configs/default.yaml      all knobs (quantiles, model params, backtest windows, promise, A/B, monitoring)
src/leadtime/             the package (see table above)
pipelines/                PySpark jobs 01..06 + Databricks helpers
databricks/databricks.yml Databricks Asset Bundle (job with six tasks)
scripts/download_olist.py Kaggle download helper
tests/                    unit tests incl. point-in-time leakage checks
docs/                     architecture, modelling notes, results of the shipped run
```

## Development

```bash
make lint   # ruff
make test   # pytest (Spark smoke test runs only when pyspark + a JVM are present)
```

## License

MIT
