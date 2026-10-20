# Architecture

```
                 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 Olist CSVs ───► │ 01 ingest    │──►│ 02 silver    │──►│ 03 gold      │──►│ 04 train     │──►│ 06 monitor   │
 (9 files)       │ bronze Delta │   │ orders + 8 ts│   │ features     │   │ MLflow + preds│  │ PSI, coverage│
                 └──────────────┘   └──────────────┘   └──────┬───────┘   └──────────────┘   └──────────────┘
                                                              │           ┌──────────────┐          ▲
                                                              └──────────►│ 05 evaluate  │──────────┘
                                                                          │ backtest, A/B│
                                                                          └──────────────┘
```

The repository has two execution surfaces that share one code base (`src/leadtime`):

| | offline prototype (`leadtime` CLI) | production pipeline (`pipelines/`, Databricks) |
|---|---|---|
| storage | parquet in `data/processed`, `reports/` | Delta tables, medallion layout |
| compute | pandas + LightGBM on one machine | Spark for joins/static features, `applyInPandas` for grouped point-in-time features, LightGBM on the driver |
| tracking | markdown/PNG report, optional MLflow | MLflow experiment per run, metrics tables in Delta |
| orchestration | `leadtime run-all` / `make prototype` | Databricks Asset Bundle job, six tasks |

## Package layout

```
src/leadtime/
  data/        olist.py (loader, canonical schema), synthetic.py (offline stand-in), schema.py (stages, geography)
  features/    stages.py (targets + validity filter), build.py (seller/product/route/calendar/load features)
  models/      quantile.py (QuantileGBM, SegmentedQuantileModel), baselines.py (carrier estimate, route quantiles)
  evaluation/  metrics.py, backtest.py (rolling origin), report.py (markdown + figures)
  promise/     policy.py (quantile / composed / carrier promises, calibration, frontier), ab_test.py
  monitoring/  drift.py (PSI, performance drift)
  cli.py       typer CLI
pipelines/     01..06 Spark jobs + common.py (Delta paths, session)
databricks/    databricks.yml asset bundle
```

## Leakage control

Three independent guards keep future information out of a training row:

1. **Point-in-time features.** Every history statistic only counts events (hand-over, delivery)
   that happened *strictly before* the order's purchase timestamp (`utils/timeseries.asof_window_stats`).
2. **Label maturity in the backtest.** A fold trained at cutoff *T* only uses orders *delivered*
   before *T* and *purchased* at least 21 days before *T* (later purchases are right-censored: only
   the fast ones have a label yet), and is scored on orders *purchased* after *T*.
3. **Shared feature code.** The Spark job calls the same grouped pandas functions as the offline
   builder, so the production table cannot silently diverge from what was backtested.

## Per-path forecasting

The three path stages (approval, handling, transit) and the end-to-end total are modelled
separately. The direct total model drives the promise; the stage models explain *where* time is
spent and can be recombined by `ComposedQuantilePromise` (Monte-Carlo over per-stage quantile
functions) for routes that skip or swap a leg.
