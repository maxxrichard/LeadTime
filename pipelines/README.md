# PySpark / Databricks pipeline

Medallion layout on Delta Lake, orchestrated as one Databricks job (`databricks/databricks.yml`):

```
01_ingest.py          raw Olist CSVs             -> bronze.<table>
02_silver_orders.py   joins + 8 timestamps       -> silver.orders       (Spark-native)
03_gold_features.py   feature table              -> gold.features       (Spark exprs + applyInPandas per seller/route)
04_train.py           per-segment quantile GBMs  -> MLflow run + gold.predictions
05_evaluate.py        rolling-origin backtest    -> gold.backtest_metrics, gold.oos_predictions, MLflow metrics
06_monitor.py         PSI + coverage drift       -> gold.feature_drift, gold.performance_drift
```

The point-in-time feature code is shared with the offline prototype (`leadtime.features.build`):
Spark groups the silver table by seller / route and hands each group to the same pandas function, so
offline backtests and the production table agree by construction.

Run locally (needs Java 17+ and `pip install -e ".[spark,mlflow]"`):

```bash
export LEADTIME_LAKE_ROOT=./lake LEADTIME_RAW_DIR=data/raw
for job in 01_ingest 02_silver_orders 03_gold_features 04_train 05_evaluate 06_monitor; do
  python -m pipelines.$job
done
```

On Databricks: `python -m build && databricks bundle deploy -t dev && databricks bundle run leadtime_pipeline -t dev`.
