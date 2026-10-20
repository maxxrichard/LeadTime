"""Monitoring job: PSI feature drift of the latest window vs. the training reference + coverage drift."""

from __future__ import annotations

import os

import pandas as pd
from pyspark.sql import functions as F

from leadtime.config import Config
from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from leadtime.models.quantile import quantile_col
from leadtime.monitoring.drift import drift_report, performance_drift
from pipelines.common import get_spark, read_delta, write_delta


def main(window_days: int = 28, config_path: str | None = os.environ.get("LEADTIME_CONFIG")) -> None:
    spark = get_spark("leadtime-monitor")
    conf = Config.load(config_path)
    feats = read_delta(spark, "gold", "features")
    max_ts = feats.agg(F.max("purchase_ts")).collect()[0][0]
    cut = pd.Timestamp(max_ts) - pd.Timedelta(days=window_days)
    ref = feats.filter(F.col("purchase_ts") <= F.lit(cut)).toPandas()
    cur = feats.filter(F.col("purchase_ts") > F.lit(cut)).toPandas()
    rep = drift_report(
        ref, cur, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, conf.monitoring.psi_warn, conf.monitoring.psi_alert
    )
    rep["window_end"] = pd.Timestamp(max_ts)
    write_delta(spark.createDataFrame(rep), "gold", "feature_drift", mode="append")
    n_alert = int((rep.status == "alert").sum())
    print(rep.head(10).to_string(index=False))

    try:
        oos = read_delta(spark, "gold", "oos_predictions").toPandas()
        q = conf.promise.quantile
        perf = performance_drift(
            oos,
            quantile_col("total", 0.5),
            quantile_col("total", q),
            q,
            tolerance=conf.monitoring.coverage_tolerance,
        )
        write_delta(spark.createDataFrame(perf), "gold", "performance_drift")
        print(perf.tail(8).to_string(index=False))
    except Exception as e:  # first run before any backtest
        print(f"no out-of-sample predictions yet ({e.__class__.__name__})")
    if n_alert:
        print(f"ALERT: {n_alert} features above PSI {conf.monitoring.psi_alert}")


if __name__ == "__main__":
    main()
