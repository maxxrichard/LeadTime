"""Shared helpers for the PySpark / Databricks pipeline (medallion layout on Delta Lake).

Layout (``LEADTIME_LAKE_ROOT`` defaults to ``./lake``; on Databricks point it at a UC volume or DBFS path)::

    bronze/<table>        raw Olist CSVs as Delta
    silver/orders         one canonical row per order, eight timestamps, stage targets
    gold/features         model-ready feature table
    gold/predictions      scored orders + promise
    gold/backtest_metrics rolling-origin metrics per fold
    gold/feature_drift    PSI per feature per run
"""

from __future__ import annotations

import os
from pathlib import Path

LAKE_ROOT = os.environ.get("LEADTIME_LAKE_ROOT", "./lake")
RAW_DIR = os.environ.get("LEADTIME_RAW_DIR", "data/raw")
MLFLOW_EXPERIMENT = os.environ.get("LEADTIME_MLFLOW_EXPERIMENT", "/Shared/leadtime")


def table_path(layer: str, name: str) -> str:
    return str(Path(LAKE_ROOT) / layer / name)


def get_spark(app_name: str = "leadtime"):
    """Return the active Databricks session or a local Delta-enabled SparkSession."""
    try:  # Databricks / Spark Connect
        from databricks.sdk.runtime import spark  # type: ignore

        return spark
    except Exception:
        pass
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    try:
        from delta import configure_spark_with_delta_pip

        return configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:  # plain Spark without Delta jars: fall back to parquet writes
        return builder.getOrCreate()


def write_delta(
    df, layer: str, name: str, mode: str = "overwrite", partition_by: list[str] | None = None
) -> str:
    path = table_path(layer, name)
    w = df.write.format("delta").mode(mode).option("overwriteSchema", "true")
    if partition_by:
        w = w.partitionBy(*partition_by)
    w.save(path)
    return path


def read_delta(spark, layer: str, name: str):
    return spark.read.format("delta").load(table_path(layer, name))
