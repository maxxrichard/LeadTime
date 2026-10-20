"""Spark pipeline smoke test: runs only when pyspark + a JVM are available."""

import os
import shutil

import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None and "JAVA_HOME" not in os.environ:
    pytest.skip("no JVM available", allow_module_level=True)


def test_silver_and_gold_jobs(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("LEADTIME_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("LEADTIME_RAW_DIR", str(raw_dir))
    import importlib

    import pipelines.common as common

    importlib.reload(common)
    ingest = importlib.import_module("pipelines.01_ingest")
    silver = importlib.import_module("pipelines.02_silver_orders")
    gold = importlib.import_module("pipelines.03_gold_features")
    ingest.main(str(raw_dir))
    silver.main()
    gold.main()
    spark = common.get_spark()
    g = common.read_delta(spark, "gold", "features")
    from leadtime.features.build import FEATURE_COLUMNS

    assert set(FEATURE_COLUMNS) <= set(g.columns)
    assert g.count() > 1000
