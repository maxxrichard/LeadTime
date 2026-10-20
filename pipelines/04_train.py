"""Train the per-segment quantile bundle on the gold features and register it in MLflow."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow

from leadtime.config import Config
from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from leadtime.models.quantile import SegmentedQuantileModel
from leadtime.promise.policy import QuantilePromise
from pipelines.common import MLFLOW_EXPERIMENT, get_spark, read_delta, write_delta


def main(config_path: str | None = os.environ.get("LEADTIME_CONFIG")) -> None:
    spark = get_spark("leadtime-train")
    conf = Config.load(config_path)
    pdf = read_delta(spark, "gold", "features").toPandas()  # ~100k rows: fits comfortably on the driver

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"train-{conf.model.backend}") as run:
        model = SegmentedQuantileModel(conf.model, FEATURE_COLUMNS, CATEGORICAL_COLUMNS).fit(pdf)
        mlflow.log_params({"backend": conf.model.backend, "quantiles": conf.model.quantiles, "n_rows": len(pdf),
                           "min_segment_rows": conf.model.min_segment_rows, **conf.model.params})  # fmt: skip
        mlflow.log_metric("n_boosters", model.n_models())
        with tempfile.TemporaryDirectory() as d:
            p = model.save(Path(d) / "leadtime_model.joblib")
            mlflow.log_artifact(str(p), artifact_path="model")
            model.feature_importance("total").to_csv(Path(d) / "feature_importance.csv")
            mlflow.log_artifact(str(Path(d) / "feature_importance.csv"))

        # score the training table so the promise is available downstream (batch inference)
        preds = model.predict(pdf)
        preds.insert(0, "order_id", pdf["order_id"].values)
        preds["purchase_ts"] = pdf["purchase_ts"].values
        preds["carrier_estimate_days"] = pdf["carrier_estimate_days"].values
        pol = QuantilePromise(
            quantile=conf.promise.quantile,
            buffer_days=conf.promise.buffer_days,
            round_up=conf.promise.round_up,
        )
        preds["promised_days"] = pol.promised_days(preds)
        preds["model_run_id"] = run.info.run_id
        out = write_delta(spark.createDataFrame(preds), "gold", "predictions")
        print(f"trained {model.n_models()} boosters (run {run.info.run_id}); predictions -> {out}")


if __name__ == "__main__":
    main()
