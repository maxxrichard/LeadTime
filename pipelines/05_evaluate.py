"""Rolling-origin backtest + promise A/B simulation; metrics are written to Delta and MLflow."""

from __future__ import annotations

import os

import mlflow

from leadtime.config import Config
from leadtime.evaluation.backtest import run_backtest
from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from leadtime.promise.ab_test import simulate_ab
from leadtime.promise.policy import CarrierEstimatePromise, QuantilePromise, evaluate_promise
from pipelines.common import MLFLOW_EXPERIMENT, get_spark, read_delta, write_delta


def main(config_path: str | None = os.environ.get("LEADTIME_CONFIG")) -> None:
    spark = get_spark("leadtime-evaluate")
    conf = Config.load(config_path)
    pdf = read_delta(spark, "gold", "features").toPandas()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"backtest-{conf.model.backend}"):
        bt = run_backtest(pdf, conf.model, conf.backtest, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, verbose=False)
        summary = bt.summary().reset_index()
        tot = summary[summary.stage == "total"].set_index("model")
        for m, r in tot.iterrows():
            mlflow.log_metric(f"total_mae__{m}", float(r["mae"]))
        control = CarrierEstimatePromise(round_up=False)
        treatment = QuantilePromise(
            quantile=conf.promise.quantile,
            buffer_days=conf.promise.buffer_days,
            round_up=conf.promise.round_up,
        )
        oos = bt.oos_predictions
        for p in (control, treatment):
            ev = evaluate_promise(oos, p.promised_days(oos))
            mlflow.log_metrics(
                {
                    f"{p.name}__on_time_rate": ev["on_time_rate"],
                    f"{p.name}__mean_promised_days": ev["mean_promised_days"],
                }
            )
        ab = simulate_ab(oos, control, treatment, seed=conf.ab.seed, n_bootstrap=conf.ab.n_bootstrap, alpha=conf.ab.alpha,
                         power=conf.ab.power, match=conf.ab.match_lead_time)  # fmt: skip
        mlflow.log_metrics({"ab_rel_lift_late_rate": ab.rel_lift, "ab_p_value": ab.p_value})

        fm = bt.fold_metrics.copy()
        fm["backend"] = conf.model.backend
        write_delta(spark.createDataFrame(fm), "gold", "backtest_metrics", mode="append")
        write_delta(spark.createDataFrame(oos), "gold", "oos_predictions")
        print(summary.to_string(index=False))
        print(ab)


if __name__ == "__main__":
    main()
