import numpy as np
import pandas as pd
import pytest

from leadtime.config import BacktestConfig, ModelConfig
from leadtime.evaluation.backtest import make_folds, run_backtest
from leadtime.evaluation.metrics import coverage, pinball_loss
from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from leadtime.promise.ab_test import required_sample_size, simulate_ab, two_proportion_ztest
from leadtime.promise.policy import (
    CarrierEstimatePromise,
    ComposedQuantilePromise,
    QuantilePromise,
    calibrate_buffer,
    evaluate_promise,
)

FAST = ModelConfig(quantiles=[0.1, 0.5, 0.8, 0.9], min_segment_rows=10_000_000,
                   params={"n_estimators": 40, "learning_rate": 0.2, "num_leaves": 7, "min_child_samples": 10, "verbose": -1})  # fmt: skip
BT = BacktestConfig(n_folds=2, horizon_days=30, step_days=30, min_train_days=120)


def test_folds_are_leakage_free(features):
    for f in make_folds(features, BT):
        train, test = features.iloc[f.train_idx], features.iloc[f.test_idx]
        assert (train.delivered_ts <= f.cutoff).all()
        assert (test.purchase_ts > f.cutoff).all()
        assert (test.purchase_ts <= f.cutoff + pd.Timedelta(days=BT.horizon_days)).all()


@pytest.fixture(scope="module")
def backtest(features):
    return run_backtest(features, FAST, BT, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, verbose=False)


def test_backtest_outputs(backtest):
    s = backtest.summary().reset_index()
    tot = s[s.stage == "total"].set_index("model")
    assert {
        "lightgbm_segmented",
        "route_quantile_baseline",
        "carrier_estimate",
        "sum_of_stage_medians",
    } <= set(tot.index)
    assert tot.loc["lightgbm_segmented", "mae"] < tot.loc["carrier_estimate", "mae"]
    oos = backtest.oos_predictions
    assert oos.order_id.is_unique and set(oos.fold) == {1, 2}
    assert 0.6 < coverage(oos.total_days, oos.total_q80) < 0.95


def test_metrics_basics():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.8) == 0
    assert coverage(y, np.array([0.5, 2.0, 2.5])) == pytest.approx(1 / 3)


def test_promise_policies_and_calibration(backtest):
    oos = backtest.oos_predictions
    p80 = QuantilePromise(quantile=0.8)
    ev = evaluate_promise(oos, p80.promised_days(oos))
    assert 0.7 < ev["on_time_rate"] <= 1.0
    cal = calibrate_buffer(oos, p80, target_on_time=0.9)
    assert evaluate_promise(oos, cal.promised_days(oos))["on_time_rate"] >= 0.9 - 0.02
    comp = ComposedQuantilePromise(quantile=0.8, n_samples=64)
    d = comp.promised_days(oos)
    assert np.isfinite(d).all() and (d > 0).all()
    assert np.isfinite(CarrierEstimatePromise().promised_days(oos)).all()


def test_ab_simulation(backtest):
    oos = backtest.oos_predictions
    res = simulate_ab(
        oos, CarrierEstimatePromise(round_up=False), QuantilePromise(quantile=0.8), n_bootstrap=100
    )
    assert res.n_control + res.n_treatment == len(oos)
    assert (
        abs(res.mean_promised_treatment - res.mean_promised_control) < 0.6
    )  # lead-time matched (ceil rounding)
    assert res.rel_lift_ci_low <= res.rel_lift <= res.rel_lift_ci_high
    assert 0 <= res.p_value <= 1


def test_stats_helpers():
    z, p = two_proportion_ztest(30, 100, 50, 100)
    assert z < 0 and p < 0.01
    assert required_sample_size(0.3, -0.2) > required_sample_size(0.3, -0.4)


def test_interpolated_quantiles_and_quantile_matching(backtest):
    from leadtime.promise.ab_test import match_lead_time
    from leadtime.promise.policy import available_quantiles, interpolated_quantile

    oos = backtest.oos_predictions
    assert available_quantiles(oos) == [0.1, 0.5, 0.8, 0.9]
    mid = interpolated_quantile(oos, 0.65)
    np.testing.assert_allclose(mid, 0.5 * oos.total_q50 + 0.5 * oos.total_q80)
    np.testing.assert_allclose(interpolated_quantile(oos, 0.8), oos.total_q80)
    assert (interpolated_quantile(oos, 0.99) == oos.total_q90).all()  # clipped to the grid
    target = float(np.mean(CarrierEstimatePromise(round_up=False).promised_days(oos)))
    m = match_lead_time(oos, QuantilePromise(quantile=0.8, round_up=False), target, mode="quantile")
    assert 0.1 <= m.quantile <= 0.9
    assert abs(float(np.mean(m.promised_days(oos))) - target) < 0.05
