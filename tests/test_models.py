import numpy as np
import pandas as pd
import pytest

from leadtime.config import ModelConfig
from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from leadtime.models.baselines import RouteQuantileBaseline
from leadtime.models.quantile import QuantileGBM, SegmentedQuantileModel, quantile_col

FAST = {"n_estimators": 40, "learning_rate": 0.2, "num_leaves": 7, "min_child_samples": 10, "verbose": -1}


def _split(features):
    cut = features.purchase_ts.quantile(0.8)
    return features[features.purchase_ts <= cut], features[features.purchase_ts > cut]


def test_quantile_gbm_monotone_and_calibrated(features):
    train, test = _split(features)
    m = QuantileGBM([0.1, 0.5, 0.9], FEATURE_COLUMNS, CATEGORICAL_COLUMNS, params=FAST).fit(
        train, train["total_days"]
    )
    p = m.predict(test)
    assert p.shape == (len(test), 3)
    assert (np.diff(p, axis=1) >= 0).all()
    cov = (test["total_days"].to_numpy()[:, None] <= p).mean(axis=0)
    assert 0.03 < cov[0] < 0.25 and 0.35 < cov[1] < 0.65 and 0.78 < cov[2] < 0.98


def test_quantile_gbm_handles_unseen_categories(features):
    train, test = _split(features)
    m = QuantileGBM([0.5], FEATURE_COLUMNS, CATEGORICAL_COLUMNS, params=FAST).fit(train, train["total_days"])
    test = test.copy()
    test.loc[test.index[:5], "product_category"] = "brand_new_category"
    assert np.isfinite(m.predict(test)).all()


def test_segmented_model_fallback_and_save_load(features, tmp_path):
    train, test = _split(features)
    cfg = ModelConfig(
        quantiles=[0.5, 0.8], min_segment_rows=10_000_000, params=FAST
    )  # forces global fallback
    m = SegmentedQuantileModel(cfg, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, stages=["handling", "total"]).fit(
        train
    )
    assert not m.segment_models_ and set(m.global_) == {"handling", "total"}
    pred = m.predict(test)
    assert list(pred.columns) == [
        quantile_col("handling", 0.5),
        quantile_col("handling", 0.8),
        quantile_col("total", 0.5),
        quantile_col("total", 0.8),
    ]
    path = m.save(tmp_path / "m.joblib")
    pred2 = SegmentedQuantileModel.load(path).predict(test)
    pd.testing.assert_frame_equal(pred, pred2)

    cfg2 = ModelConfig(quantiles=[0.5], min_segment_rows=50, params=FAST)
    m2 = SegmentedQuantileModel(cfg2, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, stages=["total"]).fit(train)
    assert len(m2.segment_models_) >= 1


def test_route_baseline(features):
    train, test = _split(features)
    b = RouteQuantileBaseline([0.5, 0.9]).fit(train)
    p = b.predict(test)
    assert p.shape == (len(test), 2) and (p[:, 1] >= p[:, 0]).all()


@pytest.mark.skipif(
    pytest.importorskip("xgboost", reason="xgboost not installed") is None, reason="xgboost missing"
)
def test_xgboost_backend(features):
    train, test = _split(features)
    m = QuantileGBM(
        [0.5],
        FEATURE_COLUMNS,
        CATEGORICAL_COLUMNS,
        backend="xgboost",
        params={"n_estimators": 20, "learning_rate": 0.3},
    ).fit(train, train["total_days"])
    assert np.isfinite(m.predict(test)).all()
