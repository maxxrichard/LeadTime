import numpy as np
import pandas as pd

from leadtime.features.build import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, assign_segment, haversine_km
from leadtime.utils.timeseries import asof_window_stats, rolling_count


def test_asof_window_stats_excludes_future_and_same_instant():
    ev = np.array([1, 2, 3, 10], dtype="int64") * 86_400_000_000_000
    vals = np.array([1.0, 2.0, 3.0, 100.0])
    q = np.array([3, 4, 11], dtype="int64") * 86_400_000_000_000
    n, m = asof_window_stats(ev, vals, q)
    assert n.tolist() == [2, 3, 4]  # event at day 3 is NOT visible to a query at day 3
    np.testing.assert_allclose(m, [1.5, 2.0, 26.5])
    n_w, m_w = asof_window_stats(ev, vals, q, window_days=2)
    assert n_w.tolist() == [2, 2, 1]
    np.testing.assert_allclose(m_w, [1.5, 2.5, 100.0])


def test_rolling_count_half_open_window():
    t = np.array([0, 1, 2, 2, 5], dtype="int64") * 86_400_000_000_000
    assert rolling_count(t, t, 2).tolist() == [0, 1, 2, 2, 0]


def test_haversine_known_distance():
    # Sao Paulo -> Rio de Janeiro ~ 360 km
    d = haversine_km(np.array([-23.55]), np.array([-46.63]), np.array([-22.91]), np.array([-43.17]))
    assert 340 < d[0] < 380


def test_segments(orders):
    seg = assign_segment(orders)
    assert set(seg.unique()) <= {"intra_state", "inter_state", "cross_region"}
    same = orders["customer_state"] == orders["seller_state"]
    assert (seg[same] == "intra_state").all()


def test_feature_table_complete(features):
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    assert not missing
    num = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in num)
    assert (features["seller_backlog"] >= 0).all()
    assert (features["network_orders_7d"] >= 0).all()


def test_seller_history_is_point_in_time(features):
    """Brute-force check: history stats only use hand-overs that happened before the purchase."""
    rng = np.random.default_rng(0)
    for i in rng.choice(len(features), 40, replace=False):
        row = features.iloc[i]
        prev = features[(features.seller_id == row.seller_id) & (features.carrier_ts < row.purchase_ts)]
        assert len(prev) == row.seller_hist_n
        if len(prev):
            assert abs(prev.handling_days.mean() - row.seller_hist_handling_mean) < 1e-9
        else:
            assert np.isnan(row.seller_hist_handling_mean)
