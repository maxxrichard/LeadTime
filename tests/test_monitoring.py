import numpy as np
import pandas as pd

from leadtime.monitoring.drift import categorical_psi, drift_report, performance_drift, psi


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-9
    assert psi(a, rng.normal(size=5000)) < 0.05
    assert psi(a, rng.normal(loc=1.5, size=5000)) > 0.25


def test_categorical_psi_and_report():
    ref = pd.DataFrame({"x": np.arange(1000.0), "c": ["a"] * 800 + ["b"] * 200})
    cur = pd.DataFrame({"x": np.arange(1000.0) + 500, "c": ["a"] * 200 + ["b"] * 800})
    assert categorical_psi(ref["c"], cur["c"]) > 0.25
    rep = drift_report(ref, cur, ["x", "c"], categorical=["c"])
    assert set(rep.status) == {"alert"}


def test_performance_drift_weekly():
    ts = pd.date_range("2018-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {"purchase_ts": ts, "total_days": 5.0, "q50": 5.0, "q80": np.r_[np.full(100, 6.0), np.full(100, 4.0)]}
    )
    d = performance_drift(df, "q50", "q80", 0.8)
    assert d.status.iloc[0] == "alert" and (d.mae == 0).all()
