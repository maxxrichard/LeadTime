import numpy as np
import pandas as pd

from leadtime.data.olist import OLIST_FILES
from leadtime.data.schema import CANONICAL_COLUMNS, TIMESTAMP_COLS
from leadtime.features.stages import add_stage_targets


def test_synthetic_writes_all_olist_files(raw_dir):
    for fname in OLIST_FILES.values():
        assert (raw_dir / fname).exists(), fname


def test_canonical_schema_and_timestamps(orders):
    assert list(orders.columns)[: len(CANONICAL_COLUMNS)] == CANONICAL_COLUMNS
    for c in TIMESTAMP_COLS:
        assert pd.api.types.is_datetime64_any_dtype(orders[c]), c
    assert orders["order_id"].is_unique
    assert (orders["order_status"] == "delivered").all()


def test_stage_path_is_monotone_and_sums(orders):
    df = add_stage_targets(orders)
    for c in ("approval_days", "handling_days", "transit_days", "total_days"):
        assert (df[c] >= 0).all()
    np.testing.assert_allclose(
        df["approval_days"] + df["handling_days"] + df["transit_days"], df["total_days"], atol=1e-6
    )


def test_carrier_estimate_is_beatable_but_sane(orders):
    err = (orders["estimated_ts"] - orders["delivered_ts"]).dt.total_seconds() / 86400
    assert 1.0 < err.abs().mean() < 6.0
    assert 0.1 < (err < 0).mean() < 0.6  # a meaningful share of late deliveries
