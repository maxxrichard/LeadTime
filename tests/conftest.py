import pandas as pd
import pytest

from leadtime.data.olist import build_orders, load_raw
from leadtime.data.synthetic import generate_olist_like
from leadtime.features.build import build_features
from leadtime.features.stages import filter_valid_orders


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate_olist_like(n_orders=4000, seed=123, out_dir=d)
    return d


@pytest.fixture(scope="session")
def orders(raw_dir) -> pd.DataFrame:
    return filter_valid_orders(build_orders(load_raw(raw_dir)))


@pytest.fixture(scope="session")
def features(orders) -> pd.DataFrame:
    return build_features(orders)
