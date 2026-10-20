"""Leakage-safe feature engineering for scoring an order at purchase time.

Feature families (as in the project brief):
  * seller   - point-in-time handling history, hand-over SLA misses, tenure
  * product  - category, weight, volume, price, freight, item count
  * route    - origin/destination state, region, haversine distance, route transit history
  * calendar - day-of-week, hour, month, holidays, peak season
  * load     - seller and network order volume in trailing windows, open seller backlog

All historical statistics use only orders whose *relevant stage completed before* the order's
purchase timestamp, so nothing from the future leaks into the training rows (the backtest in
:mod:`leadtime.evaluation.backtest` additionally restricts training rows by delivery date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadtime.data.schema import BR_REGIONS, CARRIER_ESTIMATE_COL, SEGMENT_COL
from leadtime.utils.timeseries import asof_window_stats, count_before, rolling_count, to_int_ns

CATEGORICAL_COLUMNS = [
    "product_category",
    "seller_state",
    "customer_state",
    "payment_type",
    "customer_region",
    SEGMENT_COL,
]

NUMERIC_COLUMNS = [
    # product
    "n_items", "total_weight_g", "total_volume_cm3", "total_price", "total_freight", "freight_ratio", "payment_installments",
    # route
    "distance_km", "same_state", "same_region", CARRIER_ESTIMATE_COL, "shipping_limit_days",
    # calendar
    "purchase_dow", "purchase_hour", "purchase_month", "purchase_weekofyear", "is_weekend", "is_holiday",
    "days_to_next_holiday", "is_peak_season", "days_to_black_friday",
    # seller history
    "seller_hist_n", "seller_hist_handling_mean", "seller_hist_n_90d", "seller_hist_handling_mean_90d",
    "seller_hist_handling_mean_30d", "seller_hist_sla_miss_rate_90d", "seller_hist_transit_mean_90d", "seller_tenure_days",
    # route history
    "route_hist_n", "route_hist_transit_mean", "route_hist_n_90d", "route_hist_transit_mean_90d",
    "dest_state_hist_transit_mean_90d",
    # load
    "seller_orders_1d", "seller_orders_7d", "seller_orders_30d", "seller_backlog", "seller_load_ratio",
    "network_orders_7d", "network_orders_1d",
]  # fmt: skip

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

_BR_FIXED_HOLIDAYS = [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25)]
_BR_MOVEABLE_HOLIDAYS = {
    2016: ["2016-02-08", "2016-02-09", "2016-03-25", "2016-05-26"],
    2017: ["2017-02-27", "2017-02-28", "2017-04-14", "2017-06-15"],
    2018: ["2018-02-12", "2018-02-13", "2018-03-30", "2018-05-31"],
    2019: ["2019-03-04", "2019-03-05", "2019-04-19", "2019-06-20"],
}


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lng2 - lng1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def holiday_calendar(years: list[int]) -> pd.DatetimeIndex:
    days: list[pd.Timestamp] = []
    for y in years:
        days += [pd.Timestamp(y, m, d) for m, d in _BR_FIXED_HOLIDAYS]
        days += [pd.Timestamp(s) for s in _BR_MOVEABLE_HOLIDAYS.get(y, [])]
    return pd.DatetimeIndex(sorted(set(days)))


def black_fridays(years: list[int]) -> pd.DatetimeIndex:
    out = []
    for y in years:
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        out.append(nov[nov.dayofweek == 4][3])  # fourth Friday
    return pd.DatetimeIndex(out)


# --------------------------------------------------------------------------------------------- #
# Static feature families
# --------------------------------------------------------------------------------------------- #
def assign_segment(df: pd.DataFrame) -> pd.Series:
    same_state = df["customer_state"] == df["seller_state"]
    same_region = df["customer_state"].map(BR_REGIONS) == df["seller_state"].map(BR_REGIONS)
    seg = np.where(same_state, "intra_state", np.where(same_region, "inter_state", "cross_region"))
    return pd.Series(seg, index=df.index, name=SEGMENT_COL)


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df["freight_ratio"] = df["total_freight"] / (df["total_price"] + 1e-6)
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_km"] = haversine_km(
        df["seller_lat"].values, df["seller_lng"].values, df["customer_lat"].values, df["customer_lng"].values
    )
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["customer_region"] = df["customer_state"].map(BR_REGIONS).fillna("UNK")
    df["same_region"] = (df["customer_region"] == df["seller_state"].map(BR_REGIONS)).astype(int)
    df[SEGMENT_COL] = assign_segment(df)
    return df


def add_calendar_features(df: pd.DataFrame, ts_col: str = "purchase_ts") -> pd.DataFrame:
    ts = df[ts_col]
    df["purchase_dow"] = ts.dt.dayofweek
    df["purchase_hour"] = ts.dt.hour
    df["purchase_month"] = ts.dt.month
    df["purchase_weekofyear"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["purchase_dow"] >= 5).astype(int)
    years = sorted(set(ts.dt.year.unique().tolist()) | {ts.dt.year.max() + 1})
    hol = holiday_calendar(years)
    day = ts.dt.normalize()
    df["is_holiday"] = day.isin(hol).astype(int)
    nxt = np.searchsorted(hol.values, day.values, side="left")
    nxt = np.minimum(nxt, len(hol) - 1)
    df["days_to_next_holiday"] = ((hol.values[nxt] - day.values) / np.timedelta64(1, "D")).astype(float)
    bf = black_fridays(years)
    nb = np.minimum(np.searchsorted(bf.values, day.values, side="left"), len(bf) - 1)
    df["days_to_black_friday"] = ((bf.values[nb] - day.values) / np.timedelta64(1, "D")).astype(float)
    df["is_peak_season"] = (
        ((ts.dt.month == 11) & (ts.dt.day >= 15)) | ((ts.dt.month == 12) & (ts.dt.day <= 23))
    ).astype(int)
    return df


# --------------------------------------------------------------------------------------------- #
# Point-in-time history & load features (grouped: these functions also back the Spark pipeline)
# --------------------------------------------------------------------------------------------- #
def _nan_if_missing(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype="float64") if col in df else np.full(len(df), np.nan)


def seller_history_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features for the rows of ONE seller. Uses hand-over/delivery events completed before purchase."""
    q = to_int_ns(g["purchase_ts"])
    carrier = to_int_ns(g["carrier_ts"]).astype("float64")
    carrier[g["carrier_ts"].isna().values] = np.nan
    delivered = to_int_ns(g["delivered_ts"]).astype("float64")
    delivered[g["delivered_ts"].isna().values] = np.nan

    handling = _nan_if_missing(g, "handling_days")
    transit = _nan_if_missing(g, "transit_days")
    sla_miss = np.where(
        np.isnan(carrier), np.nan, (g["carrier_ts"] > g["shipping_limit_ts"]).to_numpy(dtype="float64")
    )

    # events keyed on hand-over time (handling known once handed over)
    ev_c = np.where(np.isnan(carrier), np.iinfo("int64").max, carrier).astype("int64")
    n_all, m_all = asof_window_stats(ev_c, handling, q)
    n_90, m_90 = asof_window_stats(ev_c, handling, q, 90)
    _, m_30 = asof_window_stats(ev_c, handling, q, 30)
    _, miss_90 = asof_window_stats(ev_c, sla_miss, q, 90)
    # events keyed on delivery time (transit known once delivered)
    ev_d = np.where(np.isnan(delivered), np.iinfo("int64").max, delivered).astype("int64")
    _, t_90 = asof_window_stats(ev_d, transit, q, 90)

    first = q.min()
    out = pd.DataFrame(
        {
            "seller_hist_n": n_all,
            "seller_hist_handling_mean": m_all,
            "seller_hist_n_90d": n_90,
            "seller_hist_handling_mean_90d": m_90,
            "seller_hist_handling_mean_30d": m_30,
            "seller_hist_sla_miss_rate_90d": miss_90,
            "seller_hist_transit_mean_90d": t_90,
            "seller_tenure_days": (q - first) / 86_400e9,
            "seller_orders_1d": rolling_count(q, q, 1),
            "seller_orders_7d": rolling_count(q, q, 7),
            "seller_orders_30d": rolling_count(q, q, 30),
            "seller_backlog": count_before(q, q)
            - count_before(np.where(np.isnan(carrier), np.iinfo("int64").max, carrier).astype("int64"), q),
        },
        index=g.index,
    )
    out["seller_load_ratio"] = out["seller_orders_7d"] / (out["seller_orders_30d"] / 30.0 * 7.0 + 1.0)
    return out


def route_history_features(g: pd.DataFrame) -> pd.DataFrame:
    """Transit history for ONE route (seller_state -> customer_state), keyed on delivery time."""
    q = to_int_ns(g["purchase_ts"])
    delivered = to_int_ns(g["delivered_ts"]).astype("float64")
    delivered[g["delivered_ts"].isna().values] = np.nan
    transit = _nan_if_missing(g, "transit_days")
    ev = np.where(np.isnan(delivered), np.iinfo("int64").max, delivered).astype("int64")
    n_all, m_all = asof_window_stats(ev, transit, q)
    n_90, m_90 = asof_window_stats(ev, transit, q, 90)
    return pd.DataFrame(
        {
            "route_hist_n": n_all,
            "route_hist_transit_mean": m_all,
            "route_hist_n_90d": n_90,
            "route_hist_transit_mean_90d": m_90,
        },
        index=g.index,
    )


def dest_state_history_features(g: pd.DataFrame) -> pd.DataFrame:
    """Coarser fallback for sparse routes: transit history into ONE destination state."""
    q = to_int_ns(g["purchase_ts"])
    delivered = to_int_ns(g["delivered_ts"]).astype("float64")
    delivered[g["delivered_ts"].isna().values] = np.nan
    transit = _nan_if_missing(g, "transit_days")
    ev = np.where(np.isnan(delivered), np.iinfo("int64").max, delivered).astype("int64")
    _, m_90 = asof_window_stats(ev, transit, q, 90)
    return pd.DataFrame({"dest_state_hist_transit_mean_90d": m_90}, index=g.index)


def network_features(df: pd.DataFrame) -> pd.DataFrame:
    q = to_int_ns(df["purchase_ts"])
    return pd.DataFrame(
        {"network_orders_7d": rolling_count(q, q, 7), "network_orders_1d": rolling_count(q, q, 1)},
        index=df.index,
    )


def _apply_grouped(df: pd.DataFrame, keys: list[str], fn) -> pd.DataFrame:
    parts = [fn(df.iloc[idx]) for _, idx in df.groupby(keys, sort=False).indices.items()]
    return pd.concat(parts).reindex(df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with every feature column added (targets must already be present for history features)."""
    df = df.copy()
    df = add_product_features(df)
    df = add_route_features(df)
    df = add_calendar_features(df)
    df["route_key"] = df["seller_state"].astype(str) + "->" + df["customer_state"].astype(str)
    df = df.join(_apply_grouped(df, ["seller_id"], seller_history_features))
    df = df.join(_apply_grouped(df, ["route_key"], route_history_features))
    df = df.join(_apply_grouped(df, ["customer_state"], dest_state_history_features))
    df = df.join(network_features(df))
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype(str)
    return df
