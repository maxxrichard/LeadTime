"""Load the public Olist e-commerce dataset (9 CSVs) and build one canonical row per order.

The same loader is used for the real Kaggle files and for the synthetic Olist-shaped files written by
:mod:`leadtime.data.synthetic`, so the whole pipeline is exercised end-to-end either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leadtime.data.schema import CANONICAL_COLUMNS, STATE_CENTROIDS

OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

_DATE_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def load_raw(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read every Olist CSV found in `raw_dir` (missing optional tables are tolerated)."""
    raw_dir = Path(raw_dir)
    out: dict[str, pd.DataFrame] = {}
    for key, fname in OLIST_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            if key in {"orders", "items", "products", "sellers", "customers"}:
                raise FileNotFoundError(f"Required Olist file missing: {path}")
            continue
        df = pd.read_csv(
            path,
            dtype={
                "customer_zip_code_prefix": str,
                "seller_zip_code_prefix": str,
                "geolocation_zip_code_prefix": str,
            },
        )
        for c in _DATE_COLS.get(key, []):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        out[key] = df
    return out


def _zip_centroids(geo: pd.DataFrame | None) -> pd.DataFrame | None:
    if geo is None or geo.empty:
        return None
    g = geo.copy()
    g["geolocation_zip_code_prefix"] = g["geolocation_zip_code_prefix"].astype(str).str.zfill(5)
    # Olist geolocation has many points per prefix and a few wild outliers; use the median.
    return (
        g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def _attach_geo(
    df: pd.DataFrame, zip_col: str, state_col: str, prefix: str, zips: pd.DataFrame | None
) -> pd.DataFrame:
    lat = pd.Series(np.nan, index=df.index)
    lng = pd.Series(np.nan, index=df.index)
    if zips is not None:
        m = df[[zip_col]].join(zips, on=zip_col)
        lat, lng = m["lat"], m["lng"]
    # Fall back to the state centroid when the prefix is unknown.
    cent = df[state_col].map(STATE_CENTROIDS)
    lat = lat.fillna(cent.map(lambda t: t[0] if isinstance(t, tuple) else np.nan))
    lng = lng.fillna(cent.map(lambda t: t[1] if isinstance(t, tuple) else np.nan))
    df[f"{prefix}_lat"] = lat.astype(float)
    df[f"{prefix}_lng"] = lng.astype(float)
    return df


def build_orders(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the Olist tables into one canonical row per order with the eight timestamps.

    Multi-item orders are collapsed: the seller/product of the first item line is kept, prices,
    freight, weights and volumes are summed, and the earliest shipping limit is used.
    """
    orders = raw["orders"].rename(
        columns={
            "order_purchase_timestamp": "purchase_ts",
            "order_approved_at": "approved_ts",
            "order_delivered_carrier_date": "carrier_ts",
            "order_delivered_customer_date": "delivered_ts",
            "order_estimated_delivery_date": "estimated_ts",
        }
    )

    products = raw["products"].copy()
    if "category_translation" in raw:
        tr = raw["category_translation"].set_index("product_category_name")["product_category_name_english"]
        products["product_category_name"] = (
            products["product_category_name"].map(tr).fillna(products["product_category_name"])
        )
    products["volume_cm3"] = (
        products["product_length_cm"].fillna(0)
        * products["product_height_cm"].fillna(0)
        * products["product_width_cm"].fillna(0)
    )
    products = products[["product_id", "product_category_name", "product_weight_g", "volume_cm3"]]

    items = (
        raw["items"].merge(products, on="product_id", how="left").sort_values(["order_id", "order_item_id"])
    )
    agg = items.groupby("order_id").agg(
        seller_id=("seller_id", "first"),
        product_id=("product_id", "first"),
        product_category=("product_category_name", "first"),
        n_items=("order_item_id", "size"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        shipping_limit_ts=("shipping_limit_date", "min"),
    )
    df = orders.merge(agg, on="order_id", how="inner")

    customers = raw["customers"].rename(columns={"customer_zip_code_prefix": "customer_zip_prefix"})[
        ["customer_id", "customer_zip_prefix", "customer_state"]
    ]
    df = df.merge(customers, on="customer_id", how="left")
    sellers = raw["sellers"].rename(columns={"seller_zip_code_prefix": "seller_zip_prefix"})[
        ["seller_id", "seller_zip_prefix", "seller_state"]
    ]
    df = df.merge(sellers, on="seller_id", how="left")
    for c in ("customer_zip_prefix", "seller_zip_prefix"):
        df[c] = df[c].astype(str).str.zfill(5)

    zips = _zip_centroids(raw.get("geolocation"))
    df = _attach_geo(df, "customer_zip_prefix", "customer_state", "customer", zips)
    df = _attach_geo(df, "seller_zip_prefix", "seller_state", "seller", zips)

    if "payments" in raw:
        pay = (
            raw["payments"]
            .sort_values(["order_id", "payment_sequential"])
            .groupby("order_id")
            .agg(payment_type=("payment_type", "first"), payment_installments=("payment_installments", "max"))
        )
        df = df.merge(pay, on="order_id", how="left")
    else:
        df["payment_type"], df["payment_installments"] = "unknown", 1

    if "reviews" in raw:
        rev = (
            raw["reviews"]
            .sort_values(["order_id", "review_creation_date"])
            .groupby("order_id")
            .agg(
                review_creation_ts=("review_creation_date", "first"),
                review_answer_ts=("review_answer_timestamp", "first"),
                review_score=("review_score", "first"),
            )
        )
        df = df.merge(rev, on="order_id", how="left")
    else:
        df["review_creation_ts"] = pd.NaT
        df["review_answer_ts"] = pd.NaT
        df["review_score"] = np.nan

    df["product_category"] = df["product_category"].fillna("unknown")
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["payment_installments"] = df["payment_installments"].fillna(1).astype(int)
    df["n_items"] = df["n_items"].astype(int)
    return df[CANONICAL_COLUMNS].sort_values("purchase_ts").reset_index(drop=True)
