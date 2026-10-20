"""Canonical order schema, fulfillment stages and Brazilian geography lookups."""

from __future__ import annotations

from dataclasses import dataclass

# The eight timestamps carried by every Olist order (orders + items + reviews tables).
TIMESTAMP_COLS = [
    "purchase_ts",  # order_purchase_timestamp
    "approved_ts",  # order_approved_at
    "carrier_ts",  # order_delivered_carrier_date (hand-over to carrier)
    "delivered_ts",  # order_delivered_customer_date
    "estimated_ts",  # order_estimated_delivery_date (the "carrier estimate" shown to the customer)
    "shipping_limit_ts",  # shipping_limit_date (seller SLA to hand over)
    "review_creation_ts",  # review_creation_date
    "review_answer_ts",  # review_answer_timestamp
]


@dataclass(frozen=True)
class Stage:
    name: str
    start: str
    end: str
    description: str


# The fulfillment path we forecast: approval -> picking/packing -> carrier hand-over -> last mile.
# Olist does not expose an intermediate carrier-depot timestamp, so transit covers line-haul + last mile.
STAGES: list[Stage] = [
    Stage("approval", "purchase_ts", "approved_ts", "Payment approval (purchase -> approval)"),
    Stage("handling", "approved_ts", "carrier_ts", "Picking & packing (approval -> carrier hand-over)"),
    Stage("transit", "carrier_ts", "delivered_ts", "Carrier line-haul + last mile (hand-over -> delivered)"),
    Stage("total", "purchase_ts", "delivered_ts", "End-to-end lead time (purchase -> delivered)"),
]
STAGE_NAMES = [s.name for s in STAGES]
PATH_STAGES = ["approval", "handling", "transit"]  # the stages whose sum is `total`


def target_col(stage: str) -> str:
    return f"{stage}_days"


CARRIER_ESTIMATE_COL = "carrier_estimate_days"

SEGMENT_COL = "segment"
SEGMENTS = ["intra_state", "inter_state", "cross_region"]

BR_REGIONS: dict[str, str] = {
    "AC": "N", "AM": "N", "AP": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MT": "CO", "MS": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}  # fmt: skip

# Approximate state centroids (lat, lng) used by the synthetic generator and as a geolocation fallback.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AC": (-9.0, -70.5), "AL": (-9.6, -36.6), "AM": (-3.4, -65.0), "AP": (1.4, -51.8), "BA": (-12.6, -41.7),
    "CE": (-5.2, -39.5), "DF": (-15.8, -47.9), "ES": (-19.6, -40.7), "GO": (-16.0, -49.6), "MA": (-5.1, -45.0),
    "MG": (-18.5, -44.4), "MS": (-20.5, -54.6), "MT": (-12.9, -55.9), "PA": (-3.8, -52.5), "PB": (-7.1, -36.7),
    "PE": (-8.3, -37.9), "PI": (-7.4, -42.8), "PR": (-24.6, -51.6), "RJ": (-22.3, -42.7), "RN": (-5.8, -36.6),
    "RO": (-10.9, -63.0), "RR": (2.1, -61.4), "RS": (-30.0, -53.3), "SC": (-27.2, -50.5), "SE": (-10.6, -37.4),
    "SP": (-22.3, -48.6), "TO": (-10.2, -48.3),
}  # fmt: skip

# Approximate CEP (postal code) prefix ranges per state, inclusive.
STATE_ZIP_RANGES: dict[str, tuple[int, int]] = {
    "SP": (1000, 19999), "RJ": (20000, 28999), "ES": (29000, 29999), "MG": (30000, 39999), "BA": (40000, 48999),
    "SE": (49000, 49999), "PE": (50000, 56999), "AL": (57000, 57999), "PB": (58000, 58999), "RN": (59000, 59999),
    "CE": (60000, 63999), "PI": (64000, 64999), "MA": (65000, 65999), "PA": (66000, 68899), "AP": (68900, 68999),
    "AM": (69000, 69299), "RR": (69300, 69399), "AC": (69900, 69999), "DF": (70000, 72799), "GO": (72800, 76799),
    "RO": (76800, 76999), "TO": (77000, 77999), "MT": (78000, 78899), "MS": (79000, 79999), "PR": (80000, 87999),
    "SC": (88000, 89999), "RS": (90000, 99999),
}  # fmt: skip

CANONICAL_COLUMNS = [
    "order_id", "customer_id", "seller_id", "product_id", "order_status",
    *TIMESTAMP_COLS,
    "customer_state", "customer_zip_prefix", "customer_lat", "customer_lng",
    "seller_state", "seller_zip_prefix", "seller_lat", "seller_lng",
    "product_category", "n_items", "total_weight_g", "total_volume_cm3", "total_price", "total_freight",
    "payment_type", "payment_installments", "review_score",
]  # fmt: skip
