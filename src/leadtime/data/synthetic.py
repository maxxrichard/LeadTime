"""Synthetic Olist-shaped fulfillment data for an offline, reproducible prototype.

The generator writes the nine Olist CSV files with the original column names so that the regular
loader (:func:`leadtime.data.olist.build_orders`) and every downstream step run unchanged. The
latent process mimics what the real network shows: seller-specific handling speed, weekend and
peak-season effects, load-dependent handling, distance/region-dependent transit with a heavy tail,
and a conservative, route-level "carrier estimate" that ignores seller and load effects.

It is *not* a statistical replica of Olist; it exists so the repository runs without Kaggle access.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from leadtime.data.olist import OLIST_FILES
from leadtime.data.schema import BR_REGIONS, STATE_CENTROIDS, STATE_ZIP_RANGES
from leadtime.utils.timeseries import rolling_count, to_int_ns

CUSTOMER_STATE_W = {
    "SP": 0.42, "RJ": 0.13, "MG": 0.12, "RS": 0.055, "PR": 0.05, "SC": 0.037, "BA": 0.034, "DF": 0.021,
    "ES": 0.02, "GO": 0.02, "PE": 0.017, "CE": 0.013, "PA": 0.01, "MT": 0.009, "MA": 0.007, "MS": 0.007,
    "PB": 0.005, "PI": 0.005, "RN": 0.005, "AL": 0.004, "SE": 0.003, "TO": 0.003, "RO": 0.002, "AM": 0.0015,
    "AC": 0.0008, "AP": 0.0007, "RR": 0.0005,
}  # fmt: skip
SELLER_STATE_W = {"SP": 0.60, "PR": 0.11, "MG": 0.08, "SC": 0.06, "RJ": 0.06, "RS": 0.04, "GO": 0.015,
                  "DF": 0.01, "ES": 0.008, "BA": 0.007, "PE": 0.005, "CE": 0.005}  # fmt: skip
REGION_TRANSIT_PENALTY = {"N": 4.0, "NE": 3.0, "CO": 1.0, "S": 0.5, "SE": 0.0}

# category -> (share, median weight g, log-sigma, median price)
CATEGORIES = {
    "bed_bath_table": (0.11, 1500, 0.6, 90), "health_beauty": (0.09, 400, 0.7, 110), "sports_leisure": (0.08, 900, 0.9, 100),
    "furniture_decor": (0.08, 4000, 0.9, 120), "computers_accessories": (0.07, 500, 0.9, 120), "housewares": (0.07, 1200, 0.9, 80),
    "watches_gifts": (0.06, 300, 0.5, 180), "telephony": (0.045, 200, 0.6, 70), "garden_tools": (0.04, 3000, 1.0, 130),
    "auto": (0.04, 1500, 1.0, 130), "toys": (0.04, 700, 0.8, 90), "cool_stuff": (0.037, 900, 0.9, 150),
    "perfumery": (0.034, 300, 0.5, 120), "baby": (0.03, 1500, 0.9, 120), "electronics": (0.027, 400, 0.9, 80),
    "stationery": (0.024, 500, 0.8, 60), "fashion_bags_accessories": (0.02, 400, 0.6, 70), "pet_shop": (0.019, 2500, 1.0, 90),
    "office_furniture": (0.016, 9000, 0.6, 250), "consoles_games": (0.014, 400, 0.7, 150), "luggage_accessories": (0.011, 1800, 0.6, 130),
    "musical_instruments": (0.008, 1500, 1.0, 200), "small_appliances": (0.008, 2500, 0.8, 200), "books_general_interest": (0.006, 500, 0.5, 50),
    "home_appliances": (0.006, 4000, 0.9, 250), "construction_tools_construction": (0.005, 3500, 1.0, 120), "food_drink": (0.004, 1200, 0.8, 60),
    "fixed_telephony": (0.003, 400, 0.5, 70), "home_construction": (0.003, 5000, 1.0, 140), "audio": (0.003, 600, 0.8, 110),
    "industry_commerce_and_business": (0.002, 3000, 1.0, 200), "art": (0.002, 1000, 1.0, 80), "christmas_supplies": (0.002, 500, 0.8, 50),
    "drinks": (0.002, 2000, 0.6, 100), "market_place": (0.002, 1000, 0.8, 90), "party_supplies": (0.001, 400, 0.7, 50),
}  # fmt: skip
PAYMENT_TYPES = {"credit_card": 0.74, "boleto": 0.19, "voucher": 0.05, "debit_card": 0.02}

_BR_FIXED_HOLIDAYS = [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25)]


def _hex_id(prefix: str, i: int, seed: int) -> str:
    return hashlib.md5(f"{prefix}-{seed}-{i}".encode()).hexdigest()


def _weighted_choice(rng: np.random.Generator, table: dict[str, float], n: int) -> np.ndarray:
    keys = np.array(list(table.keys()))
    p = np.array(list(table.values()), dtype=float)
    p /= p.sum()
    return rng.choice(keys, size=n, p=p)


def _make_zip_table(rng: np.random.Generator, per_state: int = 120) -> pd.DataFrame:
    rows = []
    for st, (lo, hi) in STATE_ZIP_RANGES.items():
        lat0, lng0 = STATE_CENTROIDS[st]
        spread = 0.9 if st not in {"AM", "PA", "MT", "BA", "MG"} else 1.8
        prefixes = rng.choice(np.arange(lo, hi + 1), size=min(per_state, hi - lo + 1), replace=False)
        for p in prefixes:
            rows.append((st, f"{p:05d}", lat0 + rng.normal(0, spread), lng0 + rng.normal(0, spread)))
    return pd.DataFrame(rows, columns=["state", "zip", "lat", "lng"])


def _haversine(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _sample_purchase_times(rng: np.random.Generator, n: int, start: str, end: str) -> pd.Series:
    days = pd.date_range(start, end, freq="D")
    t = np.linspace(0, 1, len(days))
    intensity = 1.0 + 1.5 * t  # growth
    dow = days.dayofweek.values
    intensity *= np.where(dow >= 5, 0.65, np.where(dow == 4, 0.9, 1.0))
    for d in days:
        if d.month == 11 and d.dayofweek == 4 and 22 <= d.day <= 28:  # Black Friday
            i = days.get_loc(d)
            intensity[i] *= 3.5
            intensity[i + 1 : i + 4] *= 2.0
    intensity *= np.where((days.month == 12) & (days.day >= 24), 0.5, 1.0)
    intensity /= intensity.sum()
    day_idx = rng.choice(len(days), size=n, p=intensity)
    hours = rng.choice(24, size=n, p=_hour_profile())
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    ts = (
        days[day_idx]
        + pd.to_timedelta(hours, "h")
        + pd.to_timedelta(minutes, "m")
        + pd.to_timedelta(seconds, "s")
    )
    return pd.Series(ts).sort_values().reset_index(drop=True)


def _hour_profile() -> np.ndarray:
    p = np.array(
        [2, 1, 0.6, 0.4, 0.3, 0.4, 1, 2.5, 4, 5.5, 6.5, 6.5, 6, 6.5, 7, 6.8, 6.5, 6, 5.5, 5.5, 5.8, 6, 5, 3.5]
    )
    return p / p.sum()


def _is_holiday(ts: pd.Series) -> np.ndarray:
    md = list(zip(ts.dt.month, ts.dt.day, strict=True))
    return np.array([m in _BR_FIXED_HOLIDAYS for m in md])


def generate_olist_like(
    n_orders: int = 40_000,
    seed: int = 42,
    out_dir: str | Path | None = None,
    start: str = "2017-01-01",
    end: str = "2018-08-31",
) -> dict[str, pd.DataFrame]:
    """Generate Olist-shaped tables; optionally write them as CSVs into `out_dir`."""
    rng = np.random.default_rng(seed)
    zips = _make_zip_table(rng)
    zips_by_state = {st: g for st, g in zips.groupby("state")}

    # --- sellers --------------------------------------------------------------------------------
    n_sellers = max(60, n_orders // 35)
    seller_state = _weighted_choice(rng, SELLER_STATE_W, n_sellers)
    seller_zip_rows = [
        zips_by_state[s].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0] for s in seller_state
    ]
    sellers = pd.DataFrame(
        {
            "seller_id": [_hex_id("seller", i, seed) for i in range(n_sellers)],
            "seller_zip_code_prefix": [r["zip"] for r in seller_zip_rows],
            "seller_city": [f"city_{r['zip'][:3]}" for r in seller_zip_rows],
            "seller_state": seller_state,
        }
    )
    seller_lat = np.array([r["lat"] for r in seller_zip_rows])
    seller_lng = np.array([r["lng"] for r in seller_zip_rows])
    seller_log_handling = rng.normal(np.log(2.0), 0.45, n_sellers)  # median handling ~2 days, wide spread
    seller_sigma = rng.uniform(0.3, 0.6, n_sellers)
    seller_popularity = rng.pareto(1.2, n_sellers) + 0.2  # long tail of order volume

    # --- products -------------------------------------------------------------------------------
    n_products = max(200, n_orders // 4)
    cat_names = list(CATEGORIES)
    cat_p = np.array([CATEGORIES[c][0] for c in cat_names])
    cat_p /= cat_p.sum()
    prod_cat = rng.choice(cat_names, size=n_products, p=cat_p)
    med_w = np.array([CATEGORIES[c][1] for c in prod_cat])
    sig_w = np.array([CATEGORIES[c][2] for c in prod_cat])
    weight = np.clip(np.exp(rng.normal(np.log(med_w), sig_w)), 50, 40_000).round()
    side = np.cbrt(weight * rng.uniform(1.5, 6.0, n_products))
    med_price = np.array([CATEGORIES[c][3] for c in prod_cat])
    products = pd.DataFrame(
        {
            "product_id": [_hex_id("product", i, seed) for i in range(n_products)],
            "product_category_name": prod_cat,
            "product_name_lenght": rng.integers(20, 60, n_products),
            "product_description_lenght": rng.integers(100, 2000, n_products),
            "product_photos_qty": rng.integers(1, 6, n_products),
            "product_weight_g": weight,
            "product_length_cm": (side * rng.uniform(0.8, 1.5, n_products)).round(),
            "product_height_cm": (side * rng.uniform(0.4, 1.2, n_products)).round(),
            "product_width_cm": (side * rng.uniform(0.8, 1.3, n_products)).round(),
        }
    )
    product_price = np.exp(rng.normal(np.log(med_price), 0.6))
    product_seller = rng.choice(n_sellers, size=n_products, p=seller_popularity / seller_popularity.sum())

    # --- orders ---------------------------------------------------------------------------------
    purchase_ts = _sample_purchase_times(rng, n_orders, start, end)
    prod_idx = rng.choice(n_products, size=n_orders)
    sel_idx = product_seller[prod_idx]
    cust_state = _weighted_choice(rng, CUSTOMER_STATE_W, n_orders)
    cust_rows = []
    for st in np.unique(cust_state):
        m = cust_state == st
        cust_rows.append(
            zips_by_state[st]
            .sample(int(m.sum()), replace=True, random_state=int(rng.integers(1 << 31)))
            .assign(_pos=np.where(m)[0])
        )
    cust_zip = pd.concat(cust_rows).sort_values("_pos").reset_index(drop=True)
    n_items = rng.choice([1, 2, 3, 4], size=n_orders, p=[0.86, 0.09, 0.03, 0.02])
    payment_type = _weighted_choice(rng, PAYMENT_TYPES, n_orders)
    installments = np.where(
        payment_type == "credit_card", rng.choice(np.arange(1, 11), size=n_orders, p=_inst_p()), 1
    )

    # Approval: instant-ish for cards, slow for boleto (bank slip).
    approval_days = np.where(
        payment_type == "boleto",
        np.exp(rng.normal(np.log(1.2), 0.4, n_orders)),
        np.where(
            rng.random(n_orders) < 0.9, rng.exponential(0.08, n_orders), rng.uniform(0.3, 1.2, n_orders)
        ),
    )
    approved_ts = purchase_ts + pd.to_timedelta(approval_days, "D")

    # Handling: seller latent speed, weight, weekend, load and peak-season effects.
    p_ts = to_int_ns(purchase_ts)
    load7 = np.zeros(n_orders)
    for s in np.unique(sel_idx):
        m = np.where(sel_idx == s)[0]
        c7 = rolling_count(p_ts[m], p_ts[m], 7)
        avg7 = max(len(m) / max((pd.Timestamp(end) - pd.Timestamp(start)).days / 7, 1), 0.5)
        load7[m] = c7 / avg7
    weight_total = weight[prod_idx] * n_items
    log_h = (
        seller_log_handling[sel_idx]
        + 0.12 * np.log1p(weight_total / 1000)
        + 0.25 * np.maximum(0, load7 - 1.0)
        + 0.35 * ((approved_ts.dt.month == 11) & (approved_ts.dt.day >= 20)).values
        + rng.normal(0, seller_sigma[sel_idx])
    )
    handling_days = np.exp(log_h)
    weekend = approved_ts.dt.dayofweek.values >= 5
    handling_days += np.where(weekend, rng.uniform(0.5, 1.5, n_orders), 0.0)
    handling_days += np.where(_is_holiday(approved_ts), rng.uniform(0.5, 1.2, n_orders), 0.0)
    handling_days = np.clip(handling_days, 0.05, None)
    carrier_ts = approved_ts + pd.to_timedelta(handling_days, "D")

    # Transit: distance, destination region, weekend hand-over, peak season, heavy tail.
    c_lat, c_lng = cust_zip["lat"].values, cust_zip["lng"].values
    dist = _haversine(seller_lat[sel_idx], seller_lng[sel_idx], c_lat, c_lng)
    cust_region = np.array([BR_REGIONS[s] for s in cust_state])
    base_transit = 1.0 + dist / 300.0 + np.array([REGION_TRANSIT_PENALTY[r] for r in cust_region])
    same_state = cust_state == seller_state[sel_idx]
    base_transit -= 0.5 * same_state
    transit_days = base_transit * np.exp(rng.normal(0, 0.3, n_orders))
    transit_days += np.where(carrier_ts.dt.dayofweek.values >= 4, rng.uniform(0.3, 1.0, n_orders), 0.0)
    peak = ((carrier_ts.dt.month == 11) & (carrier_ts.dt.day >= 20)) | (
        (carrier_ts.dt.month == 12) & (carrier_ts.dt.day <= 20)
    )
    transit_days += np.where(peak.values, rng.uniform(1.0, 2.5, n_orders), 0.0)
    tail = rng.random(n_orders) < 0.04
    transit_days += np.where(tail, rng.exponential(8.0, n_orders), 0.0)
    transit_days = np.clip(transit_days, 0.4, None)
    delivered_ts = carrier_ts + pd.to_timedelta(transit_days, "D")

    # Carrier estimate: a conservative route-level rule that ignores seller speed and load.
    est_days = (
        np.where(payment_type == "boleto", 2.0, 1.0) + 2.5 + base_transit + 2.0 + rng.normal(0, 1.5, n_orders)
    )
    estimated_ts = (purchase_ts + pd.to_timedelta(np.clip(np.round(est_days), 3, None), "D")).dt.normalize()
    shipping_limit_ts = approved_ts + pd.to_timedelta(
        rng.choice([3, 4, 5, 6, 7], size=n_orders, p=[0.1, 0.2, 0.4, 0.2, 0.1]), "D"
    )

    # Order status: a small share never delivered (cancelled / lost).
    status = np.full(n_orders, "delivered", dtype=object)
    u = rng.random(n_orders)
    status[u < 0.012] = "canceled"
    status[(u >= 0.012) & (u < 0.02)] = "shipped"
    status[(u >= 0.02) & (u < 0.022)] = "unavailable"
    # Olist only has delivered orders up to the dump date; keep the last few weeks right-censored.
    status[(status == "delivered") & (delivered_ts > pd.Timestamp(end) + pd.Timedelta(days=20))] = "shipped"
    delivered_mask = status == "delivered"
    carrier_mask = np.isin(status, ["delivered", "shipped"])
    approved_mask = status != "unavailable"

    order_ids = [_hex_id("order", i, seed) for i in range(n_orders)]
    customer_ids = [_hex_id("customer", i, seed) for i in range(n_orders)]
    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_status": status,
            "order_purchase_timestamp": purchase_ts,
            "order_approved_at": approved_ts.where(approved_mask),
            "order_delivered_carrier_date": carrier_ts.where(carrier_mask),
            "order_delivered_customer_date": delivered_ts.where(delivered_mask),
            "order_estimated_delivery_date": estimated_ts,
        }
    )
    for c in orders.columns:
        if c.startswith("order_") and orders[c].dtype.kind == "M":
            orders[c] = orders[c].dt.floor("s")

    # items: `n_items` lines per order, same product & seller (as in most Olist multi-item orders)
    rep = np.repeat(np.arange(n_orders), n_items)
    item_no = np.concatenate([np.arange(1, k + 1) for k in n_items])
    items = pd.DataFrame(
        {
            "order_id": np.array(order_ids)[rep],
            "order_item_id": item_no,
            "product_id": products["product_id"].values[prod_idx[rep]],
            "seller_id": sellers["seller_id"].values[sel_idx[rep]],
            "shipping_limit_date": shipping_limit_ts.dt.floor("s").values[rep],
            "price": np.round(product_price[prod_idx[rep]], 2),
            "freight_value": np.round(
                8 + 0.004 * weight[prod_idx[rep]] + 0.01 * dist[rep] + rng.normal(0, 2, len(rep)), 2
            ).clip(3, None),
        }
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_unique_id": [
                _hex_id("cu", int(i), seed)
                for i in np.where(
                    rng.random(n_orders) < 0.97, np.arange(n_orders), rng.integers(0, n_orders, n_orders)
                )
            ],
            "customer_zip_code_prefix": cust_zip["zip"].values,
            "customer_city": [f"city_{z[:3]}" for z in cust_zip["zip"].values],
            "customer_state": cust_state,
        }
    )

    geo_rows = []
    for _, r in zips.iterrows():
        for _ in range(3):
            geo_rows.append(
                (
                    r["zip"],
                    r["lat"] + rng.normal(0, 0.02),
                    r["lng"] + rng.normal(0, 0.02),
                    f"city_{r['zip'][:3]}",
                    r["state"],
                )
            )
    geolocation = pd.DataFrame(
        geo_rows,
        columns=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ],
    )

    payments = pd.DataFrame(
        {
            "order_id": order_ids,
            "payment_sequential": 1,
            "payment_type": payment_type,
            "payment_installments": installments,
            "payment_value": np.round(product_price[prod_idx] * n_items + 8 + 0.004 * weight_total, 2),
        }
    )

    has_review = delivered_mask & (rng.random(n_orders) < 0.95)
    review_creation = (delivered_ts + pd.Timedelta(days=1)).dt.normalize()
    review_answer = review_creation + pd.to_timedelta(np.exp(rng.normal(np.log(1.5), 0.6, n_orders)), "D")
    late = delivered_ts > estimated_ts
    score = np.where(
        late,
        rng.choice([1, 2, 3, 4, 5], n_orders, p=[0.35, 0.15, 0.15, 0.15, 0.2]),
        rng.choice([1, 2, 3, 4, 5], n_orders, p=[0.06, 0.03, 0.07, 0.19, 0.65]),
    )
    reviews = pd.DataFrame(
        {
            "review_id": [_hex_id("review", i, seed) for i in range(n_orders)],
            "order_id": order_ids,
            "review_score": score,
            "review_comment_title": "",
            "review_comment_message": "",
            "review_creation_date": review_creation.where(has_review),
            "review_answer_timestamp": review_answer.dt.floor("s").where(has_review),
        }
    )[has_review].reset_index(drop=True)

    translation = pd.DataFrame(
        {"product_category_name": cat_names, "product_category_name_english": cat_names}
    )

    tables = {
        "orders": orders, "items": items, "products": products, "sellers": sellers, "customers": customers,
        "geolocation": geolocation, "payments": payments, "reviews": reviews, "category_translation": translation,
    }  # fmt: skip
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in tables.items():
            df.to_csv(out_dir / OLIST_FILES[key], index=False)
    return tables


def _inst_p() -> np.ndarray:
    p = np.array([0.5, 0.12, 0.1, 0.07, 0.06, 0.05, 0.03, 0.03, 0.02, 0.02])
    return p / p.sum()
