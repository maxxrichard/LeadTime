"""Gold: model-ready features.

Static families (product, route, calendar) are computed with Spark column expressions on the
executors. Point-in-time history and load features need the full event history of a seller / route,
so they are computed with the *same* pandas functions used offline, applied per group with
``applyInPandas`` (one seller or one route per group). Network-wide load uses a Spark range window.
"""

from __future__ import annotations

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from leadtime.features.build import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    add_calendar_features,
    add_product_features,
    add_route_features,
    dest_state_history_features,
    route_history_features,
    seller_history_features,
)
from pipelines.common import get_spark, read_delta, write_delta

_EVENT_COLS = [
    "order_id",
    "purchase_ts",
    "carrier_ts",
    "delivered_ts",
    "shipping_limit_ts",
    "handling_days",
    "transit_days",
]


def _grouped_udf(fn):
    def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
        out = fn(pdf)
        out.insert(0, "order_id", pdf["order_id"].values)
        return out.reset_index(drop=True)

    return _apply


def _schema_for(fn, sample: pd.DataFrame) -> T.StructType:
    out = _grouped_udf(fn)(sample)
    fields = [T.StructField("order_id", T.StringType())]
    for c in out.columns[1:]:
        fields.append(
            T.StructField(c, T.LongType() if pd.api.types.is_integer_dtype(out[c]) else T.DoubleType())
        )
    return T.StructType(fields)


def main() -> None:
    spark = get_spark("leadtime-gold-features")
    sdf = read_delta(spark, "silver", "orders").filter("is_valid")

    # --- static families via pandas-on-driver-free Spark: use mapInPandas per partition -------------
    static_cols = [c for c in FEATURE_COLUMNS if c not in _HISTORY_COLS]

    def _static(it):
        for pdf in it:
            pdf = add_calendar_features(add_route_features(add_product_features(pdf)))
            for c in CATEGORICAL_COLUMNS:
                pdf[c] = pdf[c].astype(str)
            yield pdf

    sample = sdf.limit(50).toPandas()
    static_sample = next(_static([sample]))
    static_schema = spark.createDataFrame(static_sample).schema
    base = sdf.mapInPandas(_static, schema=static_schema)

    # --- grouped point-in-time features (same code as offline) --------------------------------------
    ev = sdf.select(*_EVENT_COLS, "seller_id", "seller_state", "customer_state", "purchase_ts").withColumn(
        "route_key", F.concat_ws("->", "seller_state", "customer_state")
    )
    ev_sample = ev.limit(50).toPandas()
    seller_feats = ev.groupBy("seller_id").applyInPandas(
        _grouped_udf(seller_history_features), schema=_schema_for(seller_history_features, ev_sample)
    )
    route_feats = ev.groupBy("route_key").applyInPandas(
        _grouped_udf(route_history_features), schema=_schema_for(route_history_features, ev_sample)
    )
    dest_feats = ev.groupBy("customer_state").applyInPandas(
        _grouped_udf(dest_state_history_features), schema=_schema_for(dest_state_history_features, ev_sample)
    )

    # --- network load via range windows (seconds) ---------------------------------------------------
    sec = F.unix_timestamp("purchase_ts")
    w7 = Window.orderBy(sec).rangeBetween(-7 * 86400, -1)
    w1 = Window.orderBy(sec).rangeBetween(-1 * 86400, -1)
    net = (
        sdf.select("order_id", "purchase_ts")
        .withColumn("network_orders_7d", F.count("*").over(w7))
        .withColumn("network_orders_1d", F.count("*").over(w1))
        .drop("purchase_ts")
    )

    gold = (
        base.join(seller_feats, "order_id")
        .join(route_feats, "order_id")
        .join(dest_feats, "order_id")
        .join(net, "order_id")
    )
    missing = [c for c in FEATURE_COLUMNS if c not in gold.columns]
    if missing:
        raise RuntimeError(f"gold.features is missing {missing}")
    gold = gold.withColumn("purchase_date", F.to_date("purchase_ts"))
    out = write_delta(gold, "gold", "features")
    print(
        f"gold.features: {gold.count():,} rows x {len(gold.columns)} cols -> {out}; static={len(static_cols)}"
    )


_HISTORY_COLS = {
    "seller_hist_n", "seller_hist_handling_mean", "seller_hist_n_90d", "seller_hist_handling_mean_90d",
    "seller_hist_handling_mean_30d", "seller_hist_sla_miss_rate_90d", "seller_hist_transit_mean_90d", "seller_tenure_days",
    "route_hist_n", "route_hist_transit_mean", "route_hist_n_90d", "route_hist_transit_mean_90d", "dest_state_hist_transit_mean_90d",
    "seller_orders_1d", "seller_orders_7d", "seller_orders_30d", "seller_backlog", "seller_load_ratio",
    "network_orders_7d", "network_orders_1d",
}  # fmt: skip


if __name__ == "__main__":
    main()
