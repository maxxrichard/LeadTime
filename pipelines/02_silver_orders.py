"""Silver: one canonical row per order with the eight timestamps, geo and stage targets (Spark-native)."""

from __future__ import annotations

from pyspark.sql import Window
from pyspark.sql import functions as F

from leadtime.data.schema import STAGES, STATE_CENTROIDS
from pipelines.common import get_spark, read_delta, write_delta

TS_FMT = "yyyy-MM-dd HH:mm:ss"


def main(max_total_days: float = 90.0) -> None:
    spark = get_spark("leadtime-silver")
    orders = read_delta(spark, "bronze", "orders")
    items = read_delta(spark, "bronze", "items")
    products = read_delta(spark, "bronze", "products")
    sellers = read_delta(spark, "bronze", "sellers")
    customers = read_delta(spark, "bronze", "customers")

    def ts(c):
        return F.to_timestamp(F.col(c), TS_FMT)

    o = orders.select(
        "order_id", "customer_id", "order_status",
        ts("order_purchase_timestamp").alias("purchase_ts"),
        ts("order_approved_at").alias("approved_ts"),
        ts("order_delivered_carrier_date").alias("carrier_ts"),
        ts("order_delivered_customer_date").alias("delivered_ts"),
        ts("order_estimated_delivery_date").alias("estimated_ts"),
    )  # fmt: skip

    try:
        tr = read_delta(spark, "bronze", "category_translation")
        products = products.join(tr, "product_category_name", "left").withColumn(
            "product_category_name", F.coalesce("product_category_name_english", "product_category_name")
        )
    except Exception:
        pass
    p = products.select(
        "product_id",
        F.col("product_category_name").alias("product_category"),
        F.col("product_weight_g").cast("double"),
        (
            F.col("product_length_cm").cast("double")
            * F.col("product_height_cm").cast("double")
            * F.col("product_width_cm").cast("double")
        ).alias("volume_cm3"),
    )
    w = Window.partitionBy("order_id").orderBy(F.col("order_item_id").cast("int"))
    it = (
        items.join(p, "product_id", "left")
        .withColumn("_rn", F.row_number().over(w))
        .groupBy("order_id")
        .agg(
            F.first(F.when(F.col("_rn") == 1, F.col("seller_id")), ignorenulls=True).alias("seller_id"),
            F.first(F.when(F.col("_rn") == 1, F.col("product_id")), ignorenulls=True).alias("product_id"),
            F.first(F.when(F.col("_rn") == 1, F.col("product_category")), ignorenulls=True).alias(
                "product_category"
            ),
            F.count("*").alias("n_items"),
            F.sum("product_weight_g").alias("total_weight_g"),
            F.sum("volume_cm3").alias("total_volume_cm3"),
            F.sum(F.col("price").cast("double")).alias("total_price"),
            F.sum(F.col("freight_value").cast("double")).alias("total_freight"),
            F.min(ts("shipping_limit_date")).alias("shipping_limit_ts"),
        )
    )
    df = o.join(it, "order_id", "inner")
    df = df.join(
        customers.select(
            "customer_id",
            F.lpad("customer_zip_code_prefix", 5, "0").alias("customer_zip_prefix"),
            "customer_state",
        ),
        "customer_id",
        "left",
    )
    df = df.join(
        sellers.select(
            "seller_id", F.lpad("seller_zip_code_prefix", 5, "0").alias("seller_zip_prefix"), "seller_state"
        ),
        "seller_id",
        "left",
    )

    # geolocation: median lat/lng per zip prefix, falling back to the state centroid
    cent = spark.createDataFrame(
        [(k, v[0], v[1]) for k, v in STATE_CENTROIDS.items()], ["state", "c_lat", "c_lng"]
    )
    try:
        geo = (
            read_delta(spark, "bronze", "geolocation")
            .groupBy(F.lpad("geolocation_zip_code_prefix", 5, "0").alias("zip"))
            .agg(
                F.percentile_approx(F.col("geolocation_lat").cast("double"), 0.5).alias("lat"),
                F.percentile_approx(F.col("geolocation_lng").cast("double"), 0.5).alias("lng"),
            )
        )
    except Exception:
        geo = spark.createDataFrame([], "zip string, lat double, lng double")
    for side in ("customer", "seller"):
        df = (
            df.join(
                geo.select(
                    F.col("zip").alias(f"{side}_zip_prefix"),
                    F.col("lat").alias(f"{side}_lat"),
                    F.col("lng").alias(f"{side}_lng"),
                ),
                f"{side}_zip_prefix",
                "left",
            )
            .join(
                cent.select(
                    F.col("state").alias(f"{side}_state"),
                    F.col("c_lat").alias(f"_{side}_clat"),
                    F.col("c_lng").alias(f"_{side}_clng"),
                ),
                f"{side}_state",
                "left",
            )
            .withColumn(f"{side}_lat", F.coalesce(f"{side}_lat", f"_{side}_clat"))
            .withColumn(f"{side}_lng", F.coalesce(f"{side}_lng", f"_{side}_clng"))
            .drop(f"_{side}_clat", f"_{side}_clng")
        )

    try:
        pay = read_delta(spark, "bronze", "payments")
        wp = Window.partitionBy("order_id").orderBy(F.col("payment_sequential").cast("int"))
        pay = (
            pay.withColumn("_rn", F.row_number().over(wp))
            .groupBy("order_id")
            .agg(
                F.first(F.when(F.col("_rn") == 1, F.col("payment_type"))).alias("payment_type"),
                F.max(F.col("payment_installments").cast("int")).alias("payment_installments"),
            )
        )
        df = df.join(pay, "order_id", "left")
    except Exception:
        df = df.withColumn("payment_type", F.lit("unknown")).withColumn("payment_installments", F.lit(1))
    try:
        rev = read_delta(spark, "bronze", "reviews")
        wr = Window.partitionBy("order_id").orderBy(ts("review_creation_date"))
        rev = rev.withColumn("_rn", F.row_number().over(wr)).filter("_rn = 1").select(
            "order_id", ts("review_creation_date").alias("review_creation_ts"),
            ts("review_answer_timestamp").alias("review_answer_ts"), F.col("review_score").cast("double"),
        )  # fmt: skip
        df = df.join(rev, "order_id", "left")
    except Exception:
        df = (
            df.withColumn("review_creation_ts", F.lit(None).cast("timestamp"))
            .withColumn("review_answer_ts", F.lit(None).cast("timestamp"))
            .withColumn("review_score", F.lit(None).cast("double"))
        )

    df = df.fillna({"product_category": "unknown", "payment_type": "unknown", "payment_installments": 1})

    # stage targets (days) + validity filter mirroring leadtime.features.stages.filter_valid_orders
    for s in STAGES:
        df = df.withColumn(f"{s.name}_days", (F.unix_timestamp(s.end) - F.unix_timestamp(s.start)) / 86400.0)
    df = df.withColumn(
        "carrier_estimate_days",
        (F.unix_timestamp("estimated_ts") - F.unix_timestamp("purchase_ts")) / 86400.0,
    )
    df = df.withColumn(
        "shipping_limit_days",
        (F.unix_timestamp("shipping_limit_ts") - F.unix_timestamp("purchase_ts")) / 86400.0,
    )
    valid = (
        (F.col("order_status") == "delivered")
        & F.col("approved_ts").isNotNull() & F.col("carrier_ts").isNotNull() & F.col("delivered_ts").isNotNull()
        & (F.col("approval_days") >= 0) & (F.col("handling_days") >= 0) & (F.col("transit_days") >= 0)
        & (F.col("total_days") <= max_total_days)
    )  # fmt: skip
    df = df.withColumn("is_valid", valid)
    for s in STAGES:
        df = df.withColumn(f"{s.name}_days", F.greatest(F.col(f"{s.name}_days"), F.lit(1.0 / 1440)))
    out = write_delta(df, "silver", "orders", partition_by=["is_valid"])
    print(f"silver.orders: {df.count():,} rows ({df.filter('is_valid').count():,} valid) -> {out}")


if __name__ == "__main__":
    main()
