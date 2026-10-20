"""Bronze: load the nine Olist CSVs into Delta tables (schema-on-read, one table per file)."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import functions as F

from leadtime.data.olist import OLIST_FILES
from pipelines.common import RAW_DIR, get_spark, write_delta


def main(raw_dir: str = RAW_DIR) -> None:
    spark = get_spark("leadtime-ingest")
    for key, fname in OLIST_FILES.items():
        path = Path(raw_dir) / fname
        if not path.exists():
            print(f"skip {key}: {path} not found")
            continue
        df = (
            spark.read.option("header", True)
            .option("inferSchema", False)
            .csv(str(path))
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_source_file", F.lit(fname))
        )
        out = write_delta(df, "bronze", key)
        print(f"bronze.{key}: {df.count():,} rows -> {out}")


if __name__ == "__main__":
    main()
