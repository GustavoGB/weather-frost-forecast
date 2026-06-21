"""
Parse NOAA GHCN-Daily metadata (fixed-width text) into Parquet.

Inputs (under data/raw/metadata/):
    ghcnd-stations.txt    one row per station (~125k rows)
    ghcnd-countries.txt   2-char country code → name
    ghcnd-states.txt      2-char US/CA state code → name
    ghcnd-inventory.txt   per-station, per-ELEMENT first/last year of record

Outputs (under data/bronze/):
    stations/    countries/    states/    inventory/

The files are small enough that pandas would handle them, but we use Spark
to keep one consistent runtime and so downstream silver jobs can join in-engine.
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.ingestion.utils import build_spark, get_logger, load_config, resolve

log = get_logger("ingest_stations")


# Fixed-width specs taken from NOAA's `readme.txt`. (start, end) are 1-indexed, inclusive.
STATIONS_FIELDS = [
    ("id",          1, 11,  StringType()),
    ("latitude",   13, 20,  DoubleType()),
    ("longitude",  22, 30,  DoubleType()),
    ("elevation",  32, 37,  DoubleType()),
    ("state",      39, 40,  StringType()),
    ("name",       42, 71,  StringType()),
    ("gsn_flag",   73, 75,  StringType()),
    ("hcn_crn_flag", 77, 79, StringType()),
    ("wmo_id",     81, 85,  StringType()),
]

COUNTRIES_FIELDS = [
    ("code", 1, 2,  StringType()),
    ("name", 4, 50, StringType()),
]

STATES_FIELDS = [
    ("code", 1, 2,  StringType()),
    ("name", 4, 50, StringType()),
]

INVENTORY_FIELDS = [
    ("id",         1, 11,  StringType()),
    ("latitude",  13, 20,  DoubleType()),
    ("longitude", 22, 30,  DoubleType()),
    ("element",   32, 35,  StringType()),
    ("first_year", 37, 40, IntegerType()),
    ("last_year",  42, 45, IntegerType()),
]


def read_fixed_width(spark: SparkSession, path: Path, fields: list) -> DataFrame:
    """Generic fixed-width parser: read as one column `line`, then substring + cast."""
    raw = spark.read.text(str(path)).withColumnRenamed("value", "line")
    cols = []
    for name, start, end, dtype in fields:
        length = end - start + 1
        col = F.trim(F.substring("line", start, length))
        col = F.when(col == "", None).otherwise(col).cast(dtype).alias(name)
        cols.append(col)
    return raw.select(*cols)


def write_parquet(df: DataFrame, dest: Path, name: str) -> None:
    out = dest / name
    log.info("→ writing %s", out)
    df.coalesce(1).write.mode("overwrite").parquet(str(out))


def main() -> None:
    cfg = load_config()
    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    meta_dir = resolve(cfg["paths"]["raw"]) / "metadata"
    bronze = resolve(cfg["paths"]["bronze"])

    stations = read_fixed_width(spark, meta_dir / "ghcnd-stations.txt", STATIONS_FIELDS)
    countries = read_fixed_width(spark, meta_dir / "ghcnd-countries.txt", COUNTRIES_FIELDS)
    states = read_fixed_width(spark, meta_dir / "ghcnd-states.txt", STATES_FIELDS)
    inventory = read_fixed_width(spark, meta_dir / "ghcnd-inventory.txt", INVENTORY_FIELDS)

    # Country prefix is the first 2 chars of the station ID — surface it for joins.
    stations = stations.withColumn("country_code", F.substring("id", 1, 2))

    log.info("stations=%d  countries=%d  states=%d  inventory=%d",
             stations.count(), countries.count(), states.count(), inventory.count())

    write_parquet(stations,  bronze, "stations")
    write_parquet(countries, bronze, "countries")
    write_parquet(states,    bronze, "states")
    write_parquet(inventory, bronze, "inventory")

    spark.stop()
    log.info("Done.")


if __name__ == "__main__":
    main()

