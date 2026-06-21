"""
Bronze layer: convert NOAA GHCN-Daily yearly CSVs → partitioned Parquet.

Each yearly CSV (under data/raw/by_year/YYYY.csv) is headerless and shaped:

    ID,DATE,ELEMENT,VALUE,M_FLAG,Q_FLAG,S_FLAG,OBS_TIME
    USW00094728,20200101,TMAX,72,,,W,2400

* ID       — station id (joins to stations.id)
* DATE     — YYYYMMDD
* ELEMENT  — TMAX, TMIN, PRCP, SNOW, SNWD, etc.
* VALUE    — tenths of °C for temps; tenths of mm for PRCP; mm for SNOW/SNWD
* *_FLAG   — measurement / quality / source flags
* OBS_TIME — HHMM, often missing

Output layout:
    data/bronze/observations/year=YYYY/part-*.parquet

Usage:
    python -m src.ingestion.ingest_observations --start-year 2015 --end-year 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.ingestion.utils import build_spark, get_logger, load_config, resolve

log = get_logger("ingest_observations")

SCHEMA = StructType([
    StructField("id",       StringType(),  False),
    StructField("date_raw", StringType(),  False),
    StructField("element",  StringType(),  False),
    StructField("value",    IntegerType(), True),
    StructField("m_flag",   StringType(),  True),
    StructField("q_flag",   StringType(),  True),
    StructField("s_flag",   StringType(),  True),
    StructField("obs_time", StringType(),  True),
])


def resolve_year_inputs(years_dir: Path, years: range) -> list[str]:
    """Prefer plain .csv; fall back to .csv.gz (Spark reads gzip transparently but
    splittability is lost — uncompressed is faster when local disk allows it)."""
    inputs = []
    for y in years:
        csv = years_dir / f"{y}.csv"
        gz  = years_dir / f"{y}.csv.gz"
        if csv.exists():
            inputs.append(str(csv))
        elif gz.exists():
            inputs.append(str(gz))
        else:
            log.warning("missing year file: %d", y)
    return inputs


def read_yearly(spark: SparkSession, paths: list[str]) -> DataFrame:
    return (
        spark.read
        .schema(SCHEMA)
        .option("header", "false")
        .option("mode", "PERMISSIVE")
        .csv(paths)
    )


def transform(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("date", F.to_date("date_raw", "yyyyMMdd"))
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .drop("date_raw")
        .filter(F.col("date").isNotNull())
        .filter((F.col("q_flag").isNull()) | (F.col("q_flag") == ""))  # drop flagged-bad rows
    )


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=cfg["ingestion"]["default_start_year"])
    ap.add_argument("--end-year",   type=int, default=cfg["ingestion"]["default_end_year"])
    ap.add_argument("--elements", nargs="*", default=None,
                    help="optional ELEMENT whitelist, e.g. TMAX TMIN PRCP SNOW SNWD")
    ap.add_argument("--cleanup", action="store_true",
                    help="Delete source CSV(.gz) files after a successful Parquet write")
    args = ap.parse_args()

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    raw = resolve(cfg["paths"]["raw"])
    bronze = resolve(cfg["paths"]["bronze"]) / "observations"
    years = range(args.start_year, args.end_year + 1)

    inputs = resolve_year_inputs(raw / "by_year", years)
    if not inputs:
        raise SystemExit("No yearly files found — run download_ghcn.py first.")

    log.info("Reading %d yearly files (%d–%d)", len(inputs), args.start_year, args.end_year)

    df = transform(read_yearly(spark, inputs))
    if args.elements:
        df = df.filter(F.col("element").isin(*args.elements))
        log.info("Filtered to ELEMENTs: %s", args.elements)

    parts_per_year = cfg["ingestion"]["bronze_partitions_per_year"]
    n_years = args.end_year - args.start_year + 1
    df = df.repartition(parts_per_year * n_years, "year")

    log.info("Writing bronze parquet → %s", bronze)
    (
        df.write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(str(bronze))
    )

    n = spark.read.parquet(str(bronze)).count()
    log.info("Bronze observations written: %s rows", f"{n:,}")
    spark.stop()

    if args.cleanup:
        for path in inputs:
            try:
                Path(path).unlink()
                log.info("removed source %s", path)
            except OSError as e:
                log.warning("could not remove %s: %s", path, e)


if __name__ == "__main__":
    main()

