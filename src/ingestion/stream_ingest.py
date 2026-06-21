"""
Disk-light streaming ingestion: for each year in the range, do

    download .csv.gz  →  read with Spark  →  write Parquet  →  delete .csv.gz

so peak disk stays around one year's compressed CSV (~250-400 MB) regardless of
how many years you ingest. Slower than the batch path (Spark starts/stops once
per year), but ideal for laptops with tight SSDs.

Usage:
    uv run python -m src.ingestion.stream_ingest --start-year 2015 --end-year 2024
    uv run python -m src.ingestion.stream_ingest --start-year 2020 --end-year 2024 \
        --elements TMAX TMIN PRCP
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingestion.download_ghcn import build_year_urls, download_file
from src.ingestion.ingest_observations import SCHEMA, transform
from src.ingestion.utils import build_spark, get_logger, load_config, resolve

log = get_logger("stream_ingest")


def region_station_ids(spark: SparkSession, bronze: Path, region: dict) -> DataFrame:
    """Station ids inside the configured geographic box (and country).

    Built from the already-ingested station metadata. Lets us keep only the
    region's stations in bronze, so we can go back many years cheaply.
    """
    s = spark.read.parquet(str(bronze / "stations"))
    s = s.filter(F.col("country_code") == region["country"])
    s = s.filter(F.col("latitude").between(region["lat_min"], region["lat_max"]))
    s = s.filter(F.col("longitude").between(region["lon_min"], region["lon_max"]))
    return s.select("id")


def ingest_year(spark: SparkSession, csv_path: Path, bronze: Path,
                elements: list[str] | None,
                region_ids: DataFrame | None = None) -> int:
    df = (
        spark.read
        .schema(SCHEMA)
        .option("header", "false")
        .option("mode", "PERMISSIVE")
        .csv(str(csv_path))
    )
    df = transform(df)
    if elements:
        df = df.filter(F.col("element").isin(*elements))
    if region_ids is not None:
        # Broadcast the small region allow-list and keep only its stations.
        df = df.join(F.broadcast(region_ids), on="id", how="inner")

    df = df.repartition(8, "year")
    (
        df.write
        .mode("append")
        .partitionBy("year")
        .parquet(str(bronze))
    )
    return df.count()


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=cfg["ingestion"]["default_start_year"])
    ap.add_argument("--end-year",   type=int, default=cfg["ingestion"]["default_end_year"])
    ap.add_argument("--elements", nargs="*", default=None,
                    help="optional ELEMENT whitelist, e.g. TMAX TMIN PRCP")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Don't delete the .csv.gz after ingesting (default: delete)")
    ap.add_argument("--region", action="store_true",
                    help="Keep only stations inside the geographic box in config "
                         "(`region:` block, default US Corn Belt). Requires that "
                         "ingest_stations has already run.")
    args = ap.parse_args()

    raw = resolve(cfg["paths"]["raw"]) / "by_year"
    bronze_root = resolve(cfg["paths"]["bronze"])
    bronze = bronze_root / "observations"
    raw.mkdir(parents=True, exist_ok=True)
    bronze.mkdir(parents=True, exist_ok=True)

    base = cfg["source"]["base_url"]
    prefix = cfg["source"]["by_year_prefix"]

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    region_ids = None
    if args.region:
        region = cfg["region"]
        region_ids = region_station_ids(spark, bronze_root, region).cache()
        log.info("Region filter '%s' active: %s stations in box "
                 "(lat %s..%s, lon %s..%s)", region["name"], f"{region_ids.count():,}",
                 region["lat_min"], region["lat_max"], region["lon_min"], region["lon_max"])

    years = range(args.start_year, args.end_year + 1)
    urls = build_year_urls(base, prefix, years)

    total = 0
    for (url, name), year in zip(urls, years):
        dest = raw / name
        log.info("[%d] downloading %s", year, name)
        download_file(url, dest)

        log.info("[%d] ingesting → parquet", year)
        rows = ingest_year(spark, dest, bronze, args.elements, region_ids)
        total += rows
        log.info("[%d] wrote %s rows", year, f"{rows:,}")

        if not args.keep_raw:
            try:
                dest.unlink()
                log.info("[%d] removed %s", year, dest.name)
            except OSError as e:
                log.warning("[%d] could not remove %s: %s", year, dest, e)

    log.info("Done. Total rows written across %d years: %s",
             len(years), f"{total:,}")
    spark.stop()


if __name__ == "__main__":
    main()

