"""
Silver layer: turn bronze (long format, one row per ELEMENT) into an
ML-friendly wide table (one row per station-day) with units in SI, station
metadata joined in, and a handful of derived columns useful for both EDA
and the downstream Frost-Risk / GDD models.

Bronze schema (long):
    id, date, year, month, element, value (int, in NOAA tenths), flags...

Silver schema (wide, this script's output):
    id, date, year, month, day_of_year,
    tmax_c, tmin_c, tavg_c, temp_range_c,         # in °C
    prcp_mm, snow_mm, snwd_mm,                    # in mm
    is_frost_day, is_heat_stress_day,             # boolean flags
    name, latitude, longitude, elevation_m,       # from stations
    state, country_code

Why this enables ML later
-------------------------
* Wide format is what MLlib's VectorAssembler (and every other modeling lib)
  needs — one row per observation, one column per feature.
* Units are in SI, no more "tenths-of-°C" footguns hiding in the data.
* Station coords come along, so spatial features (nearest-N neighbors,
  altitude effects) are a one-line join away in gold.
* Derived flags (is_frost_day, is_heat_stress_day) are already the *labels*
  the agricultural use case cares about — Frost-Risk classification and
  heat-stress regression can target them directly.

Usage:
    uv run python -m src.processing.build_silver
    uv run python -m src.processing.build_silver --start-year 2022 --end-year 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingestion.utils import build_spark, get_logger, load_config, resolve

log = get_logger("build_silver")

# Elements we keep + their unit conversions to SI.
# Per NOAA's readme: TMAX/TMIN/TAVG are in tenths of °C; PRCP is tenths of mm;
# SNOW/SNWD are already in mm (no division).
ELEMENT_MAP = {
    "TMAX": ("tmax_c",  0.1),
    "TMIN": ("tmin_c",  0.1),
    "PRCP": ("prcp_mm", 0.1),
    "SNOW": ("snow_mm", 1.0),
    "SNWD": ("snwd_mm", 1.0),
}

FROST_THRESHOLD_C = 0.0       # TMIN ≤ 0 °C → frost-risk label
HEAT_STRESS_C     = 30.0      # TMAX ≥ 30 °C → heat-stress label (corn/soy)

# Physically-possible temperature envelope (°C). World records are ~−89.2 / +56.7;
# we use a slightly wider clip as a *defense-in-depth* check on top of NOAA's Q_FLAG.
TEMP_MIN_C, TEMP_MAX_C = -90.0, 60.0


def clean_observations(df: DataFrame) -> DataFrame:
    """Defensive cleaning beyond NOAA's Q_FLAG.

    NOAA already drops Q-flagged rows at ingest, but QC is not perfect. We add
    three cheap, physically-motivated checks. Bad values are set to NULL (kept as
    honest "unknown") rather than dropping the whole station-day, so the other
    valid columns survive and the downstream median Imputer handles the gap.
    """
    # 1) Range check: implausible temperatures → NULL.
    for c in ("tmax_c", "tmin_c"):
        df = df.withColumn(c, F.when(F.col(c).between(TEMP_MIN_C, TEMP_MAX_C), F.col(c)))
    # 2) Cross-field consistency: TMIN must not exceed TMAX. If it does, both are suspect.
    inconsistent = F.col("tmin_c") > F.col("tmax_c")
    df = (df.withColumn("tmin_c", F.when(~inconsistent | inconsistent.isNull(), F.col("tmin_c")))
            .withColumn("tmax_c", F.when(~inconsistent | inconsistent.isNull(), F.col("tmax_c"))))
    # 3) Negative precipitation / snow is impossible → NULL.
    for c in ("prcp_mm", "snow_mm", "snwd_mm"):
        df = df.withColumn(c, F.when(F.col(c) >= 0, F.col(c)))
    return df


def read_bronze_observations(spark: SparkSession, bronze: Path,
                             years: range | None) -> DataFrame:
    df = spark.read.parquet(str(bronze / "observations"))
    if years is not None:
        df = df.filter(F.col("year").between(years.start, years.stop - 1))
    return df.filter(F.col("element").isin(*ELEMENT_MAP.keys()))


def pivot_wide(df: DataFrame) -> DataFrame:
    """Long → wide. groupBy station-day, pivot ELEMENT."""
    return (
        df.groupBy("id", "date", "year", "month")
          .pivot("element", list(ELEMENT_MAP.keys()))
          .agg(F.first("value"))
    )


def apply_units(df: DataFrame) -> DataFrame:
    for element, (out_col, scale) in ELEMENT_MAP.items():
        df = df.withColumn(out_col, F.col(element) * F.lit(scale)).drop(element)
    return df


def add_derived(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("day_of_year", F.dayofyear("date"))
        .withColumn("tavg_c",
                    F.when((F.col("tmax_c").isNotNull()) & (F.col("tmin_c").isNotNull()),
                           (F.col("tmax_c") + F.col("tmin_c")) / 2.0))
        .withColumn("temp_range_c", F.col("tmax_c") - F.col("tmin_c"))
        .withColumn("is_frost_day",
                    F.when(F.col("tmin_c").isNotNull(),
                           F.col("tmin_c") <= F.lit(FROST_THRESHOLD_C)))
        .withColumn("is_heat_stress_day",
                    F.when(F.col("tmax_c").isNotNull(),
                           F.col("tmax_c") >= F.lit(HEAT_STRESS_C)))
    )


def join_stations(obs: DataFrame, stations: DataFrame) -> DataFrame:
    s = (stations
         .withColumnRenamed("elevation", "elevation_m")
         .select("id", "name", "latitude", "longitude", "elevation_m",
                 "state", "country_code"))
    return obs.join(F.broadcast(s), on="id", how="left")


def reorder(df: DataFrame) -> DataFrame:
    return df.select(
        "id", "date", "year", "month", "day_of_year",
        "tmax_c", "tmin_c", "tavg_c", "temp_range_c",
        "prcp_mm", "snow_mm", "snwd_mm",
        "is_frost_day", "is_heat_stress_day",
        "name", "latitude", "longitude", "elevation_m",
        "state", "country_code",
    )


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=None,
                    help="optional lower bound; default: all bronze partitions")
    ap.add_argument("--end-year", type=int, default=None,
                    help="optional upper bound; default: all bronze partitions")
    args = ap.parse_args()

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    bronze = resolve(cfg["paths"]["bronze"])
    silver = resolve(cfg["paths"]["silver"]) / "observations_daily"

    years = None
    if args.start_year is not None and args.end_year is not None:
        years = range(args.start_year, args.end_year + 1)

    stations = spark.read.parquet(str(bronze / "stations"))

    log.info("Reading bronze observations%s",
             f" for {args.start_year}-{args.end_year}" if years else "")
    obs = read_bronze_observations(spark, bronze, years)

    log.info("Pivoting long → wide")
    wide = pivot_wide(obs)
    wide = apply_units(wide)
    wide = clean_observations(wide)   # defensive outlier / cross-field cleaning
    wide = add_derived(wide)
    wide = join_stations(wide, stations)
    wide = reorder(wide)

    log.info("Writing silver → %s", silver)
    (
        wide.repartition("year")
            .write.mode("overwrite")
            .partitionBy("year")
            .parquet(str(silver))
    )

    out = spark.read.parquet(str(silver))
    n = out.count()
    n_stations = out.select("id").distinct().count()
    n_frost = out.filter(F.col("is_frost_day")).count()
    n_heat  = out.filter(F.col("is_heat_stress_day")).count()
    log.info("silver rows           : %s", f"{n:,}")
    log.info("distinct stations     : %s", f"{n_stations:,}")
    log.info("frost-day rows        : %s (%.2f%%)", f"{n_frost:,}", 100 * n_frost / n)
    log.info("heat-stress-day rows  : %s (%.2f%%)", f"{n_heat:,}", 100 * n_heat / n)

    spark.stop()


if __name__ == "__main__":
    main()

