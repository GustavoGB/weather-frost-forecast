"""
Gold layer: ML-ready station-day features for the agricultural use case.

Builds on silver by adding the things a real Frost-Risk / GDD model needs:

* **Lag features** — yesterday and last-week values of TMIN / TMAX / PRCP.
  These are the dominant short-term predictors of next-day frost.
* **Rolling-window stats** — 7-day and 30-day mean / min / max / sum.
  Captures recent climate state without leaking the future.
* **Year-to-date GDD accumulation** for two reference crops
  (corn base 10 °C, wheat base 4 °C) — the primary feature in maturity-date
  and yield-forecast models.
* **Forward labels** — `next_day_is_frost`, `next_day_tmin_c`. Computed via
  Window.lead so the model can train on tomorrow's outcome without leakage.
* **ENSO state** — the monthly ONI (El Niño / La Niña) anomaly + phase, joined by
  (year, month). A basin-scale climate prior that station history alone can't see
  (La Niña winters tend to run colder in the US Corn Belt).

Inputs:  data/silver/observations_daily/  +  data/ref/oni.csv (scripts/process/download_oni.py)
Output:  data/gold/station_daily_features/year=YYYY/

NOTE on lag/window semantics: F.lag(N) returns the row N positions earlier
*within the station's date-ordered series*. If a station has gaps (sensor
offline), "lag 1" can refer to a date more than 1 day in the past. That's
fine for tree-based models (they're robust to it), but worth knowing if
you switch to a strict-time-series model later.

Usage:
    uv run python -m src.processing.build_gold
    uv run python -m src.processing.build_gold --start-year 2022 --end-year 2024
"""
from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.ingestion.utils import build_spark, get_logger, load_config, resolve

log = get_logger("build_gold")

# Reference crop base temperatures for GDD accumulation.
GDD_BASES_C = {"corn": 10.0, "wheat": 4.0}


def add_lag_features(df: DataFrame) -> DataFrame:
    """Yesterday & last-week values within each station's history."""
    w = Window.partitionBy("id").orderBy("date")
    lag_specs = [
        ("tmin_c", [1, 7]),
        ("tmax_c", [1, 7]),
        ("prcp_mm", [1, 7]),
    ]
    for col, lags in lag_specs:
        for k in lags:
            df = df.withColumn(f"{col}_lag_{k}", F.lag(col, k).over(w))
    return df


def add_rolling_features(df: DataFrame) -> DataFrame:
    """Trailing 7d / 30d aggregates per station, ending at the current day."""
    w7  = Window.partitionBy("id").orderBy("date").rowsBetween(-6, 0)
    w30 = Window.partitionBy("id").orderBy("date").rowsBetween(-29, 0)

    df = (
        df
        .withColumn("tmin_c_roll7_mean",  F.avg("tmin_c").over(w7))
        .withColumn("tmin_c_roll7_min",   F.min("tmin_c").over(w7))
        .withColumn("tmax_c_roll7_mean",  F.avg("tmax_c").over(w7))
        .withColumn("tmax_c_roll7_max",   F.max("tmax_c").over(w7))
        .withColumn("prcp_mm_roll7_sum",  F.sum("prcp_mm").over(w7))
        .withColumn("tmin_c_roll30_mean", F.avg("tmin_c").over(w30))
        .withColumn("tmax_c_roll30_mean", F.avg("tmax_c").over(w30))
        .withColumn("prcp_mm_roll30_sum", F.sum("prcp_mm").over(w30))
        .withColumn("frost_days_roll30",
                    F.sum(F.col("is_frost_day").cast("int")).over(w30))
    )
    return df


def add_gdd_accumulation(df: DataFrame) -> DataFrame:
    """Year-to-date growing-degree-day accumulation for reference crops."""
    w_ytd = (
        Window.partitionBy("id", "year")
              .orderBy("date")
              .rowsBetween(Window.unboundedPreceding, 0)
    )
    for crop, base in GDD_BASES_C.items():
        daily = F.greatest(F.lit(0.0), F.col("tavg_c") - F.lit(base))
        df = (
            df
            .withColumn(f"gdd_{crop}_day", daily)
            .withColumn(f"gdd_{crop}_ytd", F.sum(f"gdd_{crop}_day").over(w_ytd))
        )
    return df


def add_regional_features(df: DataFrame, grid_deg: int = 1) -> DataFrame:
    """Neighbour-station ("regional") features — the highest-yield addition flagged
    by the baseline model's weak spatial importances.

    For each day we bucket stations into ~`grid_deg`° lat/lon cells (~111 km at the
    equator) and compute the cell's mean / min TMIN that day, then join it back. This
    is an O(rows) groupBy+join, NOT an O(stations^2) nearest-neighbour search, so it
    scales. The derived `tmin_minus_region` is a microclimate signal: how much colder
    a station runs than its neighbourhood (frost pockets, valleys, altitude).

    No leakage: every column here is built from *today's* observations across space,
    used to predict *tomorrow's* frost. Nothing from the future is referenced.
    """
    df = (df.withColumn("grid_lat", F.floor(F.col("latitude") / grid_deg) * grid_deg)
            .withColumn("grid_lon", F.floor(F.col("longitude") / grid_deg) * grid_deg))
    region = (
        df.where(F.col("tmin_c").isNotNull())
          .groupBy("grid_lat", "grid_lon", "date")
          .agg(F.avg("tmin_c").alias("region_tmin_mean"),
               F.min("tmin_c").alias("region_tmin_min"),
               F.count("*").alias("region_n_stations"))
    )
    df = df.join(region, on=["grid_lat", "grid_lon", "date"], how="left")
    df = df.withColumn("tmin_minus_region", F.col("tmin_c") - F.col("region_tmin_mean"))
    return df.drop("grid_lat", "grid_lon")



def add_forward_labels(df: DataFrame) -> DataFrame:
    """Tomorrow's outcomes — the ML targets. Use Window.lead, NEVER raw joins,
    so we don't accidentally leak features computed up to today."""
    w = Window.partitionBy("id").orderBy("date")
    return (
        df
        .withColumn("next_day_is_frost", F.lead("is_frost_day", 1).over(w))
        .withColumn("next_day_tmin_c",   F.lead("tmin_c", 1).over(w))
        .withColumn("next_day_tmax_c",   F.lead("tmax_c", 1).over(w))
    )


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year",   type=int, default=None)
    ap.add_argument("--min-station-days", type=int, default=30,
                    help="Drop stations with fewer than this many silver rows. "
                         "Stations with too little history give no useful lags.")
    args = ap.parse_args()

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    silver = resolve(cfg["paths"]["silver"]) / "observations_daily"
    gold   = resolve(cfg["paths"]["gold"])   / "station_daily_features"

    log.info("Reading silver")
    df = spark.read.parquet(str(silver))
    if args.start_year and args.end_year:
        df = df.filter(F.col("year").between(args.start_year, args.end_year))

    if args.min_station_days > 0:
        keep = (
            df.groupBy("id")
              .count()
              .filter(F.col("count") >= args.min_station_days)
              .select("id")
        )
        df = df.join(F.broadcast(keep), on="id", how="inner")

    log.info("Computing lag / rolling / GDD / regional / ENSO / forward-label features")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_gdd_accumulation(df)
    df = add_regional_features(df)
    oni_path = resolve("data/ref/oni.csv")
    if oni_path.exists():
        df = add_enso_features(df, spark, oni_path)
    else:
        log.warning("ONI file missing (%s) — run scripts/process/download_oni.py to add ENSO "
                    "features; continuing without them", oni_path)
    df = add_forward_labels(df)

    log.info("Writing gold → %s", gold)
    (
        df.repartition("year")
          .write.mode("overwrite")
          .partitionBy("year")
          .parquet(str(gold))
    )

    out = spark.read.parquet(str(gold))
    n = out.count()
    n_stations = out.select("id").distinct().count()
    n_train = out.filter(F.col("next_day_is_frost").isNotNull()).count()
    pos_rate = (
        out.filter(F.col("next_day_is_frost").isNotNull())
           .agg(F.avg(F.col("next_day_is_frost").cast("double")))
           .first()[0]
    )
    log.info("gold rows              : %s", f"{n:,}")
    log.info("distinct stations      : %s", f"{n_stations:,}")
    log.info("rows usable for ML     : %s (have next_day label)", f"{n_train:,}")
    log.info("frost positive rate    : %.2f%% (base rate for the classifier)",
             100 * (pos_rate or 0))

    spark.stop()


if __name__ == "__main__":
    main()

