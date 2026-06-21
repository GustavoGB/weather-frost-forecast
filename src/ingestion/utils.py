from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def resolve(path_str: str) -> Path:
    """Resolve a path from config relative to the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def build_spark(cfg: dict) -> SparkSession:
    s = cfg["spark"]
    return (
        SparkSession.builder
        .appName(s["app_name"])
        .master(s["master"])
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", s["shuffle_partitions"])
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
        # Quiet the console: no per-stage progress bars ("[Stage 9676:===> ...]")
        # cluttering notebook output. Combine with setLogLevel("ERROR").
        .config("spark.ui.showConsoleProgress", "false")
        # Adaptive query execution: lets Spark merge tiny output partitions
        # and split skewed ones at runtime — meaningful win on the wide
        # silver/gold writes when one year is heavier than others.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Dynamic partition overwrite means re-running silver/gold only
        # overwrites the partitions actually being written, rather than
        # nuking the whole table — useful when scaling year-by-year.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )

