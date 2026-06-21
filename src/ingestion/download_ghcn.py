"""
Download NOAA GHCN-Daily yearly CSVs + metadata over HTTPS.

The bucket is public (s3://noaa-ghcn-pds/) and also exposed at
https://noaa-ghcn-pds.s3.amazonaws.com/, so no AWS credentials are required.

Usage:
    python -m src.ingestion.download_ghcn --start-year 2015 --end-year 2024
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from src.ingestion.utils import get_logger, load_config, resolve

log = get_logger("download_ghcn")


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream a URL to disk. Skips if the file already exists with non-zero size."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    return dest


def maybe_gunzip(path: Path) -> Path:
    """If `path` ends with .gz, decompress next to it and return the new path."""
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    with gzip.open(path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def build_year_urls(base_url: str, prefix: str, years: range) -> list[tuple[str, str]]:
    """Year CSVs are published gzipped on the bucket."""
    urls = []
    for y in years:
        fname = f"{y}.csv.gz"
        urls.append((f"{base_url}/{prefix}/{fname}", fname))
    return urls


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=cfg["ingestion"]["default_start_year"])
    ap.add_argument("--end-year", type=int, default=cfg["ingestion"]["default_end_year"])
    ap.add_argument("--workers", type=int, default=cfg["ingestion"]["download_workers"])
    ap.add_argument("--skip-years", action="store_true", help="Only download metadata")
    ap.add_argument("--gunzip", action="store_true",
                    help="Also decompress the yearly .csv.gz files (faster Spark reads, "
                         "but ~6× more disk). Default: keep compressed.")
    args = ap.parse_args()

    raw_dir = resolve(cfg["paths"]["raw"])
    years_dir = raw_dir / "by_year"
    meta_dir = raw_dir / "metadata"
    years_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    base = cfg["source"]["base_url"]

    log.info("Downloading metadata to %s", meta_dir)
    for name in cfg["source"]["metadata_files"]:
        url = f"{base}/{name}"
        dest = meta_dir / name
        download_file(url, dest)
        log.info("  ✓ %s (%.1f KB)", name, dest.stat().st_size / 1024)

    if args.skip_years:
        return

    years = range(args.start_year, args.end_year + 1)
    jobs = build_year_urls(base, cfg["source"]["by_year_prefix"], years)
    log.info("Downloading %d yearly files (%d–%d) with %d workers",
             len(jobs), args.start_year, args.end_year, args.workers)

    def _job(url_name):
        url, name = url_name
        dest = years_dir / name
        download_file(url, dest)
        if args.gunzip:
            maybe_gunzip(dest)
        return name

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="years"):
            name = f.result()
            log.debug("done %s", name)

    log.info("All downloads complete → %s", raw_dir)


if __name__ == "__main__":
    main()
