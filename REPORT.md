# NOAA GHCN-Daily — End-to-End Big Data Pipeline & Frost-Risk Classifier

**Course:** Big Data
**Author:** Gustavo Gobetti
**Date:** May 2026
**Repository:** `weather-frost-forecast/`
**Runtime:** PySpark 4.1 · Python 3.11 · uv-managed env

---

## 1 · Executive summary

This project builds an end-to-end big-data pipeline on top of NOAA's **Global
Historical Climatology Network — Daily** (GHCN-Daily) dataset and trains a
**Spark ML next-day frost-risk classifier** on top of it. The motivating
business context is an **agricultural company** that needs station-level
historical weather to power Growing-Degree-Day (GDD) yield forecasts and
parametric crop-insurance pricing.

The pipeline ingests **111.0 M raw daily observations** (2005–2024, ~19.7 k
weather stations across the **US Corn Belt**), transforms them into a **wide
ML-ready feature table of 40.3 M rows × 45 columns** (now including
neighbour-station "regional" features), and trains a Spark ML
gradient-boosted-tree next-day frost classifier on a strict time-based split
(train 2005–2023, test 2024) — reaching **ROC-AUC ≈ 0.974** on the held-out
2024 year, with the regional features turning out to be its strongest signal.

The full medallion lake is **~2.0 GB on disk** (Parquet + Snappy), but the
gold feature table alone expands to **~15 GB at an absolute floor and
realistically 20–30 GB** once materialised in a single pandas DataFrame —
and the bronze→silver pivot over 111 M long rows needs more still. On a
16 GB laptop that is an immediate `MemoryError`. This is exactly why the
whole pipeline runs on **PySpark** (which streams from Parquet and spills to
disk) rather than pandas — even with no cluster, on one machine.

---

## 2 · Business problem and motivation

### 2.1 Context

The downstream consumer is an **agricultural company** that operates two
revenue lines requiring station-level historical weather:

1. **Growing-Degree-Day (GDD) and yield forecasting.**
   Crop development tracks accumulated heat, not calendar time
   (`GDD = max(0, (TMAX + TMIN)/2 − base)`). Hybrid recommendations and
   planting-date advice — sold by Corteva, Bayer/Climate Corp,
   FarmersEdge, Syngenta — are priced on this signal. A 3–5 day error in
   predicted maturity can cost 5–15 % of yield per hectare in marginal
   climates.

2. **Frost and extreme-event risk for parametric crop insurance.**
   Last-spring frost / first-fall frost defines the safe planting window.
   The ~$30 B global parametric crop-insurance market prices contracts
   directly off historical frost frequency, consecutive-dry-day counts,
   and heat-stress days (`TMAX > crop-specific threshold`).

### 2.2 Why GHCN-Daily fits

| Property | Benefit for the agricultural use case |
|---|---|
| 120 k+ stations, daily resolution, back to 1763 | Enough history to estimate real 1-in-10 / 1-in-50-year tail risks. |
| Free and re-distributable | No per-call cost, no licensing constraints on derived products. |
| Station-level granularity | Enables neighbor-interpolation models; competitive moat over "nearest-airport METAR" approaches. |
| QC-flagged by NOAA | Engineering time goes into features and models, not into raw sensor de-noising. |

### 2.3 Target deliverable

A regional **Frost-Risk Score** per postal code and planting week, computed
from ≥30 years of nearest-N station histories, served as a Parquet/Delta
table consumed by both the insurance pricing engine and the agronomy app.
This project builds the data foundation plus a working baseline classifier
for that score.

---

## 3 · Dataset

**Source:** NOAA Open Data Dissemination, GHCN-Daily.
URL: <https://www.noaa.gov/nodd/datasets>
Bucket: `s3://noaa-ghcn-pds/` (public, HTTPS-accessible, no AWS account
required).

| Resource | Description | Format |
|---|---|---|
| `csv.gz/by_year/YYYY.csv.gz` | One row per (station, date, ELEMENT). One file per year, gzip-compressed (~170 MB / year). | Headerless CSV |
| `ghcnd-stations.txt` | Station metadata (id, lat, lon, elevation, name). | Fixed-width text |
| `ghcnd-countries.txt` | 2-letter country code → name. | Fixed-width text |
| `ghcnd-states.txt` | 2-letter US/Canada state code → name. | Fixed-width text |
| `ghcnd-inventory.txt` | Per-station × ELEMENT first/last year of record. | Fixed-width text |

### 3.1 ELEMENTs retained

| ELEMENT | Meaning | Source unit | Silver unit |
|---|---|---|---|
| `TMAX` | Daily maximum air temperature | tenths of °C | `tmax_c` (°C) |
| `TMIN` | Daily minimum air temperature | tenths of °C | `tmin_c` (°C) |
| `PRCP` | Daily precipitation | tenths of mm | `prcp_mm` (mm) |
| `SNOW` | Snowfall | mm | `snow_mm` (mm) |
| `SNWD` | Snow depth | mm | `snwd_mm` (mm) |

GHCN contains ~30 ELEMENTs total (including rarely-recorded ones like
evaporation pan and peak wind). Filtering at ingest to the 5 most relevant
agricultural variables cut downstream pivot/window work by ~70 %.

### 3.2 Data quality and missingness

GHCN observations carry three flags:

- **`M_FLAG`** — measurement flag (e.g. `P` = no measurement possible).
- **`Q_FLAG`** — quality-control flag (empty = passed all of NOAA's ~15 QC
  checks: range, internal consistency, spatial coherence, etc.).
- **`S_FLAG`** — source flag identifying which network produced the row.

The ingestion step **drops any row where `Q_FLAG` is non-empty** (≈1–3 % of
raw rows). Missing observations are encoded as *absence of rows* (long
format), not as `NULL` cells. After pivot to wide format, sparsity surfaces
as nulls and is handled by the silver/gold layers and by the median imputer
in the ML pipeline.

### 3.3 Scale chosen for this project

| Property | Value |
|---|---|
| Year range | 2005 – 2024 (20 years) |
| Region | **US Corn Belt** — lat 37–49 °N, lon 80–104 °W (`config.yaml › region`) |
| Download size | ~3.4 GB of `by_year` archives (20 × ~170 MB), filtered to the region at ingest |
| Raw observations ingested (bronze, region-scoped, long) | **111 026 457** rows |
| Distinct stations in region | **≈ 19.7 k** (silver 19 767 → gold 19 678 after the ≥30-day filter) |
| ELEMENT filter | TMAX, TMIN, PRCP, SNOW, SNWD |

This is well past where pandas is workable. The **gold** table is
40 278 633 rows × 45 columns; held in a pandas DataFrame that is
≈ **15 GB at the cheapest 8-byte-per-cell floor** and realistically
**20–30 GB** once the string columns (`id`, `name`, `state`, …) sit in
`object` dtype. The bronze→silver **pivot** over 111 M long rows is worse
still, because pandas must materialise input, output and grouping
intermediates at once. Spark handles the whole corpus on a 16 GB laptop by
reading Parquet columnarly and spilling shuffles to disk (§8.1).

---

## 4 · Architecture

### 4.1 Medallion layout

```
raw            bronze              silver                gold                model
─────          ──────              ──────                ────                ─────
HTTPS    →     Parquet      →     wide,             →   ML features  →    Spark ML
download       partitioned        SI units,             lags, rolling,    Pipeline
(or stream)    by year            station joined        GDD, labels       (GBT/RF)
```

| Layer | Purpose | Format | Partitioning |
|---|---|---|---|
| `raw/` | Untouched bytes from NOAA. Deleted automatically after ingestion in the streaming mode. | CSV.gz / fixed-width text | n/a |
| `bronze/` | Typed Parquet of GHCN observations, **scoped to the region box**, Q-flagged rows dropped. | Parquet + Snappy | `year=YYYY` |
| `silver/` | Long → wide pivot, NOAA tenths → SI, station metadata joined, basic derived columns. | Parquet | `year=YYYY` |
| `gold/` | ML-ready features: lags, rolling stats, GDD accumulation, forward labels. | Parquet | `year=YYYY` |
| `models/` | Persisted `PipelineModel` + metrics JSON. | Spark ML directory | n/a |

### 4.2 Rationale for medallion

Each layer has **one job and one Parquet write**. Failures isolate by layer:
if silver is wrong, you re-run silver — bronze never has to be re-downloaded.
The boundary between *"data is raw and could be junk"* (bronze) and
*"data is clean and meaningful"* (silver) is enforced by **separate Parquet
outputs**, not just function calls.

### 4.3 Tooling stack

| Concern | Choice | Why |
|---|---|---|
| Compute engine | **PySpark 4.1** | Industry-standard distributed engine; same code runs on cluster. |
| ML framework | **Spark ML (DataFrame API)** | `pyspark.ml`, not the deprecated `pyspark.mllib` RDD API. Composes via `Pipeline`. |
| Package manager | **uv** | 10–100× faster than pip; pins Python interpreter; `uv.lock` for reproducible installs. |
| File format | **Parquet + Snappy** | Columnar; predicate pushdown; ~6× smaller than raw CSV; splittable. |
| Partitioning | **By `year`** | Aligns with the natural query pattern (time-range filters) and with NOAA's source layout. |

---

## 5 · Implementation

### 5.1 Project layout

```text
weather-frost-forecast/
├── config/config.yaml        # paths, S3 URL, Spark tunables (driver_memory, shuffle_partitions)
├── pyproject.toml            # uv-managed deps (pyspark, numpy, requests, tqdm, pyyaml)
├── uv.lock                   # locked deps for reproducible installs
├── data/
│   ├── raw/                  # transient, auto-deleted by streaming ingest
│   ├── bronze/               # stations + countries + states + inventory + observations
│   ├── silver/               # observations_daily/
│   ├── gold/                 # station_daily_features/
│   └── models/               # frost_classifier_gbt/ + frost_classifier_gbt_metrics.json
└── src/
    ├── ingestion/
    │   ├── download_ghcn.py          # parallel HTTPS downloader
    │   ├── ingest_stations.py        # fixed-width → Parquet metadata
    │   ├── ingest_observations.py    # bulk CSV → bronze
    │   ├── stream_ingest.py          # disk-light per-year: download → parquet → delete
    │   └── utils.py                  # SparkSession factory + config loader
    ├── processing/
    │   ├── build_silver.py           # long→wide, SI units, station join
    │   └── build_gold.py             # lags, rolling, GDD, forward labels
    └── ml/
        └── train_frost_classifier.py # Spark ML Pipeline (Imputer → Assembler → GBT/RF)
```

### 5.2 Spark configuration

Tuned for a 16 GB laptop. All choices documented in `src/ingestion/utils.py`:

| Setting | Value | Reason |
|---|---|---|
| `spark.master` | `local[8]` | 8 worker threads, **not** `local[*]` — the window-heavy gold build OOMs if every core competes for the 11 GB heap. |
| `spark.driver.memory` | `11g` | ~⅔ of system RAM; needed to fit the long→wide pivot and the per-station window shuffles. |
| `spark.sql.shuffle.partitions` | `1024` | High partition count keeps each shuffle task small — the key lever against gold-build OOM. |
| `spark.sql.adaptive.enabled` | `true` | Dynamically coalesces tiny output partitions; merges skewed joins. |
| `spark.sql.adaptive.coalescePartitions.enabled` | `true` | Cuts a 200-task stage to ~20–30 final files where appropriate. |
| `spark.sql.sources.partitionOverwriteMode` | `dynamic` | Re-running on a year subset only overwrites those partitions. |
| `spark.sql.parquet.compression.codec` | `snappy` | Default; fast read/write tradeoff. |
| `spark.sql.session.timeZone` | `UTC` | Avoids date-skew issues at year-partition boundaries. |

### 5.3 Bronze — ingestion

Two ingestion paths are implemented; both are idempotent and append-only.

**Streaming mode (`stream_ingest.py`)** — preferred for laptops. For each
year:
```
download YYYY.csv.gz  →  read with Spark  →  write Parquet partition  →  delete .csv.gz
```
Peak transient disk stays around 170 MB regardless of how many years are
processed.

**Batch mode (`download_ghcn.py` + `ingest_observations.py`)** — faster
total wall time (Spark starts once instead of per year) but peak disk equals
all year files at once.

Both paths drop rows where `Q_FLAG` is non-empty and write to
`data/bronze/observations/year=YYYY/`.

### 5.4 Silver — wide, SI, station-joined

`src/processing/build_silver.py`. Three transformations:

1. **Pivot ELEMENT → columns.** `groupBy("id","date","year","month").pivot("element").agg(F.first("value"))`.
2. **Unit conversion to SI.** TMAX/TMIN/PRCP are stored as NOAA tenths;
   silver multiplies by 0.1 once so no downstream code has to remember.
3. **Station-metadata join (broadcast).** Lat / lon / elevation_m / state /
   country_code are joined in via a broadcast hash join — stations is small
   (~125 k rows) so it avoids a costly shuffle.

Derived columns added in silver:

```
tavg_c              = (tmax_c + tmin_c) / 2
temp_range_c        = tmax_c − tmin_c
day_of_year         = day-of-year integer
is_frost_day        = tmin_c ≤ 0   ← label seed for the classifier
is_heat_stress_day  = tmax_c ≥ 30  ← label seed for heat-stress model
```

### 5.5 Gold — ML feature table

`src/processing/build_gold.py`. Uses Spark window functions per station,
ordered by date.

| Feature group | Columns | Spark primitive |
|---|---|---|
| Lags | `tmin_c_lag_{1,7}`, `tmax_c_lag_{1,7}`, `prcp_mm_lag_{1,7}` | `F.lag(col, k).over(W.partitionBy("id").orderBy("date"))` |
| Rolling 7d | `tmin_c_roll7_{mean,min}`, `tmax_c_roll7_{mean,max}`, `prcp_mm_roll7_sum` | `F.avg / F.min / F.max / F.sum .over(... rowsBetween(-6, 0))` |
| Rolling 30d | `tmin_c_roll30_mean`, `tmax_c_roll30_mean`, `prcp_mm_roll30_sum`, `frost_days_roll30` | same with `rowsBetween(-29, 0)` |
| GDD | `gdd_corn_day`, `gdd_corn_ytd`, `gdd_wheat_day`, `gdd_wheat_ytd` | year-to-date sum via `rowsBetween(Window.unboundedPreceding, 0)` |
| **Regional (neighbour)** | `region_tmin_mean`, `region_tmin_min`, `region_n_stations`, `tmin_minus_region` | `groupBy(grid_lat, grid_lon, date)` on a 1° grid, then **left join** back; `tmin_minus_region` is the microclimate signal |
| Forward labels | `next_day_is_frost`, `next_day_tmin_c`, `next_day_tmax_c` | `F.lead(col, 1)` over the same window |

The gold table is **45 columns** total. The neighbour-station "regional"
features were the highest-yield addition flagged as future work in earlier
versions of this report — they are now part of the build, and turn out to be
the model's single strongest signal (§7.3).

**Why `F.lead` for labels?** A naïve self-join on `date + 1 day` would risk
leaking tomorrow's *features* into the training set. Computing the label via
`Window.lead` on the same DataFrame guarantees only the label column is
pulled forward.

Stations with fewer than 30 silver rows are dropped (`--min-station-days
30`) since they can't yield meaningful 30-day rolling features.

### 5.6 ML — Spark ML Pipeline

`src/ml/train_frost_classifier.py`. The pipeline is fully declarative:

```python
Pipeline(stages=[
    Imputer(strategy="median",
            inputCols=ALL_FEATURES,
            outputCols=[f"{c}__imp" for c in ALL_FEATURES]),
    VectorAssembler(inputCols=[f"{c}__imp" for c in ALL_FEATURES],
                    outputCol="features",
                    handleInvalid="keep"),
    GBTClassifier(labelCol="label",
                  featuresCol="features",
                  maxIter=40, maxDepth=6,
                  stepSize=0.1, subsamplingRate=0.8,
                  seed=42),
])
```

**Design choices, with reasoning:**

| Choice | Reason |
|---|---|
| **Spark ML (DataFrame API)** | `pyspark.mllib` is deprecated. `pyspark.ml` composes via `Pipeline`, scales unchanged from laptop to cluster. |
| **Time-based split (`year < 2024` train, `year == 2024` test)** | A random split would leak future weather into training. Time-based is the only honest evaluation of a forecast model. |
| **GBTClassifier** | Gradient-boosted trees dominate tabular tasks with mixed continuous + spatial + seasonal features. Beats Random Forest empirically on this data. |
| **`Imputer(strategy="median")`** | Many lag/rolling columns are null for the first 30 days of a station's history; PRCP is null where the sensor doesn't measure rain. Median imputation keeps these rows in training instead of biasing toward long-running stations. |
| **`--sample-fraction`** | Train on a fraction of the train rows to keep laptop runs fast (the reported tuned GBT used `0.1` → 1 319 457 rows). Set to 1.0 on a cluster. |
| **`PipelineModel.save`** | Persists Imputer fits AND the trained classifier as one artifact, so serving code only loads one path. |

---

## 6 · Reproducing the pipeline

```bash
# install
uv sync

# ingest metadata + bronze observations (20 years, 5 elements)
uv run python -m src.ingestion.download_ghcn --skip-years
uv run python -m src.ingestion.ingest_stations
uv run python -m src.ingestion.stream_ingest \
    --start-year 2005 --end-year 2024 \
    --elements TMAX TMIN PRCP SNOW SNWD

# build silver + gold
uv run python -m src.processing.build_silver
uv run python -m src.processing.build_gold

# train and evaluate
uv run python -m src.ml.train_frost_classifier \
    --test-year 2024 --sample-fraction 0.1 --algorithm gbt --tune
```

**End-to-end wall-clock on this laptop:** ~50 minutes.

---

## 7 · Results

### 7.1 Scale achieved

| Layer | Rows | Disk (Parquet + Snappy) |
|---|---|---|
| Bronze observations (20 years × 5 elements, region-scoped, long) | **111 026 457** | 366 MB |
| Silver (wide, daily) | **40 279 661** | 614 MB |
| Gold (ML features, 45 columns) | **40 278 633** | 1.1 GB |
| Models + metrics | — | ~430 KB |
| **Total `data/`** | | **≈ 2.0 GB** |

- **≈ 19.7 k distinct stations** in the region (silver 19 767; gold 19 678
  after the ≥30-day filter).
- **13 805 719** gold rows carry a non-null `next_day_is_frost` label —
  i.e., usable for supervised training.
- **39.54 %** frost positive rate (healthy class balance for a binary
  classifier; no oversampling needed).

### 7.2 Classifier performance (held-out 2024)

The headline model is a **CrossValidator-tuned GBT** over the **30-feature**
set (§5.6), trained on a 0.1 sample of the pre-2024 data (**1 319 457** rows)
and evaluated on the **full 2024 test year** (**615 491** station-days). A
naïve `TMIN ≤ 2 °C` rule is the baseline to beat.

| Metric | Tuned GBT | Baseline (`TMIN ≤ 2 °C`) |
|---|---:|---:|
| **ROC-AUC** | **0.9742** | — |
| PR-AUC | 0.9525 | — |
| Accuracy | 0.9111 | 0.8884 |
| F1 | 0.9114 | 0.8901 |
| Weighted precision | 0.9118 | 0.8969 |
| Weighted recall | 0.9111 | 0.8884 |

**Confusion matrix** (test set: 615 491 station-days from 2024):

|              | predicted frost = 0 | predicted frost = 1 |
|---|---:|---:|
| **actual = 0** | TN = 373 250 | FP = 30 142 |
| **actual = 1** | FN = 24 577 | TP = 187 522 |

Precision = 0.8615, Recall = 0.8842, Specificity = 0.9253 — the tuned GBT
clears the baseline on every metric, and recall (catching real frosts) is
deliberately favoured, since a missed frost costs far more than a false alarm.

### 7.3 Feature importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `region_tmin_mean` (neighbour-station mean TMIN) | 0.4107 |
| 2 | `tmin_c` (today's minimum temperature) | 0.2086 |
| 3 | `region_tmin_min` (coldest neighbour tonight) | 0.1436 |
| 4 | `tavg_c` | 0.0600 |
| 5 | `day_of_year` | 0.0217 |
| 6 | `tmax_c` | 0.0124 |
| 7 | `tmin_minus_region` (microclimate vs. neighbours) | 0.0115 |
| 8 | `longitude` | 0.0088 |

The decisive result: the **regional neighbour features are the model's
strongest signal**. `region_tmin_mean` alone (0.4107) outweighs today's own
`tmin_c` (0.2086), and three of the top seven features are regional. This is
exactly the "add what the neighbourhood is doing" bet that earlier versions
flagged as the highest-yield next move — and it paid off: adding the regional
group was the single largest accuracy gain in the build.

### 7.4 Ablations and skill decay

Two honest stress-tests, both on the held-out 2024 year:

**Precipitation ablation.** Dropping the 5 precipitation features (`prcp_*`)
leaves ROC-AUC unchanged at **0.9743** — precip carries essentially no
next-day-frost signal once temperature features are present, so the model
loses nothing by ignoring it.

| Model | Features | ROC-AUC |
|---|---:|---:|
| Full | 30 | 0.9743 |
| No precip | 25 | 0.9743 |

**Forecast horizon (skill decay).** Re-pointing the label `k` days ahead
shows how far the signal carries:

| Horizon | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| D+1 | 0.9127 | 0.9745 | 0.9130 |
| D+2 | 0.8787 | 0.9534 | 0.8799 |
| D+3 | 0.8722 | ≈ 0.94 | — |

Skill is strong at D+1, still useful at D+2, and decays by D+3 — the expected
shape for a model that leans on persistence and the neighbourhood's recent
state rather than on numerical-weather-prediction inputs.

---

## 8 · Discussion

### 8.1 Where Spark earned its keep — the pandas wall

The data is **2.0 GB on disk** but does not *stay* 2.0 GB. Parquet is
Snappy-compressed and columnar; the moment a step needs rows in memory, it
inflates. Concretely, on this 16 GB laptop:

| Step | What pandas would need in RAM | What Spark does |
|---|---|---|
| Hold the **gold** table (40.3 M × 45) | **≈ 14 GB** floor (8 B/cell), **20–30 GB** real once `id`/`name`/`state` are `object` dtype | streams from Parquet; never materialises the whole frame |
| **Pivot** bronze→silver (111 M long rows) | **~30–40 GB** — input **+** output **+** grouping intermediates live at once | shuffle-aggregation, **spills to disk** (the `1024` shuffle partitions matter here) |
| **Window** features over **19.7 k** station histories | sequential `groupby().apply()` (slow) or per-station vectorisation (memory-heavy) | `Window.partitionBy("id")` runs the histories **in parallel** |

So even with **no cluster**, on a single machine, pandas hits a `MemoryError`
on the gold load alone (15 GB > 16 GB once the OS and Python are accounted
for), and the bronze→silver pivot is hopeless. Spark wins on one laptop
because:

1. Parquet + columnar reads only materialise the needed columns.
2. Partitioning by `year` lets Spark parallelise across 20 disjoint chunks.
3. Wide ops are shuffle-aggregations that **spill to disk** instead of
   demanding everything resident — the `1024` shuffle-partitions tuning is
   what keeps each task small enough to fit.

> The repo also ships a deliberately tiny `data/share/gold_sample.parquet`
> precisely so that *single-node* pandas/sklearn experiments are possible on
> a sample — an explicit acknowledgement that the **full** gold table is not
> a pandas-sized object. *(Note: the sample's data part is currently missing
> on disk — only `_SUCCESS`/`.crc` remain — so regenerate it before use.)*

### 8.2 Limitations of the current model

| Limitation | Impact |
|---|---|
| Fixed-grid regional features | Neighbour features use a 1° grid box, not a learned/inverse-distance interpolation — coarser near grid edges and in sparse areas. |
| `F.lag(1)` is "previous row, not previous day" | Across multi-day station gaps, a "lag-1" can refer to a row >1 day in the past. Tree models are robust to this; strict time-series models would not be. |
| Single global model | A single GBT learns one decision boundary for all ~19.7 k Corn-Belt stations; per-region or per-climate models would likely do better. |
| `--sample-fraction` < 1.0 | Final metrics are computed on the full held-out test year, but training uses only a fraction of train rows (default 0.2). On a cluster, train on 1.0. |
| No NWP / reanalysis inputs | Adding ECMWF / GFS forecast variables would lift the model from "interpolate the past" to "fuse with the future". |

### 8.3 Production considerations

What would change to put this in production:

1. **Object storage instead of local disk** (S3 / GCS). The pipeline already
   writes Parquet — only the path prefix changes.
2. **Delta / Iceberg / Hudi** instead of plain Parquet. Brings ACID
   transactions, time travel, partition evolution.
3. **Orchestration** (Airflow / Prefect / Dagster) for daily refresh after
   NOAA's nightly upload.
4. **Cluster deployment** (EMR, Dataproc, Databricks). `local[*]` becomes
   `yarn` / `k8s` — no code change needed because the SparkSession config
   is the only switch.
5. **Model registry** (MLflow). The current `PipelineModel.save` artifact
   maps directly onto MLflow's `pyfunc` flavor.

---

## 9 · Future work

1. **Spatial features — *done in this build*.** Neighbour-station regional
   features (`region_tmin_mean/min/n_stations`, `tmin_minus_region`, on a 1°
   grid) are now computed in gold (§5.5) and are the model's **strongest
   features** (§7.3). The next spatial step is a *learned* nearest-N
   interpolation (inverse-distance or a small GNN) rather than the fixed grid
   box.
2. **Crop-specific GDD thresholds.** Add base temperatures and kill
   thresholds for soybean, cotton, sorghum — product differentiation for
   the agricultural client.
3. **Hyperparameter search — *done in this build*.** The headline model is a
   `CrossValidator` + `ParamGridBuilder` (`pyspark.ml.tuning`) tuned GBT
   (§7.2). Remaining headroom is a wider grid and per-region tuning.
4. **Switch to regression.** Predict `next_day_tmin_c` directly, then derive
   probabilistic risk via a calibrated regressor. Gives the agronomy app a
   probability surface instead of a hard binary label.
5. **Daily refresh job.** Wrap `stream_ingest` + `build_*` in an Airflow
   DAG that runs nightly against NOAA's most recent upload.
6. **Scale to 30+ years.** The current pipeline scales linearly with no code
   change — `stream_ingest --start-year 1995 --end-year 2024` is the only
   command difference. Expected disk: ~18 GB; expected wall-clock: ~2.5 h
   on this laptop, far less on a cluster.

---

## 10 · Conclusion

This project demonstrates a complete medallion data pipeline on real
public-cloud-scale weather data, ending in a CrossValidator-tuned Spark ML
classifier that delivers strong performance (ROC-AUC ≈ 0.974) for an
agricultural-business use case (next-day frost-risk forecasting).

The most informative result is not the headline metric — it is **where the
signal came from**: the neighbour-station **regional features** are the
model's single strongest input (`region_tmin_mean` importance 0.4107,
ahead of today's own `tmin_c` at 0.2086), while five precipitation features
add essentially nothing (§7.4). This is the kind of insight you can only
surface by *actually building the pipeline end-to-end*, and it confirms
where the next engineering hour goes: **richer spatial feature engineering**
(learned nearest-N interpolation, NWP fusion), not more data volume.

The repo is fully reproducible — `uv sync` + the six commands in §6 take
any reviewer from a clean checkout to the same `data/models/` artifact in
~50 minutes on commodity hardware.

---

## Appendix A · Cleanup

The pipeline is fully re-derivable, so any layer can be deleted to recover
disk:

```bash
rm -rf data/gold                       # ~1.1 GB freed; regen with build_gold
rm -rf data/silver                     # ~614 MB freed; regen with build_silver
rm -rf data/bronze/observations        # ~366 MB freed; regen with stream_ingest
rm -rf data/                           # nuclear; ~2.0 GB freed
rm -rf .venv                           # ~500 MB freed; regen with uv sync
```

## Appendix B · Repository contents

```text
weather-frost-forecast/
├── REPORT.md                   ← this document
├── README.md                   ← run-guide + design rationale
├── pyproject.toml              ← uv-managed dependencies
├── uv.lock                     ← locked deps for reproducible installs
├── config/config.yaml          ← paths, S3 URL, Spark tunables
├── src/
│   ├── ingestion/              ← 5 modules + utils
│   ├── processing/             ← build_silver + build_gold
│   └── ml/                     ← train_frost_classifier
└── data/                       ← raw, bronze, silver, gold, models
```
