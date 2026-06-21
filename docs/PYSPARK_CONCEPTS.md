# PySpark Concepts — Reference & Examples

A complete tour of every PySpark concept used across this work, with runnable
examples. Examples are drawn from the actual notebooks and the weather pipeline:

| Source | What it demonstrates |
|---|---|
| `rdd_intro.ipynb`, `PADSONL08_AULA04.ipynb` | RDD fundamentals, transformations/actions |
| `PADSONL08_Aula04_Supermercado_GGB.ipynb` | RDD practical data parsing |
| `aula09_SparkML_GGB.ipynb` | Spark ML: regression, pipelines, tuning |
| `Aula10/nyc_taxi_ml_GGB.ipynb` | Schemas, feature engineering, scaling |
| `pads-big-data-weather/` | Production: windows, joins, medallion, classification |

---

## Table of contents

1. [What Spark is](#1-what-spark-is)
2. [SparkSession & SparkContext](#2-sparksession--sparkcontext)
3. [RDDs — the low-level model](#3-rdds--the-low-level-model)
4. [Transformations vs Actions (lazy evaluation)](#4-transformations-vs-actions-lazy-evaluation)
5. [Pair RDDs & word count](#5-pair-rdds--word-count)
6. [RDD practical parsing](#6-rdd-practical-parsing)
7. [DataFrames](#7-dataframes)
8. [Schemas: inferSchema vs explicit StructType](#8-schemas-inferschema-vs-explicit-structtype)
9. [Column expressions & functions](#9-column-expressions--functions)
10. [Aggregations & groupBy](#10-aggregations--groupby)
11. [Spark SQL](#11-spark-sql)
12. [Joins (including broadcast)](#12-joins-including-broadcast)
13. [Pivot (long → wide)](#13-pivot-long--wide)
14. [Window functions](#14-window-functions)
15. [Reading & writing data (Parquet, partitioning)](#15-reading--writing-data-parquet-partitioning)
16. [Spark ML — feature engineering](#16-spark-ml--feature-engineering)
17. [Spark ML — models](#17-spark-ml--models)
18. [Pipelines](#18-pipelines)
19. [Evaluation](#19-evaluation)
20. [Hyperparameter tuning & cross-validation](#20-hyperparameter-tuning--cross-validation)
21. [Saving & loading models](#21-saving--loading-models)
22. [Performance: partitioning, caching, AQE](#22-performance-partitioning-caching-aqe)
23. [Spark ↔ pandas interop](#23-spark--pandas-interop)
24. [Classic example: estimating π](#24-classic-example-estimating-π)

---

## 1. What Spark is

Apache Spark is a **distributed computing engine**: instead of processing data on
one machine, it splits the work across many cores/machines and runs tasks in
parallel. **PySpark** is its Python API.

Two layers you'll use:

- **RDD** — the low-level "resilient distributed dataset". Functional API (`map`,
  `filter`, `reduce`). Maximum control, no automatic optimization.
- **DataFrame / Spark SQL** — a typed, table-like API. Goes through the
  **Catalyst optimizer**, so it's faster and more concise. This is what you use
  for almost everything in practice.

---

## 2. SparkSession & SparkContext

`SparkSession` is the single entry point for DataFrames, SQL, and RDDs.

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")          # run locally, one thread per CPU core
    .appName("my-app")
    .config("spark.sql.shuffle.partitions", "4")  # keep shuffles light locally
    .getOrCreate()
)

print(spark.version)
sc = spark.sparkContext          # the SparkContext: where RDDs live
```

`master` controls *where* it runs:

| Value | Meaning |
|---|---|
| `local[*]` | Local, one thread per core |
| `local[2]` | Local, exactly 2 threads |
| `yarn` / `spark://host:port` | A real cluster |

**Production config** (from `pads-big-data-weather/src/ingestion/utils.py`) — the
same builder, with tunables that matter at scale:

```python
spark = (
    SparkSession.builder
    .appName("weather")
    .master("local[8]")                              # 8 worker threads
    .config("spark.driver.memory", "10g")
    .config("spark.sql.shuffle.partitions", 1024)    # post-shuffle partition count
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.sql.adaptive.enabled", "true")    # Adaptive Query Execution
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")               # quiet the logs

# ... do work ...
spark.stop()                                         # always release resources
```

---

## 3. RDDs — the low-level model

**RDD = Resilient Distributed Dataset**: an immutable, distributed collection of
objects spread across **partitions**.

```
RDD[T]
  Partition 0 → [item, item, item]   ← Worker 1
  Partition 1 → [item, item, item]   ← Worker 2
  Partition 2 → [item, item, item]   ← Worker 3
```

- **Resilient** — if a partition is lost, Spark recomputes it from its *lineage*.
- **Distributed** — data lives across many threads/nodes.
- **Dataset** — any Python objects (tuples, dicts, strings…).

Create one from a Python list or a file:

```python
sc = spark.sparkContext

rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8])   # from memory
print(rdd.getNumPartitions())                     # how many partitions
print(rdd.collect())                              # pull all elements to the driver

lines = sc.textFile("supermercado.csv")           # from a file (one element per line)
```

---

## 4. Transformations vs Actions (lazy evaluation)

This is the single most important mental model in Spark.

| | Transformations | Actions |
|---|---|---|
| What | Define a *new* RDD/DataFrame | Trigger computation, return a result |
| When it runs | **Lazy** — nothing happens yet | **Eager** — runs the whole plan |
| Examples | `map`, `filter`, `flatMap`, `select`, `withColumn` | `collect`, `count`, `reduce`, `take`, `show`, `write` |

Spark builds a **DAG** (directed acyclic graph) of transformations and only
executes it when you call an action.

```python
rdd = sc.parallelize([1, 2, 3, 4, 5, 6])

# Transformations — no work done yet, just a plan
evens   = rdd.filter(lambda x: x % 2 == 0)
doubled = evens.map(lambda x: x * 2)

# Action — NOW Spark runs the full plan
print(doubled.collect())   # [4, 8, 12]
```

Common transformations and actions:

```python
rdd = sc.parallelize([1, 2, 3, 4, 5])
rdd.map(lambda x: x * 2).collect()       # [2, 4, 6, 8, 10]
rdd.filter(lambda x: x > 3).collect()    # [4, 5]
rdd.flatMap(lambda x: [x, x * 10]).collect()  # [1,10,2,20,...] (flattens nested)

rdd = sc.parallelize([10, 20, 30, 40, 50])
rdd.count()                              # 5
rdd.first()                              # 10
rdd.take(3)                              # [10, 20, 30]
rdd.reduce(lambda a, b: a + b)           # 150
rdd.takeOrdered(2, key=lambda x: -x)     # [50, 40]  (top-2 by descending value)
```

---

## 5. Pair RDDs & word count

A **pair RDD** is an RDD of `(key, value)` tuples. It unlocks "by-key" operations
like `reduceByKey`, `groupByKey`, `mapValues`.

```python
rdd = sc.parallelize([1, 2, 3, 4, 5, 6])
pairs = rdd.map(lambda x: (x % 2, x))            # key = 0 (even) or 1 (odd)
pairs.reduceByKey(lambda a, b: a + b).collect()  # [(0, 12), (1, 9)]
```

The classic **word count** — the "Hello World" of Spark:

```python
text = sc.parallelize(["the cat sat on the mat", "the cat is on the mat"])

word_counts = (
    text
    .flatMap(lambda line: line.split(" "))   # lines → individual words
    .map(lambda word: (word, 1))             # (word, 1) pairs
    .reduceByKey(lambda a, b: a + b)         # sum counts per word
    .sortBy(lambda x: x[1], ascending=False) # most frequent first
)
for word, count in word_counts.collect():
    print(f"{word:10s}: {count}")
```

> **`reduceByKey` vs `groupByKey`:** prefer `reduceByKey`. It combines values
> *locally on each partition first* (a "map-side combine") before the shuffle, so
> far less data moves across the network. `groupByKey` shuffles everything.

The **MovieLens** patterns (from `PADSONL08_Aula04`) are all variations of this:

```python
ratings = sc.textFile("ratings.csv")  # userId,movieId,rating,timestamp

# Average rating per movie: (movieId, (sum, count)) → divide
avg_per_movie = (
    ratings
    .map(lambda line: line.split(","))
    .map(lambda c: (c[1], (float(c[2]), 1)))                  # (movie, (rating, 1))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))     # sum ratings & counts
    .mapValues(lambda sc: sc[0] / sc[1])                      # mean = sum / count
)
```

---

## 6. RDD practical parsing

From the **supermercado** exercise — turning a raw CSV into typed records with the
RDD API (this is the work a DataFrame's `read.csv` does for you, done manually):

```python
rdd1 = sc.textFile("supermercado.csv")
header = rdd1.take(1)[0]                         # 'PRODUTO;QUANTIDADE;PRECO UNIT. (R$)'

rdd2 = rdd1.filter(lambda x: x != header)        # 1) drop the header row
rdd3 = rdd2.map(lambda x: x.split(";"))          # 2) split on ';'
rdd4 = rdd3.map(lambda x: [x[0], float(x[1]), float(x[2])])  # 3) cast to floats

# Total spend = sum of qty * unit_price
total = rdd4.map(lambda x: x[1] * x[2]).sum()    # 798.85

# Most expensive product by unit price
priciest = rdd4.map(lambda x: (x[0], x[2])).takeOrdered(1, key=lambda x: -x[1])
```

---

## 7. DataFrames

A DataFrame is a distributed table with named, typed columns — like a pandas
DataFrame but lazy and distributed. The everyday API:

```python
data = [("Alice", "Engineering", 95000),
        ("Bob",   "Marketing",   72000),
        ("Carol", "Engineering", 105000)]
df = spark.createDataFrame(data, ["name", "department", "salary"])

df.show(5)            # print first 5 rows (an ACTION)
df.printSchema()      # column names + types
df.describe().show()  # count/mean/stddev/min/max per numeric column

df.select("name", "salary")               # pick columns
df.filter(df.salary > 80000)              # keep rows (alias: .where)
df.drop("department")                     # remove a column
df.withColumn("bonus", df.salary * 0.1)   # add/replace a column
df.withColumnRenamed("salary", "pay")     # rename
df.orderBy("salary", ascending=False)     # sort
df.limit(10)                              # first N rows (a TRANSFORMATION)
```

Reading a CSV (from `aula09`):

```python
data = spark.read.csv("Advertising.csv", header=True, inferSchema=True)
data = data.drop("_c0")     # drop the unnamed index column
data.show(5)
```

### RDD vs DataFrame — when to use which

| | RDD | DataFrame |
|---|---|---|
| Data shape | Any Python object | Typed columns (a table) |
| Optimization | Manual | Automatic (Catalyst) |
| API style | Functional (`map`/`filter`) | SQL-like (`select`/`where`/`groupBy`) |
| Use for | Unstructured data, custom logic | Structured data — **the default** |

---

## 8. Schemas: inferSchema vs explicit StructType

`inferSchema=True` makes Spark scan the data to guess types — convenient, but it
costs an extra pass and can guess wrong. For production, **declare the schema**.

```python
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType, TimestampType,
)

# Explicit schema (from the NYC taxi notebook)
labels = [
    ("VendorID",        StringType()),
    ("passenger_count", FloatType()),
    ("trip_distance",   FloatType()),
    ("fare_amount",     FloatType()),
    ("pickup_datetime", TimestampType()),
]
schema = StructType([StructField(name, dtype, True) for name, dtype in labels])
#                                              ↑ nullable = True

df = spark.read.csv("nyc-data", header=True, schema=schema)
```

The weather pipeline declares its bronze schema the same way
(`ingest_observations.py`):

```python
SCHEMA = StructType([
    StructField("id",      StringType(),  False),   # nullable=False: required
    StructField("date_raw", StringType(), False),
    StructField("element", StringType(),  False),
    StructField("value",   IntegerType(), True),
    StructField("q_flag",  StringType(),  True),
])
df = spark.read.schema(SCHEMA).option("header", "false").csv(paths)
```

---

## 9. Column expressions & functions

`pyspark.sql.functions` (conventionally imported as `F`) is the toolbox for
building column expressions. Everything here is a **transformation**.

```python
from pyspark.sql import functions as F

df.select(F.col("salary"))                  # reference a column
F.lit(0.0)                                   # a literal/constant column
F.col("salary").cast("double")              # change a column's type
F.col("value").between(-90, 60)             # range predicate
F.col("element").isin("TMAX", "TMIN")       # membership
F.col("q_flag").isNull()                     # null test
```

**Conditional logic** with `when` / `otherwise` (used heavily for cleaning):

```python
# Implausible temperatures → NULL (kept as honest "unknown")
df = df.withColumn(
    "tmax_c",
    F.when(F.col("tmax_c").between(-90, 60), F.col("tmax_c"))  # else → NULL
)

# A boolean label
df = df.withColumn("is_frost_day", F.col("tmin_c") <= F.lit(0.0))
```

**Date/time functions** (feature engineering, from NYC taxi & weather):

```python
df = (df
    .withColumn("date",        F.to_date("date_raw", "yyyyMMdd"))  # parse string→date
    .withColumn("year",        F.year("date"))
    .withColumn("month",       F.month("date"))
    .withColumn("day_of_year", F.dayofyear("date"))
    .withColumn("hour",        F.hour("pickup_datetime"))
    .withColumn("day_of_week", F.dayofweek("pickup_datetime")))
```

**Math / row-wise helpers** used in the gold layer:

```python
F.greatest(F.lit(0.0), F.col("tavg_c") - F.lit(10.0))  # element-wise max
F.floor(F.col("latitude") / 1) * 1                     # bucket into 1° grid cells
```

---

## 10. Aggregations & groupBy

```python
from pyspark.sql import functions as F

df.groupBy("department").agg(
    F.count("*").alias("headcount"),
    F.avg("salary").alias("avg_salary"),
    F.max("salary").alias("max_salary"),
    F.min("salary").alias("min_salary"),
    F.sum("salary").alias("total"),
).orderBy("department").show()
```

Counting per group and filtering the result (from `build_gold`, dropping stations
with too little history):

```python
keep = (df.groupBy("id")
          .count()
          .filter(F.col("count") >= 30)
          .select("id"))
```

---

## 11. Spark SQL

Register a DataFrame as a temp view and query it with SQL — fully
interchangeable with the DataFrame API (both go through Catalyst).

```python
df.createOrReplaceTempView("employees")

spark.sql("""
    SELECT department,
           ROUND(AVG(salary), 2) AS avg_salary
    FROM   employees
    GROUP  BY department
    ORDER  BY avg_salary DESC
""").show()
```

---

## 12. Joins (including broadcast)

Standard join: `df.join(other, on="key", how="left")`. `how` ∈
`inner` / `left` / `right` / `outer`.

```python
obs.join(stations, on="id", how="left")
```

**Broadcast join** — when one side is small, ship it to every executor so the big
side never shuffles. This is the key optimization in the weather pipeline:

```python
# Join 125k station rows onto millions of observations, no shuffle of the big side
result = obs.join(F.broadcast(stations), on="id", how="left")

# Join monthly climate index (≈12 rows/year) onto every station-day
df = df.join(F.broadcast(oni), on=["year", "month"], how="left")

# Use a small allow-list to filter a huge table (semi-join via inner broadcast)
df = df.join(F.broadcast(region_ids), on="id", how="inner")
```

> Rule of thumb: if one side fits comfortably in memory (a few hundred MB or
> less), `F.broadcast()` it. Spark also auto-broadcasts below
> `spark.sql.autoBroadcastJoinThreshold`.

---

## 13. Pivot (long → wide)

Reshape "one row per (entity, attribute)" into "one row per entity, one column per
attribute". The weather pipeline uses it to turn NOAA's long format (one row per
weather element) into an ML-friendly wide table (`build_silver.py`):

```python
# Long:  id | date | element | value          (TMAX/TMIN/PRCP as rows)
# Wide:  id | date | TMAX | TMIN | PRCP        (elements as columns)
wide = (
    df.groupBy("id", "date", "year", "month")
      .pivot("element", ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"])  # list = faster
      .agg(F.first("value"))
)
```

> Passing the explicit value list to `pivot()` avoids an extra scan Spark would
> otherwise need to discover the distinct values.

---

## 14. Window functions

Window functions compute across a set of rows *related to the current row* without
collapsing them (unlike `groupBy`). Essential for time-series features. A window =
`partitionBy` (the groups) + `orderBy` (the ordering within each group) + an
optional **frame** (`rowsBetween`).

```python
from pyspark.sql import Window
from pyspark.sql import functions as F
```

**Lag / lead** — previous and future rows (from `build_gold.py`):

```python
w = Window.partitionBy("id").orderBy("date")

# Look BACKWARD — yesterday's & last-week's value (predictive features)
df = (df.withColumn("tmin_lag_1", F.lag("tmin_c", 1).over(w))
        .withColumn("tmin_lag_7", F.lag("tmin_c", 7).over(w)))

# Look FORWARD — tomorrow's outcome (the ML label). Use lead, never a self-join,
# so you can't accidentally leak features into the target.
df = df.withColumn("next_day_is_frost", F.lead("is_frost_day", 1).over(w))
```

**Rolling windows** — trailing aggregates via a row frame. `rowsBetween(-6, 0)` =
"this row and the 6 before it" = a 7-day trailing window:

```python
w7  = Window.partitionBy("id").orderBy("date").rowsBetween(-6, 0)   # 7-day
w30 = Window.partitionBy("id").orderBy("date").rowsBetween(-29, 0)  # 30-day

df = (df
    .withColumn("tmin_roll7_mean", F.avg("tmin_c").over(w7))
    .withColumn("tmax_roll7_max",  F.max("tmax_c").over(w7))
    .withColumn("prcp_roll7_sum",  F.sum("prcp_mm").over(w7))
    .withColumn("tmin_roll30_mean", F.avg("tmin_c").over(w30)))
```

**Cumulative / year-to-date** — frame from the start of the partition to now:

```python
w_ytd = (Window.partitionBy("id", "year").orderBy("date")
               .rowsBetween(Window.unboundedPreceding, 0))   # all rows up to today

df = df.withColumn("gdd_corn_ytd", F.sum("gdd_corn_day").over(w_ytd))
```

> **Leakage warning:** a *backward* frame (`rowsBetween(-N, 0)`) or `lag` only sees
> the past — safe for features. `lead` and forward frames see the future — only
> use them to build **labels**, never features.

---

## 15. Reading & writing data (Parquet, partitioning)

**Parquet** is the default storage format: columnar, compressed, schema-carrying,
splittable. Writing with `partitionBy` lays the data out by a column so later reads
can skip irrelevant files ("partition pruning").

```python
# Write — partitioned by year, snappy-compressed
(df.repartition("year")          # one in-memory partition per year before write
   .write.mode("overwrite")      # overwrite | append | ignore | error
   .partitionBy("year")          # on-disk layout: .../year=2023/part-*.parquet
   .parquet("data/silver/observations_daily"))

# Read back — reading a single year only touches that folder
out = spark.read.parquet("data/silver/observations_daily")
out_2023 = out.filter(F.col("year") == 2023)   # partition pruning kicks in
```

**`repartition` vs `coalesce`:**

```python
df.repartition(1024, "year")  # full shuffle; can increase or decrease partitions
df.coalesce(1)                # no shuffle; only DECREASES (used to write 1 file)
```

`ingest_stations.py` uses `coalesce(1)` to write each small lookup table as a
single file; the big observation tables use `repartition(...)` + `partitionBy`.

CSV read options seen across the project:

```python
spark.read.csv(path, header=True, inferSchema=True)        # quick & dirty
spark.read.schema(SCHEMA).option("header", "false") \
     .option("mode", "PERMISSIVE").csv(path)               # production
spark.read.text(path)   # one column "value" per line — used for fixed-width parsing
```

---

## 16. Spark ML — feature engineering

Spark ML (the **DataFrame-based** API, `pyspark.ml` — not legacy `pyspark.mllib`)
follows a scikit-learn-like `fit`/`transform` design. Every model needs a single
column called **`features`** holding a `Vector`. These transformers build it.

### VectorAssembler — combine columns into one feature vector

```python
from pyspark.ml.feature import VectorAssembler

vec = VectorAssembler(
    inputCols=["TV", "Radio", "Newspaper"],
    outputCol="features",
)
train_vec = vec.transform(train)   # adds a `features` vector column
```

### StringIndexer + OneHotEncoder — encode categoricals

`StringIndexer` maps string labels → numeric indices; `OneHotEncoder` then expands
those indices into sparse one-hot vectors. (One-hot needs numbers, hence the
two-step.)

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder

cat_cols = ["sex", "smoker", "region"]
idx_cols = [c + "Index" for c in cat_cols]
ohe_cols = [c + "OHE"   for c in cat_cols]

indexer = StringIndexer(
    inputCols=cat_cols, outputCols=idx_cols,
    handleInvalid="keep",   # unseen category at predict time → its own bucket
)
encoder = OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols)
```

> `handleInvalid`: `"keep"` (extra bucket for unknowns), `"skip"` (drop the row),
> or `"error"` (fail). Important when a category appears in test but not train.

> **Sparse vs dense vectors:** one-hot output is a `SparseVector` — it stores only
> the non-zero positions, e.g. `SparseVector(10, [3, 5], [7, 2])` is a length-10
> vector that's 7 at index 3 and 2 at index 5, zero elsewhere. Memory-efficient
> for wide categorical encodings.

### StandardScaler — normalize numeric features

Linear/distance-based models are scale-sensitive; trees are not. Scale *before*
fitting a linear model.

```python
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="features", outputCol="features_scaled",
    withMean=False, withStd=True,
)
```

### Imputer — fill missing values

Lag/rolling columns are null for the first N days of each series; `Imputer` fills
them with the column median (better than dropping rows, which biases the sample).

```python
from pyspark.ml.feature import Imputer

imputer = Imputer(
    strategy="median",                       # or "mean"
    inputCols=ALL_FEATURES,
    outputCols=[f"{c}__imp" for c in ALL_FEATURES],
)
```

---

## 17. Spark ML — models

All estimators share `.fit(train)` → a model, then `model.transform(test)` →
predictions. Import from `pyspark.ml.regression` or `pyspark.ml.classification`.

### Linear Regression

```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="Sales")
#                     ↑ NOTE: it's featuresCol (plural), not featureCol
model = lr.fit(train_vec)

print(model.coefficients, model.intercept)   # fitted parameters
result = model.evaluate(test_vec)             # built-in metrics object
```

The NYC-taxi linear model shows the regularization knobs:

```python
LinearRegression(
    labelCol="fare_amount", featuresCol="features",
    maxIter=50,             # max optimizer iterations
    regParam=0.02,          # regularization strength
    elasticNetParam=0.2,    # 0 = Ridge (L2), 1 = Lasso (L1), 0.2 = 20% L1 / 80% L2
    solver="normal",        # closed-form normal equations
)
```

### Random Forest (regression & classification)

```python
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    labelCol="label", featuresCol="features",
    numTrees=80, maxDepth=10,
    subsamplingRate=0.7,
    featureSubsetStrategy="sqrt",
)
```

### Gradient-Boosted Trees (the weather classifier's choice)

GBT usually wins on tabular data with mixed continuous/categorical features and
non-linear interactions:

```python
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    labelCol="label", featuresCol="features",
    maxIter=40,            # number of boosting rounds (trees)
    maxDepth=6,
    stepSize=0.1,          # learning rate
    subsamplingRate=0.8,
    seed=42,
)
```

### Logistic Regression (linear classification baseline)

```python
from pyspark.ml.classification import LogisticRegression

clf = LogisticRegression(labelCol="label", featuresCol="features_scaled",
                         maxIter=50, regParam=0.0)
```

### Feature importances (tree models)

```python
classifier = model.stages[-1]                       # last stage of a fitted Pipeline
importances = classifier.featureImportances.toArray()
ranked = sorted(zip(ALL_FEATURES, importances),
                key=lambda p: p[1], reverse=True)
```

---

## 18. Pipelines

A `Pipeline` chains transformers + an estimator into one object. Calling `.fit()`
runs every stage in order; the resulting `PipelineModel.transform()` applies them
all. This guarantees train and test get **identical** preprocessing — no leakage,
no skew.

Simple two-stage pipeline (from `aula09`):

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vec, lr])   # assemble features → fit regression
model = pipeline.fit(train)             # fits every stage
pred = model.transform(test)            # applies every stage
pred.select("Sales", "prediction").show(5)
```

Full preprocessing + model pipeline (the health-cost model):

```python
pipeline = Pipeline(stages=[
    stringIndexer,   # encode categoricals → indices
    oheEncoder,      # indices → one-hot
    vecAssembler,    # all features → one vector
    lr,              # the model
])
model = pipeline.fit(train)
pred = model.transform(test)
```

The frost classifier composes `Imputer → VectorAssembler → (StandardScaler) →
Classifier` the same way — the exact same code runs on a laptop or a cluster.

---

## 19. Evaluation

Evaluators live in `pyspark.ml.evaluation` and take a predictions DataFrame.

**Regression:**

```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="charges", predictionCol="prediction", metricName="rmse",
)  # metricName: rmse | mse | mae | r2
rmse = evaluator.evaluate(pred)
```

**Binary classification:**

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",       # or areaUnderPR
).evaluate(predictions)
```

**Multiclass (also used for accuracy/F1 on a binary task):**

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = multi.evaluate(predictions, {multi.metricName: "accuracy"})
f1       = multi.evaluate(predictions, {multi.metricName: "f1"})
```

**Confusion matrix** by hand (a `groupBy` on label × prediction):

```python
cm = predictions.groupBy("label", "prediction").count().collect()
```

> **Time-based split, not random.** For time-series problems, splitting randomly
> leaks future into training and inflates metrics. Train on earlier years, hold
> out the latest:
> ```python
> train = df.filter(F.col("year") < test_year)
> test  = df.filter(F.col("year") == test_year)
> ```
> For non-temporal data, `df.randomSplit([0.7, 0.3], seed=42)` is fine.

---

## 20. Hyperparameter tuning & cross-validation

`ParamGridBuilder` enumerates a grid; `CrossValidator` does k-fold CV over it and
keeps the best model by the evaluator's metric.

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

grid = (ParamGridBuilder()
        .addGrid(rf.maxDepth, [3, 5, 7])
        .addGrid(rf.numTrees, [20, 50, 100])
        .build())                              # 3 × 3 = 9 combinations

cv = CrossValidator(
    estimator=rf_pipeline,                     # an estimator OR a whole Pipeline
    estimatorParamMaps=grid,
    evaluator=RegressionEvaluator(labelCol="charges", metricName="rmse"),
    numFolds=5,
    parallelism=1,                             # models fit in parallel (RAM-bound)
    seed=42,
)

cv_model = cv.fit(train)
print(cv_model.avgMetrics)                     # one score per grid combination
best_model = cv_model.bestModel
pred = best_model.transform(test)
```

> CV with `numFolds=k` and a grid of `n` combinations fits `n × k` models — keep
> grids small on a laptop. Spark's CV uses **random** folds, so on time-series data
> it leaks mildly; that's acceptable for *selecting* hyperparameters as long as the
> honest final score comes from a held-out future year the CV never saw.

---

## 21. Saving & loading models

A fitted pipeline (preprocessing + model) saves as one artifact, so serving code
loads a single path.

```python
from pyspark.ml import PipelineModel

model.write().overwrite().save("data/models/frost_classifier_gbt")
# (or, when fitting fresh: pipeline.fit(train).save("modelo_gasto"))

loaded = PipelineModel.load("data/models/frost_classifier_gbt")
pred = loaded.transform(new_data)
```

---

## 22. Performance: partitioning, caching, AQE

**Cache** a DataFrame you reuse (e.g. train/test across many fits) so Spark doesn't
recompute its lineage each time:

```python
train = train.cache()
test  = test.cache()
train.count()          # an action materializes the cache
```

**Sampling** for fast iteration on big data:

```python
toy   = df.sample(withReplacement=False, fraction=0.01, seed=42)  # 1% sample
small = train.sample(fraction=0.2, seed=42)                       # downsample train
```

**Adaptive Query Execution (AQE)** lets Spark re-optimize at runtime — coalescing
tiny shuffle partitions and splitting skewed ones:

```python
.config("spark.sql.adaptive.enabled", "true")
.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
.config("spark.sql.adaptive.skewJoin.enabled", "true")
```

**Other knobs used in this project:**

- `spark.sql.shuffle.partitions` — post-shuffle partition count (default 200; set
  `4` locally, `1024` for the heavy weather build).
- `spark.sql.sources.partitionOverwriteMode = dynamic` — re-running a job overwrites
  only the partitions actually written, not the whole table.
- `spark.driver.memory` — bump it (10–11g) when wide window/shuffle stages OOM.
- Streaming-style ingestion (download year → write Parquet → delete) keeps peak
  disk to ~one year of data regardless of how many years you ingest
  (`stream_ingest.py`).

---

## 23. Spark ↔ pandas interop

Pull a (small!) result into pandas for plotting or inspection. `toPandas()`
collects **everything to the driver** — always `limit`/`sample` first.

```python
pdf = df.limit(10).toPandas()          # safe: only 10 rows
sample_pdf = df.sample(0.001).toPandas()

# Build a tidy results table from CV scores
import pandas as pd
rmse_df = pd.DataFrame({"rmse": cv_model.avgMetrics, "alpha": alphas})
rmse_df.sort_values("rmse")
```

> For the weather project's single-node escape hatch, a representative **gold
> sample** is exported to one Parquet file so contributors can work in
> pandas/sklearn without running Spark at all.

---

## 24. Classic example: estimating π

Monte Carlo π — the canonical "embarrassingly parallel" Spark job. The fraction of
random points landing inside the unit quarter-circle ≈ π/4.

```python
from random import random

def inside_circle(_):
    x, y = random(), random()
    return 1 if x * x + y * y <= 1 else 0

n = 1_000_000
hits = sc.parallelize(range(n)).map(inside_circle).reduce(lambda a, b: a + b)
print(f"π ≈ {4 * hits / n}")
```

This is the whole point of Spark in one line: `parallelize` spreads the million
trials across all cores, `map` runs them in parallel, `reduce` aggregates — and it
scales to a cluster unchanged.

---

## Quick import cheat-sheet

```python
# Core
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, FloatType, DoubleType, TimestampType)

# ML — features
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (VectorAssembler, StringIndexer, OneHotEncoder,
                                StandardScaler, Imputer)
# ML — models
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
                                       GBTClassifier)
# ML — evaluation & tuning
from pyspark.ml.evaluation import (RegressionEvaluator, BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
```
