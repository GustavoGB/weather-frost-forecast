"""
Train a next-day frost-risk classifier on the gold table — the model that
backs the agricultural Frost-Risk Score in the project motivation.

This uses **Spark ML (the DataFrame-based API)**, not the legacy
`pyspark.mllib` RDD API. Spark ML is the supported path going forward; it
composes via the `Pipeline` abstraction (Imputer → VectorAssembler →
Classifier) so the same code scales unchanged from this laptop to a cluster.

Key design choices and the reasons behind them
----------------------------------------------
1. **Time-based train/test split** (not random).
   A random split would leak future weather into training and inflate the
   metrics — a meaningless model. We train on the earlier years and hold out
   the most recent year. This is the only split that simulates how the model
   would be used in production: "fit on history, predict tomorrow".

2. **GBTClassifier** (gradient-boosted trees).
   Tree ensembles dominate tabular tasks like this one (mixed continuous +
   spatial + seasonal features, non-linear interactions, robust to scale).
   GBT typically beats RandomForest on this kind of data; it's also natively
   parallel in Spark.

3. **Imputer with median strategy.**
   Many gold rows have null lag/rolling values (first N days of each
   station's history) or null PRCP (sensor doesn't measure rain). We impute
   with the column median rather than dropping rows, because dropping would
   bias the training set toward long-running stations.

4. **Class-balanced sample (optional).**
   Frost positive rate is ~30% — already healthy — but `--sample-fraction`
   lets you train on a downsample to iterate quickly on a laptop while still
   producing a model you can ship metrics for. On a cluster you'd train on
   the full set.

5. **Pipeline persisted via `PipelineModel.save`.**
   Saves both the imputer fits AND the trained classifier as one artifact,
   so serving code only needs to load one path.

6. **Optional hyperparameter search** (`--tune`).
   Wraps the Pipeline in a `CrossValidator` + `ParamGridBuilder` to pick the
   best hyperparameters by k-fold areaUnderROC. Spark's CV uses *random* folds,
   so it leaks mildly in time — fine for SELECTION, while the honest score still
   comes from the held-out test year (never seen during CV).

Usage:
    uv run python -m src.ml.train_frost_classifier
    uv run python -m src.ml.train_frost_classifier --test-year 2024 --sample-fraction 0.2
    uv run python -m src.ml.train_frost_classifier --algorithm rf  # RandomForest instead of GBT
    uv run python -m src.ml.train_frost_classifier --tune          # k-fold CV hyperparameter search
    uv run python -m src.ml.train_frost_classifier --algorithm lr --tune --cv-folds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingestion.utils import build_spark, get_logger, load_config, resolve

log = get_logger("train_frost")

LABEL_COL = "label"

# Feature groups — kept explicit so a reviewer can see exactly what the model sees.
SEASONAL = ["day_of_year", "month"]
SPATIAL  = ["latitude", "longitude", "elevation_m"]
TODAY    = ["tmin_c", "tmax_c", "tavg_c", "temp_range_c", "prcp_mm"]
LAG      = ["tmin_c_lag_1", "tmin_c_lag_7",
            "tmax_c_lag_1", "tmax_c_lag_7",
            "prcp_mm_lag_1", "prcp_mm_lag_7"]
ROLL7    = ["tmin_c_roll7_mean", "tmin_c_roll7_min",
            "tmax_c_roll7_mean", "tmax_c_roll7_max",
            "prcp_mm_roll7_sum"]
ROLL30   = ["tmin_c_roll30_mean", "tmax_c_roll30_mean",
            "prcp_mm_roll30_sum", "frost_days_roll30"]
GDD      = ["gdd_corn_ytd", "gdd_wheat_ytd"]
# Neighbour-station / regional features (added in build_gold.add_regional_features).
REGION   = ["region_tmin_mean", "region_tmin_min", "tmin_minus_region"]
ALL_FEATURES = SEASONAL + SPATIAL + TODAY + LAG + ROLL7 + ROLL30 + GDD + REGION

# Precipitation columns, grouped for ablation studies ("does precip help predict
# frost?"). Drop these from ALL_FEATURES to train a no-precip model and compare.
PRECIP_COLS = ["prcp_mm", "prcp_mm_lag_1", "prcp_mm_lag_7",
               "prcp_mm_roll7_sum", "prcp_mm_roll30_sum"]


def load_training_data(spark: SparkSession, gold_path: Path) -> DataFrame:
    df = spark.read.parquet(str(gold_path))
    # Only rows where we have tomorrow's outcome can be used.
    df = df.filter(F.col("next_day_is_frost").isNotNull())
    df = df.withColumn(LABEL_COL, F.col("next_day_is_frost").cast("double"))
    return df.select("year", LABEL_COL, *ALL_FEATURES)


def time_split(df: DataFrame, test_year: int) -> tuple[DataFrame, DataFrame]:
    train = df.filter(F.col("year") < test_year).drop("year")
    test  = df.filter(F.col("year") == test_year).drop("year")
    return train, test


def build_pipeline(algorithm: str) -> Pipeline:
    imputer = Imputer(
        strategy="median",
        inputCols=ALL_FEATURES,
        outputCols=[f"{c}__imp" for c in ALL_FEATURES],
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}__imp" for c in ALL_FEATURES],
        outputCol="features",
        handleInvalid="keep",
    )
    if algorithm == "gbt":
        clf = GBTClassifier(
            labelCol=LABEL_COL,
            featuresCol="features",
            maxIter=40,
            maxDepth=6,
            stepSize=0.1,
            subsamplingRate=0.8,
            seed=42,
        )
    elif algorithm == "rf":
        clf = RandomForestClassifier(
            labelCol=LABEL_COL,
            featuresCol="features",
            numTrees=80,
            maxDepth=10,
            subsamplingRate=0.7,
            featureSubsetStrategy="sqrt",
            seed=42,
        )
    elif algorithm == "lr":
        # Linear baseline. LR is scale-sensitive, so we standardize the assembled
        # vector first (StandardScaler) — trees don't need this, linear models do.
        scaler = StandardScaler(inputCol="features", outputCol="features_scaled",
                                withMean=False, withStd=True)
        clf = LogisticRegression(
            labelCol=LABEL_COL,
            featuresCol="features_scaled",
            maxIter=50,
            regParam=0.0,
        )
        return Pipeline(stages=[imputer, assembler, scaler, clf])
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return Pipeline(stages=[imputer, assembler, clf])


def build_param_grid(algorithm: str, clf) -> list:
    """Hyperparameter grid for `CrossValidator`, per algorithm.

    Kept deliberately small so a full grid search stays tractable on a laptop
    (combos x folds fits). The grids target the parameters that actually move
    the needle for each model family — tree depth/size for the ensembles,
    regularization for the linear baseline.
    """
    g = ParamGridBuilder()
    if algorithm == "gbt":
        g = (g.addGrid(clf.maxDepth, [4, 6])
              .addGrid(clf.maxIter, [30, 60]))          # 4 combos
    elif algorithm == "rf":
        g = (g.addGrid(clf.numTrees, [60, 120])
              .addGrid(clf.maxDepth, [8, 12]))          # 4 combos
    elif algorithm == "lr":
        g = (g.addGrid(clf.regParam, [0.0, 0.01, 0.1])
              .addGrid(clf.elasticNetParam, [0.0, 0.5]))  # 6 combos
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return g.build()


def evaluate(predictions: DataFrame) -> dict[str, float]:
    auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    pr_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(predictions)

    multi = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction"
    )
    accuracy = multi.evaluate(predictions, {multi.metricName: "accuracy"})
    f1       = multi.evaluate(predictions, {multi.metricName: "f1"})
    p_weight = multi.evaluate(predictions, {multi.metricName: "weightedPrecision"})
    r_weight = multi.evaluate(predictions, {multi.metricName: "weightedRecall"})

    return {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "f1": f1,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
    }


def confusion(predictions: DataFrame) -> dict[str, int]:
    cm = (
        predictions
        .groupBy(LABEL_COL, "prediction")
        .count()
        .collect()
    )
    out = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in cm:
        key = (int(row[LABEL_COL]), int(row["prediction"]))
        out[{(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[key]] = row["count"]
    return out


def top_features(model: PipelineModel, k: int = 12) -> list[tuple[str, float]]:
    classifier = model.stages[-1]
    if not hasattr(classifier, "featureImportances"):
        return []
    importances = classifier.featureImportances.toArray()
    pairs = sorted(zip(ALL_FEATURES, importances), key=lambda p: p[1], reverse=True)
    return pairs[:k]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-year", type=int, default=2024,
                    help="Year held out for evaluation (must be the latest in gold).")
    ap.add_argument("--sample-fraction", type=float, default=0.2,
                    help="Fraction of TRAIN rows to keep — speeds up laptop runs. "
                         "Set to 1.0 on a cluster.")
    ap.add_argument("--algorithm", choices=["gbt", "rf", "lr"], default="gbt")
    ap.add_argument("--tune", action="store_true",
                    help="Run k-fold CrossValidator + ParamGridBuilder to pick "
                         "hyperparameters (areaUnderROC), then evaluate the best "
                         "model on the held-out test year.")
    ap.add_argument("--cv-folds", type=int, default=3,
                    help="Number of CrossValidator folds when --tune is set.")
    ap.add_argument("--cv-parallelism", type=int, default=1,
                    help="How many models CrossValidator fits in parallel. Keep "
                         "at 1 on low-RAM machines (avoids Spark OOM).")
    ap.add_argument("--output-name", default=None,
                    help="Subdir under data/models/ to save the pipeline. "
                         "Defaults to frost_classifier_<algo>")
    args = ap.parse_args()

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    gold = resolve(cfg["paths"]["gold"]) / "station_daily_features"
    models_dir = resolve("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or f"frost_classifier_{args.algorithm}"
    model_path = models_dir / out_name

    log.info("Loading gold from %s", gold)
    df = load_training_data(spark, gold)
    train, test = time_split(df, args.test_year)

    if args.sample_fraction < 1.0:
        train = train.sample(withReplacement=False, fraction=args.sample_fraction, seed=42)
        log.info("Downsampled train to fraction=%.2f", args.sample_fraction)

    train = train.cache()
    test = test.cache()
    n_train, n_test = train.count(), test.count()
    train_pos = train.filter(F.col(LABEL_COL) == 1).count()
    test_pos  = test.filter(F.col(LABEL_COL) == 1).count()
    log.info("train rows: %s  (%.2f%% positive)", f"{n_train:,}", 100 * train_pos / n_train)
    log.info("test rows : %s  (%.2f%% positive)", f"{n_test:,}",  100 * test_pos  / n_test)

    log.info("Building Spark ML Pipeline (algorithm=%s)", args.algorithm)
    pipeline = build_pipeline(args.algorithm)

    cv_info = None
    if args.tune:
        # NOTE on temporal leakage: Spark's CrossValidator uses *random* folds,
        # so within the training years a fold can train on later days and validate
        # on earlier ones. That's a mild leak, tolerable for HYPERPARAMETER SELECTION
        # only — the honest performance number still comes from the held-out test
        # year below, which is never seen during CV. A fully rigorous setup would use
        # forward-chaining (time-based) folds, which Spark ML doesn't ship natively.
        clf = pipeline.getStages()[-1]
        grid = build_param_grid(args.algorithm, clf)
        cv_evaluator = BinaryClassificationEvaluator(
            labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        log.info("Tuning: %d param combos × %d folds = %d fits (parallelism=%d)",
                 len(grid), args.cv_folds, len(grid) * args.cv_folds, args.cv_parallelism)
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=grid,
            evaluator=cv_evaluator,
            numFolds=args.cv_folds,
            parallelism=args.cv_parallelism,
            seed=42,
        )
        cv_model = cv.fit(train)
        best_idx = max(range(len(cv_model.avgMetrics)), key=lambda i: cv_model.avgMetrics[i])
        best_params = {p.name: v for p, v in cv_model.getEstimatorParamMaps()[best_idx].items()}
        cv_info = {
            "cv_folds": args.cv_folds,
            "param_grid_size": len(grid),
            "metric": "areaUnderROC",
            "best_cv_score": float(cv_model.avgMetrics[best_idx]),
            "best_params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                            for k, v in best_params.items()},
            "all_cv_scores": [float(s) for s in cv_model.avgMetrics],
        }
        log.info("Best CV areaUnderROC=%.4f with %s",
                 cv_info["best_cv_score"], cv_info["best_params"])
        model = cv_model.bestModel
    else:
        log.info("Fitting model...")
        model = pipeline.fit(train)

    log.info("Predicting on test (year=%d)", args.test_year)
    predictions = model.transform(test)

    metrics = evaluate(predictions)
    cm = confusion(predictions)
    importances = top_features(model)

    print("\n" + "=" * 60)
    print(f"FROST-RISK CLASSIFIER — algorithm={args.algorithm}")
    print(f"train: years < {args.test_year}    test: year = {args.test_year}")
    if cv_info:
        print(f"tuned: {args.cv_folds}-fold CV over {cv_info['param_grid_size']} "
              f"param combos → best CV areaUnderROC={cv_info['best_cv_score']:.4f}")
        print(f"  best params: {cv_info['best_params']}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:>22} : {v:.4f}")
    print()
    print(f"  confusion matrix (test): TN={cm['tn']:,}  FP={cm['fp']:,}")
    print(f"                           FN={cm['fn']:,}  TP={cm['tp']:,}")
    print()
    print("  top features by importance:")
    for name, imp in importances:
        bar = "█" * int(imp * 80)
        print(f"    {name:>22}  {imp:.4f}  {bar}")
    print("=" * 60 + "\n")

    log.info("Saving pipeline → %s", model_path)
    model.write().overwrite().save(str(model_path))

    summary = {
        "algorithm": args.algorithm,
        "test_year": args.test_year,
        "sample_fraction": args.sample_fraction,
        "n_train": n_train,
        "n_test": n_test,
        "tuning": cv_info,
        "metrics": metrics,
        "confusion": cm,
        "top_features": [{"feature": n, "importance": float(i)} for n, i in importances],
    }
    summary_path = models_dir / f"{out_name}_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved metrics → %s", summary_path)

    spark.stop()


if __name__ == "__main__":
    main()

