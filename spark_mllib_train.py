#!/usr/bin/env python3
"""
Spark MLlib version of the model-training stage.

Reads the same hourly dataset and uses the same chronological split as
pipeline.py (first 70 days train, last 20 days test), so the metrics are directly
comparable with the scikit-learn results.

    * Regression : GBTRegressor and RandomForestRegressor  (next-hour energy, MWh)
    * Classifier : GBTClassifier                           (spike / no spike)

Results are written to artifacts/spark_metrics.json and, when a store is
reachable, to the model_metrics collection (document id "spark"), where the
dashboard picks them up.

Run:  python spark_mllib_train.py          (needs Java 17+; first start takes ~20 s)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "smart_city_energy_timeseries.csv"
OUT = ROOT / "artifacts" / "spark_metrics.json"
TRAIN_DAYS = 70

RENAME = {"Population": "population", "Area (sq. km)": "area_sqkm",
          "Smart Infrastructure Score": "infra_score", "Public Transport Usage": "public_transport",
          "Smart Grid Adoption": "grid_adoption", "Internet Speed (Mbps)": "internet_mbps"}
FEATURES = ["population", "area_sqkm", "infra_score", "public_transport", "grid_adoption", "internet_mbps",
            "hour", "dayofweek", "is_weekend", "temperature_c", "lag_1h", "lag_24h", "rolling_24h_mean"]


def main() -> None:
    spark = (SparkSession.builder.master("local[*]").appName("smart-city-energy-mllib")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", "2g").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(str(DATA), header=True, inferSchema=True)
    for old, new in RENAME.items():
        df = df.withColumnRenamed(old, new)
    df = df.withColumn("is_spike", F.col("is_spike").cast("double"))
    train, test = df.filter(F.col("day_index") < TRAIN_DAYS).cache(), df.filter(F.col("day_index") >= TRAIN_DAYS).cache()
    print(f"Spark {spark.version} | train {train.count():,} rows | test {test.count():,} rows")

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    cls_assembler = VectorAssembler(inputCols=FEATURES + ["spike_threshold"], outputCol="features")
    ev = {m: RegressionEvaluator(labelCol="energy_mwh", predictionCol="prediction", metricName=m)
          for m in ("mae", "rmse", "r2")}

    results = []
    for name, est in (("Spark GBT regressor", GBTRegressor(labelCol="energy_mwh", maxIter=80, maxDepth=6, stepSize=0.1, seed=42)),
                      ("Spark random forest", RandomForestRegressor(labelCol="energy_mwh", numTrees=80, maxDepth=10, seed=42))):
        t0 = time.time()
        model = Pipeline(stages=[assembler, est]).fit(train)
        pred = model.transform(test)
        row = {"model": name, "engine": "Spark MLlib", **{m: float(e.evaluate(pred)) for m, e in ev.items()},
               "train_seconds": round(time.time() - t0, 1)}
        results.append(row)
        print(f"  {name:<22} MAE {row['mae']:.2f}  RMSE {row['rmse']:.2f}  R2 {row['r2']:.4f}  ({row['train_seconds']} s)")

    t0 = time.time()
    clf = Pipeline(stages=[cls_assembler, GBTClassifier(labelCol="is_spike", maxIter=60, maxDepth=5, seed=42)]).fit(train)
    pred = clf.transform(test).select("is_spike", "prediction", "probability")
    tp = pred.filter((F.col("prediction") == 1) & (F.col("is_spike") == 1)).count()
    fp = pred.filter((F.col("prediction") == 1) & (F.col("is_spike") == 0)).count()
    fn = pred.filter((F.col("prediction") == 0) & (F.col("is_spike") == 1)).count()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    auc = BinaryClassificationEvaluator(labelCol="is_spike", rawPredictionCol="rawPrediction").evaluate(clf.transform(test))
    cls = {"model": "Spark GBT classifier", "precision": precision, "recall": recall,
           "f1": 2 * precision * recall / max(precision + recall, 1e-9), "roc_auc": float(auc),
           "train_seconds": round(time.time() - t0, 1)}
    print(f"  {cls['model']:<22} precision {precision:.3f}  recall {recall:.3f}  F1 {cls['f1']:.3f}  AUC {auc:.3f}")

    payload = {"spark_version": spark.version, "regression": results, "classification": [cls]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}")

    try:
        from store import Store
        store = Store()
        store.upsert_many("model_metrics", [{"_id": "spark", **payload}])
        print(f"Published to model_metrics ({store.status()['label']})")
    except Exception as exc:                                    # noqa: BLE001
        print(f"Store not updated: {exc}")
    spark.stop()


if __name__ == "__main__":
    main()
