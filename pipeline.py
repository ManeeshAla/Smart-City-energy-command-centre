#!/usr/bin/env python3
"""
Batch + ML stage of the Smart City pipeline.

    smart_city_energy_timeseries.csv
        -> Parquet snapshot + batch aggregates          (batch layer)
        -> energy forecast models (next-hour MWh)       (regression)
        -> spike-risk classifier                        (classification)
        -> models.joblib, metrics, aggregates           (artifacts/)
        -> MongoDB / local store                        (batch_summary, model_metrics)

Chronological split: the first 70 days train the models, the last 20 days are
the held-out period replayed by the live stream.

Run:  python pipeline.py            (trains and publishes)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_PATH = ROOT / "smart_city_energy_timeseries.csv"
MODELS_PATH = ARTIFACTS / "models.joblib"
PARQUET_PATH = ARTIFACTS / "smart_city_batch_history.parquet"

TRAIN_DAYS = 70
TARGET = "energy_mwh"
STATIC = ["Population", "Area (sq. km)", "Smart Infrastructure Score",
          "Public Transport Usage", "Smart Grid Adoption", "Internet Speed (Mbps)"]
FEATURES = STATIC + ["hour", "dayofweek", "is_weekend", "temperature_c",
                     "lag_1h", "lag_24h", "rolling_24h_mean"]
CLS_FEATURES = FEATURES + ["spike_threshold"]
ALERT_PROBABILITY = 0.5


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df.sort_values(["timestamp", "city"]).reset_index(drop=True)


def split(df: pd.DataFrame):
    return df[df["day_index"] < TRAIN_DAYS], df[df["day_index"] >= TRAIN_DAYS]


def reg_metrics(y, p) -> dict:
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(math.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


# --------------------------------------------------------------------------- ML
def train_regression(train, test):
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    rows = []
    for name, pred in (("Mean baseline", np.full(len(yte), ytr.mean())),
                       ("Persistence (last hour)", test["lag_1h"].to_numpy()),
                       ("Seasonal naive (same hour yesterday)", test["lag_24h"].to_numpy())):
        rows.append({"model": name, "kind": "baseline", **reg_metrics(yte, pred)})

    candidates = {
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=5,
                                               n_jobs=-1, random_state=42),
        "Gradient boosting": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08,
                                                           max_iter=250, random_state=42),
    }
    fitted = {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        rows.append({"model": name, "kind": "learned",
                     "train_seconds": round(time.time() - t0, 1),
                     **reg_metrics(yte, model.predict(Xte))})
        fitted[name] = model
    best = min((r for r in rows if r["kind"] == "learned"), key=lambda r: r["mae"])["model"]
    return rows, fitted[best], best


def train_classifier(train, test):
    Xtr, ytr, Xte, yte = train[CLS_FEATURES], train["is_spike"], test[CLS_FEATURES], test["is_spike"]
    candidates = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Gradient boosting": HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08,
                                                            max_iter=200, random_state=42),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= ALERT_PROBABILITY).astype(int)
        rows.append({"model": name,
                     "precision": float(precision_score(yte, pred, zero_division=0)),
                     "recall": float(recall_score(yte, pred, zero_division=0)),
                     "f1": float(f1_score(yte, pred, zero_division=0)),
                     "roc_auc": float(roc_auc_score(yte, prob))})
        fitted[name] = model
    best = max(rows, key=lambda r: r["f1"])["model"]
    return rows, fitted[best], best


def feature_importance(model, test) -> list[dict]:
    sample = test.sample(4000, random_state=0)
    res = permutation_importance(model, sample[FEATURES], sample[TARGET], n_repeats=3,
                                 scoring="neg_mean_absolute_error", random_state=0, n_jobs=1)
    out = [{"feature": f, "importance": float(v)} for f, v in zip(FEATURES, res.importances_mean)]
    return sorted(out, key=lambda r: r["importance"], reverse=True)


# ---------------------------------------------------------------------- batch layer
def batch_aggregates(df: pd.DataFrame) -> dict:
    heat = (df.groupby(["dayofweek", "hour"])[TARGET].mean().round(2).reset_index()
            .rename(columns={TARGET: "mean_mwh"}))

    daily = (df.groupby("day_index")
               .agg(date=("timestamp", lambda s: s.min().strftime("%Y-%m-%d")),
                    total_mwh=(TARGET, "sum"), mean_temp=("temperature_c", "mean"),
                    spikes=("is_spike", "sum"))
               .round(2).reset_index())

    city = (df.groupby("city")
              .agg(population=("Population", "first"), grid=("Smart Grid Adoption", "first"),
                   mean_mwh=(TARGET, "mean"), peak_mwh=(TARGET, "max"),
                   spike_rate=("is_spike", "mean"))
              .round(3).reset_index())
    city["mwh_per_100k"] = (city["mean_mwh"] / city["population"] * 1e5).round(2)
    city["grid_group"] = pd.qcut(city["grid"], 3, labels=["Low adoption", "Mid adoption", "High adoption"])
    city["grid_group"] = city["grid_group"].astype(str)

    tertiles = (city.groupby("grid_group")
                    .agg(cities=("city", "count"), mean_grid=("grid", "mean"),
                         mwh_per_100k=("mwh_per_100k", "mean"))
                    .round(2).reset_index())
    tertiles["order"] = tertiles["grid_group"].map({"Low adoption": 0, "Mid adoption": 1, "High adoption": 2})
    tertiles = tertiles.sort_values("order").drop(columns="order")

    kpis = {"rows": int(len(df)), "cities": int(df["city"].nunique()),
            "hours": int(df["timestamp"].nunique()),
            "period_start": df["timestamp"].min().isoformat(),
            "period_end": df["timestamp"].max().isoformat(),
            "mean_mwh": float(df[TARGET].mean()), "peak_mwh": float(df[TARGET].max()),
            "spike_rate": float(df["is_spike"].mean())}
    rec = lambda d: json.loads(d.to_json(orient="records"))      # noqa: E731
    return {"heatmap": rec(heat), "daily": rec(daily), "cities": rec(city),
            "grid_groups": rec(tertiles), "kpis": kpis}


# --------------------------------------------------------------------------- main
def run(publish: bool = True) -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_data()
    train, test = split(df)
    print(f"Loaded {len(df):,} rows | train {len(train):,} | test {len(test):,}")

    df.to_parquet(PARQUET_PATH, index=False)
    agg = batch_aggregates(df)
    for key in ("heatmap", "daily", "cities", "grid_groups"):
        pd.DataFrame(agg[key]).to_parquet(ARTIFACTS / f"batch_{key}.parquet", index=False)
    (ARTIFACTS / "batch_aggregates.json").write_text(json.dumps(agg), encoding="utf-8")
    print(f"Batch layer: Parquet snapshot + 4 aggregate tables -> {ARTIFACTS.name}/")

    reg_rows, reg_model, reg_best = train_regression(train, test)
    cls_rows, cls_model, cls_best = train_classifier(train, test)
    importance = feature_importance(reg_model, test)
    best_row = next(r for r in reg_rows if r["model"] == reg_best)
    cls_row = next(r for r in cls_rows if r["model"] == cls_best)

    joblib.dump({"regressor": reg_model, "classifier": cls_model, "features": FEATURES,
                 "cls_features": CLS_FEATURES, "alert_probability": ALERT_PROBABILITY,
                 "regressor_name": reg_best, "classifier_name": cls_best}, MODELS_PATH)

    metrics = {"regression": reg_rows, "classification": cls_rows,
               "best_regressor": reg_best, "best_classifier": cls_best,
               "importance": importance, "train_rows": int(len(train)), "test_rows": int(len(test)),
               "alert_probability": ALERT_PROBABILITY}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nEnergy forecast (held-out last 20 days)")
    for r in reg_rows:
        print(f"  {r['model']:<38} MAE {r['mae']:7.2f}  RMSE {r['rmse']:7.2f}  R2 {r['r2']:.4f}")
    print(f"  -> selected: {reg_best}")
    print("\nSpike classifier (alert if probability >= %.2f)" % ALERT_PROBABILITY)
    for r in cls_rows:
        print(f"  {r['model']:<22} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"F1 {r['f1']:.3f}  AUC {r['roc_auc']:.3f}")
    print(f"  -> selected: {cls_best}")

    if publish:
        from store import Store
        store = Store()
        store.replace_all("batch_summary", [{"_id": k, "data": v} for k, v in agg.items()])
        store.replace_all("model_metrics", [{"_id": "latest", **metrics}])
        print(f"\nPublished batch summary + model metrics -> {store.status()['label']}")
    return {"best_regressor": best_row, "best_classifier": cls_row}


if __name__ == "__main__":
    run(publish="--no-publish" not in sys.argv)
