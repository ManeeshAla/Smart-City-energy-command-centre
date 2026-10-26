#!/usr/bin/env python3
"""
Smart City Energy Intelligence -- Experiment 10
Storage backend: Parquet (batch/history) + JSONL (stream events) + Joblib (model).
No MongoDB or other NoSQL database is used.

ACADEMIC INTEGRITY NOTE
-----------------------
The BDA Experiment 10 specification originally required a NoSQL database (e.g.
MongoDB) for document storage.  This implementation intentionally replaces that
requirement with file-based alternatives for operational simplicity.  If your
instructor grades on NoSQL integration, this implementation does NOT satisfy
that criterion and should be disclosed accordingly.  Do not mark the NoSQL
requirement as complete without confirming with your instructor.
"""
from __future__ import annotations

import csv
import json
import math
import threading
import warnings
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent
DATA_PATH    = ROOT / "smart_city_dataset.csv"
PARQUET_PATH = ROOT / "smart_city_batch_history.parquet"
MODEL_PATH   = ROOT / "energy_model.joblib"
EVENTS_PATH  = ROOT / "stream_events.jsonl"
PREDS_PATH   = ROOT / "prediction_history.jsonl"
ALERTS_PATH  = ROOT / "alert_history.jsonl"
STORE_PATH   = EVENTS_PATH   # kept for dashboard label compatibility

TARGET        = "Energy Consumption"
FEATURE_NAMES = [
    "Population", "Area (sq. km)", "Smart Infrastructure Score",
    "Public Transport Usage", "Air Quality Index", "Education Index",
    "Healthcare Index", "Employment Rate", "Smart Grid Adoption",
    "Waste Management Score", "Internet Speed (Mbps)", "Safety Index",
    "Cost of Living Index",
]

LOCK   = threading.Lock()
STREAM = {"running": False, "count": 0, "cursor": 0, "recent": []}

ALERT_THRESHOLD = 900.0   # dataset units

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
_EVENT_SCHEMA = {
    "_id", "timestamp", "city", "event_type",
    "actual_energy_consumption", "predicted_energy_consumption",
    "unit", "model", "source",
}
_PRED_SCHEMA  = {"_id", "timestamp", "city", "actual", "predicted", "error", "model"}
_ALERT_SCHEMA = {
    "_id", "timestamp", "city",
    "predicted_energy_consumption", "threshold", "severity",
}


def _validate(record: dict, required_keys: set, label: str) -> None:
    missing = required_keys - record.keys()
    if missing:
        raise ValueError(f"[schema] {label} record missing keys: {missing}")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def _load_jsonl_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "_id" in obj:
                    ids.add(obj["_id"])
            except json.JSONDecodeError:
                pass
    return ids


def append_jsonl(path: Path, record: dict, required_keys: set, label: str,
                 seen_ids=None) -> bool:
    """Validate, deduplicate, and append one record. Returns True if written."""
    _validate(record, required_keys, label)
    rid = record.get("_id", "")
    if seen_ids is not None and rid in seen_ids:
        return False
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    if seen_ids is not None:
        seen_ids.add(rid)
    return True


def read_jsonl(path: Path, limit=None) -> list:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if limit:
        return records[-limit:]
    return records


def clear_jsonl(path: Path) -> None:
    path.write_text("", encoding="utf-8")


# In-memory dedup sets -- populated from disk at startup
_SEEN_EVENT_IDS: set = set()
_SEEN_PRED_IDS:  set = set()
_SEEN_ALERT_IDS: set = set()

# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------
def save_parquet(rows: list) -> None:
    if not PANDAS_AVAILABLE:
        warnings.warn(
            "pandas is not installed -- skipping Parquet snapshot. "
            "Install pandas + pyarrow for full Parquet support.",
            RuntimeWarning, stacklevel=2,
        )
        return
    records = [
        {"city": r["city"], "actual": r["actual"],
         **dict(zip(FEATURE_NAMES, r["features"]))}
        for r in rows
    ]
    df = pd.DataFrame(records)
    try:
        df.to_parquet(PARQUET_PATH, index=False, engine="auto")
    except Exception as exc:
        warnings.warn(f"Parquet write failed: {exc}", RuntimeWarning, stacklevel=2)


def load_parquet_summary() -> dict:
    if not PANDAS_AVAILABLE or not PARQUET_PATH.exists():
        return {}
    try:
        df = pd.read_parquet(PARQUET_PATH)
        return {
            "parquet_rows":    int(len(df)),
            "parquet_columns": int(len(df.columns)),
            "parquet_path":    PARQUET_PATH.name,
            "energy_mean":     float(df["actual"].mean()),
            "energy_std":      float(df["actual"].std()),
            "energy_min":      float(df["actual"].min()),
            "energy_max":      float(df["actual"].max()),
        }
    except Exception as exc:
        return {"parquet_error": str(exc)}


# ---------------------------------------------------------------------------
# Joblib model persistence
# ---------------------------------------------------------------------------
def save_model(model_obj: dict) -> None:
    if not JOBLIB_AVAILABLE:
        warnings.warn(
            "joblib is not available -- model will not be persisted to disk.",
            RuntimeWarning, stacklevel=2,
        )
        return
    joblib.dump(model_obj, MODEL_PATH)


def load_model_from_disk():
    if not JOBLIB_AVAILABLE or not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------
def as_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def read_data():
    rows = []
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or TARGET not in reader.fieldnames:
            raise ValueError(f"Dataset needs the '{TARGET}' target column")
        absent = [name for name in FEATURE_NAMES if name not in reader.fieldnames]
        if absent:
            raise ValueError(f"Dataset is missing required columns: {absent}")
        for r in reader:
            vals   = [as_float(r.get(c)) for c in FEATURE_NAMES]
            target = as_float(r.get(TARGET))
            if target is not None and all(x is not None for x in vals):
                rows.append({
                    "city":     r.get("City Name") or f"City {len(rows)+1}",
                    "features": vals,
                    "actual":   target,
                })
    if len(rows) < 20:
        raise ValueError("At least 20 complete city rows are required")
    return rows


# ---------------------------------------------------------------------------
# Fallback least-squares solver
# ---------------------------------------------------------------------------
def solve(matrix, vector):
    n = len(vector)
    a = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            raise ValueError("Regression matrix is singular")
        divisor = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= divisor
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


# ---------------------------------------------------------------------------
# Model training and prediction
# ---------------------------------------------------------------------------
def fit(train):
    if SKLEARN_AVAILABLE:
        estimator = make_pipeline(StandardScaler(), LinearRegression())
        estimator.fit([r["features"] for r in train], [r["actual"] for r in train])
        return {"estimator": estimator, "backend": "scikit-learn"}
    k      = len(FEATURE_NAMES)
    means  = [sum(r["features"][j] for r in train) / len(train) for j in range(k)]
    scales = [
        (sum((r["features"][j] - means[j]) ** 2 for r in train) / len(train)) ** 0.5 or 1.0
        for j in range(k)
    ]
    matrix = [[0.0] * (k + 1) for _ in range(k + 1)]
    vector = [0.0] * (k + 1)
    for row in train:
        x = [1.0] + [(v - m) / s for v, m, s in zip(row["features"], means, scales)]
        y = row["actual"]
        for i in range(k + 1):
            vector[i] += x[i] * y
            for j in range(k + 1):
                matrix[i][j] += x[i] * x[j]
    for i in range(1, k + 1):
        matrix[i][i] += 1e-8
    return {"means": means, "scales": scales, "coef": solve(matrix, vector),
            "backend": "standard-library fallback"}


def predict(model, features):
    if "estimator" in model:
        return float(model["estimator"].predict([features])[0])
    x = [1.0] + [(v - m) / s for v, m, s in zip(features, model["means"], model["scales"])]
    return sum(a * b for a, b in zip(model["coef"], x))


def metrics(actual, predicted):
    n    = len(actual)
    errs = [p - y for y, p in zip(actual, predicted)]
    mae  = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    mean = sum(actual) / n
    sst  = sum((y - mean) ** 2 for y in actual)
    r2   = 1 - sum(e * e for e in errs) / sst if sst else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# Boot-time initialisation
# ---------------------------------------------------------------------------
ROWS           = read_data()
SPLIT          = int(len(ROWS) * 0.8)
TRAIN, HOLDOUT = ROWS[:SPLIT], ROWS[SPLIT:]

_saved_model = load_model_from_disk()
if _saved_model is not None:
    MODEL = _saved_model
    print("[joblib] Loaded model from disk:", MODEL_PATH.name)
else:
    MODEL = fit(TRAIN)
    save_model(MODEL)
    print(f"[joblib] Trained and saved model -> {MODEL_PATH.name}")

PREDICTIONS      = [predict(MODEL, r["features"]) for r in HOLDOUT]
ACTUAL           = [r["actual"] for r in HOLDOUT]
MODEL_METRICS    = metrics(ACTUAL, PREDICTIONS)
BASELINE_PRED    = [sum(r["actual"] for r in TRAIN) / len(TRAIN)] * len(HOLDOUT)
BASELINE_METRICS = metrics(ACTUAL, BASELINE_PRED)

save_parquet(ROWS)
PARQUET_SUMMARY = load_parquet_summary()

# Bulk-write holdout predictions (deduped on restart)
_SEEN_PRED_IDS = _load_jsonl_ids(PREDS_PATH)
for _i, (_row, _pred) in enumerate(zip(HOLDOUT, PREDICTIONS)):
    _pred_rec = {
        "_id":       f"pred-city-{_i+1:04d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city":      _row["city"],
        "actual":    _row["actual"],
        "predicted": _pred,
        "error":     _pred - _row["actual"],
        "model":     "energy_ols_v1",
    }
    append_jsonl(PREDS_PATH, _pred_rec, _PRED_SCHEMA, "prediction", _SEEN_PRED_IDS)

_SEEN_EVENT_IDS = _load_jsonl_ids(EVENTS_PATH)
_SEEN_ALERT_IDS = _load_jsonl_ids(ALERTS_PATH)


# ---------------------------------------------------------------------------
# Overview payload
# ---------------------------------------------------------------------------
def make_overview():
    sorted_rows = sorted(ROWS, key=lambda r: r["features"][8])
    batches, n  = [], len(sorted_rows)
    for idx, label in enumerate(("Lower adoption", "Mid adoption", "Higher adoption")):
        chunk = sorted_rows[idx * n // 3: (idx + 1) * n // 3]
        batches.append({
            "label":       label,
            "cities":      len(chunk),
            "grid_mean":   sum(r["features"][8] for r in chunk) / len(chunk),
            "energy_mean": sum(r["actual"]       for r in chunk) / len(chunk),
        })
    sample = [
        {"city": r["city"], "actual": r["actual"], "prediction": p, "index": i}
        for i, (r, p) in enumerate(zip(HOLDOUT, PREDICTIONS))
    ]
    return {
        "dataset_rows":     len(ROWS),
        "columns":          15,
        "target":           TARGET,
        "train_rows":       len(TRAIN),
        "holdout_rows":     len(HOLDOUT),
        "model_metrics":    MODEL_METRICS,
        "baseline_metrics": BASELINE_METRICS,
        "batch":            batches,
        "holdout_preview":  sample[:120],
        "store":            EVENTS_PATH.name,
        "parquet_store":    PARQUET_PATH.name,
        "model_store":      MODEL_PATH.name,
        "pred_store":       PREDS_PATH.name,
        "alert_store":      ALERTS_PATH.name,
        "parquet_summary":  PARQUET_SUMMARY,
        "features":         FEATURE_NAMES,
        "storage_backend":  "Parquet (batch) + JSONL (stream/preds/alerts) + Joblib (model)",
        "note": (
            "The dataset has no live sensor feed. The stream demo replays held-out "
            "city rows, stores offline predictions as JSONL events, and flags "
            "high-consumption cities as JSONL alerts. The full dataset is persisted "
            "as a Parquet snapshot and the trained model is saved with Joblib. "
            "NoSQL (MongoDB) is NOT used -- see README.md for academic disclosure."
        ),
    }


OVERVIEW = make_overview()


# ---------------------------------------------------------------------------
# Stream emitter
# ---------------------------------------------------------------------------
def emit_event():
    with LOCK:
        if not STREAM["running"]:
            return
        i = STREAM["cursor"]
        if i >= len(HOLDOUT):
            STREAM["running"] = False
            return
        row  = HOLDOUT[i]
        pred = PREDICTIONS[i]
        eid  = f"energy-city-{i+1:04d}"
        item = {
            "_id":                          eid,
            "timestamp":                    datetime.now(timezone.utc).isoformat(),
            "city":                         row["city"],
            "event_type":                   "simulated_heldout_city",
            "actual_energy_consumption":    row["actual"],
            "predicted_energy_consumption": pred,
            "unit":                         "dataset units",
            "model":                        "energy_ols_v1",
            "source":                       "smart_city_dataset.csv",
        }
        append_jsonl(EVENTS_PATH, item, _EVENT_SCHEMA, "event", _SEEN_EVENT_IDS)

        if pred > ALERT_THRESHOLD:
            alert = {
                "_id":                          f"alert-city-{i+1:04d}",
                "timestamp":                    datetime.now(timezone.utc).isoformat(),
                "city":                         row["city"],
                "predicted_energy_consumption": pred,
                "threshold":                    ALERT_THRESHOLD,
                "severity":                     "HIGH" if pred > 1200 else "MEDIUM",
            }
            append_jsonl(ALERTS_PATH, alert, _ALERT_SCHEMA, "alert", _SEEN_ALERT_IDS)

        STREAM["cursor"] += 1
        STREAM["count"]  += 1
        STREAM["recent"]  = (STREAM["recent"] + [item])[-40:]

    threading.Timer(1.2, emit_event).start()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/overview":
            self.send_json(OVERVIEW)
        elif path == "/api/stream":
            with LOCK:
                state = {
                    "running": STREAM["running"],
                    "count":   STREAM["count"],
                    "recent":  STREAM["recent"][-20:],
                }
            self.send_json(state)
        elif path == "/api/predictions":
            self.send_json(read_jsonl(PREDS_PATH, limit=50))
        elif path == "/api/alerts":
            self.send_json(read_jsonl(ALERTS_PATH, limit=20))
        elif path == "/api/parquet_summary":
            self.send_json(PARQUET_SUMMARY)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/start":
            with LOCK:
                if STREAM["cursor"] >= len(HOLDOUT):
                    STREAM["cursor"] = 0
                was_running       = STREAM["running"]
                STREAM["running"] = True
            if not was_running:
                emit_event()
            self.send_json({"running": True})
        elif path == "/api/pause":
            with LOCK:
                STREAM["running"] = False
            self.send_json({"running": False})
        elif path == "/api/reset":
            with LOCK:
                STREAM.update({"running": False, "count": 0, "cursor": 0, "recent": []})
                clear_jsonl(EVENTS_PATH)
                _SEEN_EVENT_IDS.clear()
            self.send_json({"reset": True})
        else:
            self.send_json({"error": "not found"}, 404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    clear_jsonl(EVENTS_PATH)
    _SEEN_EVENT_IDS.clear()

    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print(f"\nSmart City Energy Dashboard  ->  http://127.0.0.1:8765")
    print(f"Rows: {len(ROWS):,}  |  Training: {len(TRAIN):,}  |  Holdout: {len(HOLDOUT):,}")
    print(f"MAE {MODEL_METRICS['mae']:.2f}  |  RMSE {MODEL_METRICS['rmse']:.2f}  |  R2 {MODEL_METRICS['r2']:.4f}")
    print(f"\nStorage backends:")
    print(f"  Parquet -> {PARQUET_PATH.name}  ({PARQUET_SUMMARY.get('parquet_rows', '?')} rows)")
    print(f"  Joblib  -> {MODEL_PATH.name}")
    print(f"  JSONL   -> {EVENTS_PATH.name}  (stream events, cleared on restart)")
    print(f"  JSONL   -> {PREDS_PATH.name}  (prediction history, append-only)")
    print(f"  JSONL   -> {ALERTS_PATH.name}  (alert history, append-only)")
    print(f"\nNOTE: NoSQL (MongoDB) is NOT used. See README.md for academic disclosure.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
