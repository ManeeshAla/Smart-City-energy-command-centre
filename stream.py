"""
Stream processing stage: replays the held-out 20 days one hour at a time.

For every tick (one hour, all 40 cities):
    ingest readings -> run the forecast model + spike classifier
    -> write events / predictions / alerts / tick metrics to the store
    -> update the in-memory state the dashboard polls.

The forecast for hour t only uses information available at hour t-1
(lags, rolling mean) plus the temperature forecast for hour t.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import joblib
import numpy as np

import pipeline
from store import Store

SPEEDS = {"1x": 1.0, "2x": 0.5, "5x": 0.2}
HISTORY_TICKS = 96            # ticks kept in memory for the live charts (4 days)


class StreamEngine:
    def __init__(self, store: Store):
        self.store = store
        bundle = joblib.load(pipeline.MODELS_PATH)
        self.reg = bundle["regressor"]
        self.cls = bundle["classifier"]
        self.alert_p = bundle["alert_probability"]
        self.reg_name = bundle["regressor_name"]
        self.cls_name = bundle["classifier_name"]

        _, test = pipeline.split(pipeline.load_data())
        self.ticks = [(ts, g.reset_index(drop=True)) for ts, g in test.groupby("timestamp")]
        self.n_ticks = len(self.ticks)

        self.lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.interval = SPEEDS["1x"]
        self._reset_state()

    # ----------------------------------------------------------------- state
    def _reset_state(self) -> None:
        self.cursor = 0
        self.running = False
        self.finished = False
        self.error = ""
        self.totals = {"events": 0, "predictions": 0, "alerts": 0, "confirmed": 0,
                       "abs_err": 0.0, "sq_err": 0.0, "n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.series: deque = deque(maxlen=HISTORY_TICKS)          # city-total series
        self.city_hist: dict[str, deque] = {}
        self.latest: list[dict] = []
        self.latency_ms = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            t = self.totals
            n = max(t["n"], 1)
            flagged = t["tp"] + t["fp"]
            actual_pos = t["tp"] + t["fn"]
            return {
                "running": self.running, "finished": self.finished, "error": self.error,
                "tick": self.cursor, "n_ticks": self.n_ticks, "interval": self.interval,
                "events": t["events"], "predictions": t["predictions"], "alerts": t["alerts"],
                "mae": t["abs_err"] / n, "rmse": (t["sq_err"] / n) ** 0.5,
                "precision": t["tp"] / flagged if flagged else None,
                "recall": t["tp"] / actual_pos if actual_pos else None,
                "series": list(self.series), "latest": list(self.latest),
                "city_hist": {c: list(h) for c, h in self.city_hist.items()},
                "latency_ms": self.latency_ms,
            }

    # --------------------------------------------------------------- control
    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            if self.finished or self.cursor >= self.n_ticks:
                self._reset_state()
            self.running = True
            self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        with self.lock:
            self.running = False
        self._stop.set()

    def reset(self) -> None:
        self.pause()
        if self._thread:
            self._thread.join(timeout=5)
        with self.lock:
            self._reset_state()
        for name in ("stream_events", "predictions", "alerts", "stream_metrics"):
            self.store.clear(name)

    def set_speed(self, label: str) -> None:
        self.interval = SPEEDS.get(label, 1.0)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                if self.cursor >= self.n_ticks:
                    self.running, self.finished = False, True
                    return
            t0 = time.time()
            try:
                self._process_tick()
            except Exception as exc:                             # noqa: BLE001
                with self.lock:
                    self.error = f"{type(exc).__name__}: {str(exc)[:140]}"
            self._stop.wait(max(0.0, self.interval - (time.time() - t0)))

    def _process_tick(self) -> None:
        t0 = time.time()
        i = self.cursor
        ts, rows = self.ticks[i]
        stamp = ts.isoformat()

        pred = self.reg.predict(rows[pipeline.FEATURES])
        prob = self.cls.predict_proba(rows[pipeline.CLS_FEATURES])[:, 1]
        actual = rows[pipeline.TARGET].to_numpy()
        actual_spike = rows["is_spike"].to_numpy().astype(bool)
        flagged = prob >= self.alert_p

        records = rows.to_dict("records")
        events, preds, alerts = [], [], []
        for k, r in enumerate(records):
            uid = f"{i:04d}-{r['city']}"
            events.append({"_id": f"evt-{uid}", "tick": i, "timestamp": stamp, "city": r["city"],
                           "actual_mwh": float(actual[k]), "temperature_c": float(r["temperature_c"]),
                           "hour": int(r["hour"]), "source": "smart_city_energy_timeseries.csv"})
            preds.append({"_id": f"prd-{uid}", "tick": i, "timestamp": stamp, "city": r["city"],
                          "predicted_mwh": round(float(pred[k]), 2),
                          "spike_probability": round(float(prob[k]), 4),
                          "error_mwh": round(float(pred[k] - actual[k]), 2),
                          "model": self.reg_name, "classifier": self.cls_name})
            if flagged[k]:
                alerts.append({"_id": f"alt-{uid}", "tick": i, "timestamp": stamp, "city": r["city"],
                               "spike_probability": round(float(prob[k]), 4),
                               "predicted_mwh": round(float(pred[k]), 2),
                               "threshold_mwh": float(r["spike_threshold"]),
                               "severity": "HIGH" if prob[k] >= 0.85 else "MEDIUM",
                               "confirmed": bool(actual_spike[k])})

        err = pred - actual
        metrics_doc = {"_id": f"tick-{i:04d}", "tick": i, "timestamp": stamp,
                       "actual_total_mwh": round(float(actual.sum()), 2),
                       "predicted_total_mwh": round(float(pred.sum()), 2),
                       "mae_mwh": round(float(np.abs(err).mean()), 3),
                       "alerts": len(alerts), "spikes_actual": int(actual_spike.sum())}

        self.store.upsert_many("stream_events", events)
        self.store.upsert_many("predictions", preds)
        self.store.upsert_many("alerts", alerts)
        self.store.upsert_many("stream_metrics", [metrics_doc])

        with self.lock:
            t = self.totals
            t["events"] += len(events)
            t["predictions"] += len(preds)
            t["alerts"] += len(alerts)
            t["confirmed"] += int((flagged & actual_spike).sum())
            t["abs_err"] += float(np.abs(err).sum())
            t["sq_err"] += float((err ** 2).sum())
            t["n"] += len(err)
            t["tp"] += int((flagged & actual_spike).sum())
            t["fp"] += int((flagged & ~actual_spike).sum())
            t["fn"] += int((~flagged & actual_spike).sum())
            t["tn"] += int((~flagged & ~actual_spike).sum())
            self.series.append({"tick": i, "timestamp": stamp,
                                "actual": metrics_doc["actual_total_mwh"],
                                "predicted": metrics_doc["predicted_total_mwh"],
                                "temperature": round(float(rows["temperature_c"].mean()), 2),
                                "alerts": len(alerts)})
            for k, r in enumerate(records):
                h = self.city_hist.setdefault(r["city"], deque(maxlen=HISTORY_TICKS))
                h.append({"tick": i, "timestamp": stamp, "actual": float(actual[k]),
                          "predicted": round(float(pred[k]), 2), "prob": round(float(prob[k]), 3)})
            self.latest = [{"city": r["city"], "actual": float(actual[k]),
                            "predicted": round(float(pred[k]), 2),
                            "prob": round(float(prob[k]), 3), "flag": bool(flagged[k])}
                           for k, r in enumerate(records)]
            self.cursor += 1
            self.latency_ms = (time.time() - t0) * 1000
