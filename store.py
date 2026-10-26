"""
NoSQL document store for the Smart City pipeline.

Primary backend : MongoDB - a local server (e.g. Compass at mongodb://127.0.0.1:27017)
                  or Atlas.  Set the MONGODB_URI environment variable.
Fallback backend: local JSONL files in artifacts/local_store/ - used automatically
                  when no URI is set or the cluster cannot be reached, so the demo
                  never stops.  `Store.backend` tells the dashboard which one is live.

Collections
    stream_events  one document per city per streamed hour (actual + forecast)
    predictions    one document per city per streamed hour (forecast, spike risk)
    alerts         spike alerts raised by the classifier
    stream_metrics one document per streamed hour (city totals, running error)
    batch_summary  batch-layer aggregates (heatmap, city table, daily totals)
    model_metrics  model comparison table and feature importance

Every document carries a unique `_id`, so writes are idempotent upserts.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DIR = ROOT / "artifacts" / "local_store"
DB_NAME = os.environ.get("MONGODB_DB", "smart_city")
COLLECTIONS = ("stream_events", "predictions", "alerts", "stream_metrics",
               "batch_summary", "model_metrics")


class Store:
    def __init__(self, client=None, uri: str | None = None):
        self.error = ""
        self._lock = threading.RLock()
        self._db = None
        self._mem: dict[str, dict] = {}
        uri = uri or os.environ.get("MONGODB_URI", "").strip()
        self.uri = uri
        self.host = redacted_host(uri) if uri else ""

        if client is None and uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
            except Exception as exc:                          # noqa: BLE001 - any failure -> fallback
                self.error = f"{type(exc).__name__}: {str(exc)[:160]}"
                client = None
        elif client is None:
            self.error = "MONGODB_URI is not set"

        if client is not None:
            self.backend = "mongodb"
            self._db = client[DB_NAME]
            for name in ("stream_events", "predictions", "alerts", "stream_metrics"):
                self._db[name].create_index([("tick", -1)])
        else:
            self.backend = "local"
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            for name in COLLECTIONS:
                self._mem[name] = self._load_local(name)

    # ------------------------------------------------------------------ local
    def _path(self, name: str) -> Path:
        return LOCAL_DIR / f"{name}.jsonl"

    def _load_local(self, name: str) -> dict:
        docs: dict = {}
        p = self._path(name)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    docs[d["_id"]] = d
        return docs

    def _flush_local(self, name: str) -> None:
        text = "".join(json.dumps(d, separators=(",", ":")) + "\n" for d in self._mem[name].values())
        self._path(name).write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ writes
    def upsert_many(self, name: str, docs: list[dict]) -> int:
        if not docs:
            return 0
        if self.backend == "mongodb":
            coll = self._db[name]
            try:
                # atomic per-document upsert: idempotent even with concurrent writers
                from pymongo import ReplaceOne
                coll.bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs], ordered=False)
            except TypeError:
                # mongomock (offline self-test) predates this pymongo's ReplaceOne signature
                coll.delete_many({"_id": {"$in": [d["_id"] for d in docs]}})
                coll.insert_many(docs, ordered=False)
        else:
            with self._lock:
                mem = self._mem[name]
                all_new = all(d["_id"] not in mem for d in docs)
                for d in docs:
                    mem[d["_id"]] = d
                if all_new:                       # common case while streaming: append only
                    with self._path(name).open("a", encoding="utf-8") as f:
                        f.writelines(json.dumps(d, separators=(",", ":")) + "\n" for d in docs)
                else:
                    self._flush_local(name)
        return len(docs)

    def replace_all(self, name: str, docs: list[dict]) -> None:
        self.clear(name)
        self.upsert_many(name, docs)

    def clear(self, name: str) -> None:
        if self.backend == "mongodb":
            self._db[name].delete_many({})
        else:
            with self._lock:
                self._mem[name] = {}
                self._flush_local(name)

    # ------------------------------------------------------------------ reads
    def count(self, name: str) -> int:
        if self.backend == "mongodb":
            return self._db[name].count_documents({})
        return len(self._mem[name])

    def find(self, name: str, query: dict | None = None, sort: str | None = None,
             descending: bool = True, limit: int = 0) -> list[dict]:
        query = query or {}
        if self.backend == "mongodb":
            cur = self._db[name].find(query)
            if sort:
                cur = cur.sort(sort, -1 if descending else 1)
            if limit:
                cur = cur.limit(limit)
            return list(cur)
        with self._lock:
            docs = [d for d in self._mem[name].values()
                    if all(d.get(k) == v for k, v in query.items())]
        if sort:
            docs.sort(key=lambda d: d.get(sort, 0), reverse=descending)
        return docs[:limit] if limit else docs

    def get(self, name: str, doc_id: str) -> dict | None:
        if self.backend == "mongodb":
            return self._db[name].find_one({"_id": doc_id})
        return self._mem[name].get(doc_id)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        counts = {n: self.count(n) for n in COLLECTIONS}
        if self.backend == "mongodb":
            label = server_label(self.uri)
            detail = f"database '{DB_NAME}' on {self.host}"
        else:
            label, detail = "Local files (no database connected)", self.error
        return {"backend": self.backend, "label": label, "detail": detail,
                "counts": counts}


def server_label(uri: str) -> str:
    """Human name for the server a URI points at. Never includes credentials."""
    u = uri.lower()
    if "mongodb.net" in u or u.startswith("mongodb+srv://"):
        return "MongoDB Atlas"
    if "127.0.0.1" in u or "localhost" in u:
        return "MongoDB (local server)"
    return "MongoDB"


def redacted_host(uri: str) -> str:
    """host[:port] of a URI with any user:password removed."""
    rest = uri.split("://", 1)[-1]
    return rest.rsplit("@", 1)[-1].split("/", 1)[0].split("?", 1)[0] or "unknown host"
