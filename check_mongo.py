#!/usr/bin/env python3
"""
Verify the MongoDB connection used by the pipeline before a demo.

    $env:MONGODB_URI = "mongodb://127.0.0.1:27017"     (PowerShell)
    python check_mongo.py

Pings the server, writes / reads / deletes one test document in
smart_city.healthcheck, and prints the document count of every pipeline
collection.  Other databases on the server are never touched.
"""
import sys

from store import DB_NAME, Store

store = Store()
status = store.status()

print(f"Backend : {status['label']}")
print(f"Detail  : {status['detail'] or 'ok'}")

if store.backend != "mongodb":
    print("\nNot connected to MongoDB - the app will use local files.")
    print("Set MONGODB_URI (e.g. mongodb://127.0.0.1:27017) and make sure the server is running.")
    sys.exit(1)

db = store._db
doc = {"_id": "ping", "ok": True}
db["healthcheck"].replace_one({"_id": "ping"}, doc, upsert=True)
assert db["healthcheck"].find_one({"_id": "ping"}) == doc, "read-back failed"
db["healthcheck"].delete_one({"_id": "ping"})
print(f"\nWrite / read / delete test passed in database '{DB_NAME}'.")

print("\nCollection counts:")
for name, n in status["counts"].items():
    print(f"  {name:<15} {n:>8,}")
print("\nOpen MongoDB Compass -> connect to the same address -> database", f"'{DB_NAME}'.")
