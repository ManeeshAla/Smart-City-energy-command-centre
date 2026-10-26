# Smart City Energy Command Center

An end-to-end smart-city energy analytics system: it ingests hourly city data,
runs a batch analytics layer, trains machine-learning models to forecast demand
and flag consumption spikes, streams held-out days through those models in real
time, stores every event in a NoSQL document store, and visualises the whole
pipeline in a single live dashboard.

```
CSV ingestion ─► Batch layer (Parquet + aggregates)
                     │
                     ▼
            ML training (scikit-learn)          Spark MLlib cross-check
          energy forecast + spike classifier    (spark_mllib_train.py)
                     │
                     ▼
         Stream replay (held-out 20 days, hour by hour)
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
   NoSQL store              Live dashboard
 (MongoDB local or        (Plotly Dash,
  Atlas / fallback)         auto-refresh)
```

## Pipeline stages

| Stage | File | What it does |
|-------|------|--------------|
| Ingestion + data build | `generate_data.py` | Builds the hourly city energy dataset (40 cities × 90 days = 85,440 rows) from `smart_city_dataset.csv`. |
| Batch + ML | `pipeline.py` | Parquet snapshot, batch aggregates, trains the forecast and spike models, publishes metrics. |
| NoSQL store | `store.py` | MongoDB document store (local server or Atlas) with an automatic local-file fallback. |
| Streaming | `stream.py` | Replays the held-out 20 days one hour at a time: predict → store → alert. |
| Dashboard | `dashboard.py` | The single live Plotly Dash dashboard (4 views, alerts, CSV export). |
| Big-data model | `spark_mllib_train.py` | Trains the same models with Spark MLlib for comparison. |

## Quick start

1. Install Python 3.10+ and the dependencies:

   ```
   python -m pip install -r requirements.txt
   ```

2. Build the models, then launch the dashboard. The hourly dataset
   (`smart_city_energy_timeseries.csv`) is already included; `generate_data.py`
   rebuilds the identical file from `smart_city_dataset.csv` if you ever need it:

   ```
   python generate_data.py
   python pipeline.py
   python dashboard.py
   ```

   Or just double-click **`run.bat`** (it runs all three steps the first time).

3. Open <http://127.0.0.1:8050> and press **▶ Start live demo**. Use the speed
   toggle (1×/2×/5×), **Pause**, and **Reset** to drive the stream.

## Connecting MongoDB (recommended)

By default the app uses a local NoSQL document store (JSON files in
`artifacts/local_store/`), so it runs with no setup. To use a real MongoDB
server, set the `MONGODB_URI` environment variable **before** starting the app.
Credentials never go in the code.

**Option A — local MongoDB (the server MongoDB Compass connects to):**

```
# PowerShell, current terminal only
$env:MONGODB_URI = "mongodb://127.0.0.1:27017"
python check_mongo.py        # verifies the connection
python pipeline.py
python dashboard.py
```

Open **Compass**, connect to `mongodb://127.0.0.1:27017`, and watch the new
`smart_city` database fill up while the demo runs (press refresh in Compass).
Only the `smart_city` database is touched; other databases are left alone.

**Option B — MongoDB Atlas:** create a free M0 cluster, add your IP under
*Network Access*, and set `MONGODB_URI` to the `mongodb+srv://…` connection
string (`setx` makes it permanent for new terminals).

The top-right badge shows **“MongoDB (local server) connected”** (or
**“MongoDB Atlas connected”**) when live, and **“Local NoSQL store”** otherwise.
Collections: `stream_events`, `predictions`, `alerts`, `stream_metrics`,
`batch_summary`, `model_metrics`.

## Dashboard features

| Feature | What it does |
|---------|--------------|
| **Live / Analytics / System / Presentation** views | Instant switching (no server call). Presentation shows only the stepper, KPIs, live chart and spike risk, in large type for a projector. |
| **Keyboard shortcuts** | `Space` start / pause, `1`-`4` switch view. |
| **Alert toasts** | A card slides in when a city reaches HIGH spike risk (85%+). |
| **Download predictions.csv** | One click exports every forecast streamed so far (hour, city, actual, forecast, error, spike probability, alert). |
| **Smooth updates** | Nothing is redrawn when nothing changed (0 KB while idle or paused), and zoom/hover state is kept between updates. |

Set `PORT` to run on another port, e.g. `$env:PORT = "8060"`.

## Results (held-out last 20 days)

| Model | MAE (MWh) | RMSE | R² |
|-------|-----------|------|-----|
| Mean baseline | 52.9 | 63.6 | 0.00 |
| Persistence (last hour) | 6.96 | 10.8 | 0.971 |
| Ridge regression | 5.54 | 8.59 | 0.982 |
| Random forest | 2.73 | 5.67 | 0.992 |
| **Gradient boosting (selected)** | **2.62** | **5.22** | **0.993** |

Spike classifier (gradient boosting): precision 0.87, recall 0.87, F1 0.87, ROC-AUC 0.99.
Spark MLlib GBT regressor: MAE 3.12, R² 0.992 — consistent with scikit-learn.

## Cloud mapping

The project runs locally. Each component maps to a managed cloud service if
deployed (AWS / Azure / GCP):

| Local component | AWS | Azure | GCP |
|-----------------|-----|-------|-----|
| Parquet batch store | S3 | Blob Storage | Cloud Storage |
| Stream replay | Kinesis | Event Hubs | Pub/Sub |
| Spark MLlib job | EMR | Synapse / HDInsight | Dataproc |
| NoSQL store | DocumentDB | Cosmos DB | Firestore |
| Dashboard host | Elastic Beanstalk | App Service | Cloud Run |

Pointing `MONGODB_URI` at an Atlas cluster moves the storage layer to the cloud
with no code change.

## Files

| Path | Purpose |
|------|---------|
| `smart_city_dataset.csv` | Source city attributes (supplied, unchanged). |
| `smart_city_energy_timeseries.csv` | Generated hourly dataset. |
| `assets/` | Dashboard theme and keyboard shortcuts (auto-loaded by Dash). |
| `artifacts/` | Generated models, metrics, Parquet, local NoSQL store. |
| `legacy/` | Earlier versions, kept for reference. |
| `DATA_SOURCE.md` | How the hourly dataset is built. |
| `Experiment10_Smart_City_Energy_Report.docx` | Full project report. |
