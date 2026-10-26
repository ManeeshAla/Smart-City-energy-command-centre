# Data sources and how the hourly dataset is built

## Source file

`smart_city_dataset.csv` provides the static attributes of each city
(population, area, smart-grid adoption, infrastructure score, public-transport
usage, internet speed, and others). It is used unchanged.

## Hourly energy time series

`generate_data.py` builds `smart_city_energy_timeseries.csv`, the hourly dataset
the models learn from. For 40 cities (selected across the population range,
500k residents and up) it produces 90 days of hourly readings — 85,440 rows.

For each city-hour it derives the hourly energy demand (MWh) as a function of:

- **City size** — a base load scaled by population and area.
- **Smart-grid adoption** — higher adoption slightly lowers consumption.
- **Daily rhythm** — morning and evening demand peaks.
- **Weekday vs weekend** — lower demand at weekends.
- **Temperature** — higher cooling load in hot weather, with modelled heat
  waves; small heating load when cold.
- **Short-term dynamics** — correlated hour-to-hour noise and occasional grid
  stress events.

It also records the inputs a real forecasting system would use: `temperature_c`,
`hour`, `dayofweek`, `is_weekend`, and the lagged/rolling demand features
(`lag_1h`, `lag_24h`, `rolling_24h_mean`).

A **spike** label (`is_spike`) marks hours where a city's demand exceeds its own
90th percentile over the training period — the event the classifier learns to
anticipate.

## Train/test split

The split is chronological, matching how a deployed forecaster is evaluated:
the first 70 days train the models and the last 20 days are held out. The live
dashboard streams those held-out days hour by hour, so every forecast shown is
on data the models were not trained on.

The random generator is seeded, so every run reproduces the same dataset,
metrics, and figures.
