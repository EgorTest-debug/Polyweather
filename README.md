# 🌤 PolyWeather

Multi-model ensemble postprocessing for daily maximum temperature forecasts at 10 globally distributed airport stations. Uses LightGBM regression with ensemble member shifting to produce calibrated bucket probability predictions.

This repository contains the code and dataset for the NLP Course final project (Spring 2026).

## How It Works

```
11 NWP Models (Previous Runs API)             ICON Ensemble (39 members)
  ECMWF, ICON, GFS, JMA, GEM,                  raw probability distribution
  UKMO, KNMI, DMI, CMA, MeteoFrance,                       │
  NCEP GraphCast                                           │
         │                                                 │
         ▼                                                 │
   LightGBM Regressor                                      │
   (8,339 samples, 27 features)                            │
   → calibrated T_max prediction                           │
         │                                                 │
         └─────────── shift ensemble mean ─────────────────┘
                              │
                              ▼
              P(bucket) = shifted members in bucket / N
```

## Quick Start

```bash
git clone <repo-url>
cd polyweather
pip install -r requirements.txt

# Collect data and train model from scratch
python retrain.py --from-scratch

# Or use the included pretrained model
python bot.py --test-forecast

# Single scan / continuous run
python bot.py --once
python bot.py --interval 60
```

## Repository Structure

| File | What it does |
|------|-------------|
| `cities.py` | City configs — coordinates, stations, timezones |
| `forecast.py` | Ensemble fetch + LightGBM calibration |
| `predictor.py` | LightGBM inference — fetches 11-model features, predicts T_max |
| `markets.py` | Polymarket market scanner |
| `strategy.py` | Edge detection, position sizing, exits |
| `executor.py` | Order execution |
| `bot.py` | Main scan loop |
| `retrain.py` | Automated data collection + model retraining |

## Dataset

The dataset is included in `data/`:

| File | Description | Rows |
|------|-------------|------|
| `data/model_forecasts_day1.csv` | 24-hour-ahead forecasts from 11 NWP models + 7 context variables | 8,340 |
| `data/wu_historical.csv` | Daily T_max from Weather Underground (ground truth) | 11,138 |
| `data/clim_normals_era5.csv` | ERA5 1991-2020 climatological normals per city/day | 3,660 |
| `data/models/global_model_tuned.txt` | Trained LightGBM model (native format) | — |
| `data/models/feature_cols.pkl` | Feature column list | — |
| `data/models/best_params.json` | Optimal hyperparameters (Optuna, 80 trials) | — |

**Coverage:** 10 airport stations, Feb 5, 2024 — May 18, 2026 (~28 months).

**Data collection:** Run `python retrain.py --from-scratch` to reproduce the dataset from public APIs.

## Model

LightGBM regression trained on 27 features:
- **NWP forecasts (11):** ECMWF IFS, GFS, ICON, JMA, GEM, MeteoFrance, UKMO, KNMI, DMI, CMA GRAPES, NCEP GraphCast
- **Context (7):** Cloud cover, precipitation, radiation, humidity, wind speed, pressure, dew point (ECMWF IFS, 24h ahead)
- **Seasonal (3):** doy_sin, doy_cos, city_id
- **History (2):** WU T_max d-2 and d-3
- **Ensemble stats (4):** mean, std, min, max across NWP models

**Hyperparameters** (Optuna tuned): 313 estimators, lr=0.029, max_depth=6, num_leaves=35.

**Honest walk-forward CV bucket accuracy: 37.9%** (1°C discretization).

## Per-city Performance

| City | Bucket Accuracy | MAE (°C) |
|------|----------------|----------|
| Wellington | 43.8% | 0.75 |
| Helsinki | 42.2% | 0.92 |
| Ankara | 41.8% | 0.81 |
| Singapore | 41.6% | 0.77 |
| Tel Aviv | 41.4% | 0.78 |
| Buenos Aires | 35.6% | 0.99 |
| Seoul | 34.6% | 1.00 |
| Toronto | 33.3% | 1.13 |
| Tokyo | 32.8% | 0.98 |
| São Paulo | 32.0% | 1.04 |

## Data Sources

- **NWP forecasts:** [Open-Meteo Previous Runs API](https://previous-runs-api.open-meteo.com/) — `temperature_2m_previous_day1` variable for 24h-ahead forecasts
- **Ground truth:** Weather Company V1 API — daily T_max at airport ICAO stations
- **Climatology:** [Open-Meteo Climate API](https://climate-api.open-meteo.com/) — ERA5 1991-2020 normals
- **Markets:** Polymarket Gamma API + CLOB `/price` endpoint (downstream application)

## Notes

This project explores statistical postprocessing of ensemble weather forecasts — a well-established field in meteorology. The downstream application is automated trading on Polymarket temperature prediction markets, but the methodology applies to any setting requiring calibrated short-term temperature forecasts (renewable energy, agriculture, climate risk assessment).