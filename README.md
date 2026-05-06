# 🌤 PolyWeather

Weather trading bot for Polymarket. Uses ECMWF 50-member ensemble + LightGBM calibration to find mispriced temperature markets.

**Status:** Paper trading / pre-production

## How It Works

```
11 Weather Models (Previous Runs API)        ECMWF Ensemble (50 members)
  ECMWF, ICON, GFS, JMA, GEM...               raw probability distribution
         │                                              │
         ▼                                              │
   LightGBM Calibrator                                  │
   (trained on 2 years WU data)                         │
   → calibrated T_max prediction                        │
         │                                              │
         └──────── shift ensemble mean ─────────────────┘
                        │
                        ▼
              P(bucket) = shifted members in bucket / 50
                        │
                        ▼
              Gamma + CLOB price → edge = P(bucket) - ask
                        │
                        ▼
              Kelly sizing → paper/live order
```

## Quick Start

```bash
git clone <your-repo-url>
cd polyweather
pip install -r requirements.txt

# Test forecast engine
python bot.py --test-forecast

# Paper mode
python bot.py --once           # single scan
python bot.py --interval 60    # continuous (hourly)

# Check results
python bot.py --status

# Live mode (after paper testing)
cp .env.example .env
python bot.py --live --once
python bot.py --live
```

## Architecture

| File | What it does |
|------|-------------|
| `cities.py` | City configs — coordinates, stations, timezones |
| `forecast.py` | ECMWF ensemble fetch + LightGBM shift |
| `predictor.py` | LightGBM calibrator — fetches 11-model features, predicts T_max |
| `markets.py` | Polymarket scanner + CLOB price check |
| `strategy.py` | Edge detection, Kelly sizing, per-city limits, exits |
| `executor.py` | Order placement (paper + live) |
| `bot.py` | Main scan loop |

## Model

LightGBM trained on 2 years of historical data (Feb 2024 — Apr 2026):
- **Input:** 11 weather model forecasts + 7 context variables (cloud cover, precipitation, radiation, humidity, wind, pressure, dew point) + seasonality + inter-model statistics
- **Target:** Wunderground daily T_max (same source as Polymarket resolution)
- **CV bucket accuracy:** 45–63% by city (vs ECMWF raw: 20–38%)

## Cities & Limits

| City | CV Accuracy | Max Entry |
|------|------------|-----------|
| Tel Aviv | 62.6% | 50¢ |
| Helsinki | 60.4% | 50¢ |
| Singapore | 60.0% | 50¢ |
| Wellington | 59.7% | 50¢ |
| Ankara | 57.5% | 40¢ |
| Toronto | 55.3% | 40¢ |
| Seoul | 54.9% | 40¢ |
| Buenos Aires | 54.6% | 40¢ |
| Sao Paulo | 50.2% | 35¢ |
| Tokyo | 44.9% | 25¢ |

## Entry Filters

- Edge ≥ 10% (P(bucket) − CLOB ask)
- Ensemble agreement ≥ 55%
- Per-city max entry price (see table above)
- Volume ≥ $200
- Spread ≤ 3¢
- 12h ≤ time to resolution ≤ 36h (D+1 only)
- 1 position per city per day

## Risk

- 15% fractional Kelly sizing
- Max $35 per trade, max 5% of balance
- Max 8 simultaneous positions
- Daily loss limit: $50
- Drawdown stop: -30% from peak
- Exits: stop-loss 20%, auto-resolution via Gamma API

## Data Sources

- **ECMWF 50-member ensemble** — Open-Meteo ensemble API
- **11 model forecasts** — Open-Meteo Previous Runs API (D+1)
- **Wunderground T_max** — Weather Company v1 API (ground truth)
- **Polymarket markets** — Gamma API (discovery) + CLOB `/price` (real ask)