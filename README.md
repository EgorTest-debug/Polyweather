# 🌤 PolyWeather

Weather trading bot for Polymarket. Uses ensemble weather models (ECMWF 51-member, GFS 31-member) to find mispriced temperature markets and trade them with limit orders (0% maker fee).

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd polyweather
pip install -r requirements.txt

# 2. Test forecasts (no money needed)
python bot.py --test-forecast

# 3. Run paper mode (no real trades)
python bot.py --once          # single scan
python bot.py                 # continuous loop (30 min interval)

# 4. Go live (after paper testing)
cp .env.example .env
# Edit .env with your Polymarket credentials
python bot.py --live --once   # single live scan
python bot.py --live          # continuous live trading
```

## How It Works

```
Open-Meteo Ensemble API          Polymarket Gamma + CLOB API
  (ECMWF 51 / GFS 31)              (market prices + orderbook)
         │                                    │
         ▼                                    ▼
   Ensemble Probability    ───vs───    Market Price
   P(T > threshold) =                  best ask from
   count(above) / members              real orderbook
         │                                    │
         └──────────────┬─────────────────────┘
                        ▼
                 Edge Detection
              edge = model_prob - price
              filters: ≥10% edge, ≥70% agreement,
              spread ≤3¢, volume ≥$500
                        │
                        ▼
                  Kelly Sizing
              15% fractional Kelly
              cap $35 / 5% balance
                        │
                        ▼
                 CLOB Executor
              LIMIT orders only (0% fee)
              GTC via py-clob-client
                        │
                        ▼
               Position Manager
              stop-loss 20% / trailing stop
              take-profit 75-85¢
              forecast-change exit
```

## Architecture

| File | Layer | What it does |
|------|-------|--------------|
| `cities.py` | Config | 11 cities, airport coordinates, model per region |
| `forecast.py` | Layer 1-2 | Ensemble data + probability engine |
| `markets.py` | Layer 3 | Polymarket scanner + CLOB depth check |
| `strategy.py` | Layer 4-7 | Edge detection, Kelly sizing, exits, risk |
| `executor.py` | Layer 5 | CLOB order placement (paper + live) |
| `bot.py` | Main | Scan loop, ties everything together |

## City Selection

We deliberately pick "second tier" cities — decent liquidity ($30-70K/day) but fewer competing bots than NYC/London/Chicago.

| City | Model | Members | Why |
|------|-------|---------|-----|
| Seoul | ECMWF IFS | 51 | Asia, moderate competition |
| Tokyo | ECMWF IFS | 51 | High volume, ECMWF excels here |
| Singapore | ECMWF IFS | 51 | Tropical, less volatile = consistent |
| Dubai | ECMWF IFS | 51 | Desert = predictable, less bot activity |
| Tel Aviv | ECMWF IFS | 51 | Unique region, few competitors |
| Helsinki | ECMWF IFS | 51 | Nordic, good ECMWF coverage |
| Ankara | ECMWF IFS | 51 | Low competition |
| Toronto | GFS | 31 | N. America = GFS home turf |
| São Paulo | ECMWF IFS | 51 | S. America, underserved |
| Buenos Aires | ECMWF IFS | 51 | S. hemisphere diversity |
| Wellington | ECMWF IFS | 51 | Oceania, very low competition |

## Entry Filters

All must pass before a trade:
- Edge ≥ 10% (model probability − market price)
- Ensemble agreement ≥ 70% (36/51 ECMWF or 22/31 GFS)
- Entry price ≤ 45¢
- Bid-ask spread ≤ 3¢
- Market volume ≥ $500
- 2h ≤ time to resolution ≤ 72h

## Risk Management

- Max $35 per trade (5% of starting $700)
- Max 5 simultaneous positions
- Max 3 positions per city
- Daily loss limit: $35 → kill switch
- Monthly drawdown stop: -15% → full stop
- 15% fractional Kelly sizing

## Going Live

1. Create Polymarket account (email login recommended)
2. Get your private key: https://reveal.magic.link/polymarket
3. Get your wallet address from Polymarket Settings
4. Copy `.env.example` to `.env` and fill in your credentials
5. Start with `--live --once` for a single scan first
6. Monitor for a few days before continuous mode

## Commands

```bash
python bot.py                    # paper mode, continuous
python bot.py --live             # live mode, continuous
python bot.py --once             # single scan
python bot.py --status           # show balance + positions
python bot.py --test-forecast    # test ensemble engine
python bot.py --interval 15      # scan every 15 min (default 30)
```

## Data Sources (all free)

- **ECMWF IFS ensemble** via Open-Meteo — 51 members, 9km resolution
- **GFS ensemble** via Open-Meteo — 31 members, 28km resolution
- **METAR observations** via aviationweather.gov — real-time airport temps
- **Polymarket Gamma API** — market metadata (no auth)
- **Polymarket CLOB API** — real orderbook (auth for trading)
