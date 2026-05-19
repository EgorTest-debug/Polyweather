#!/usr/bin/env python3
"""
PolyWeather — retrain.py

Collects missing forecast + WU data and retrains the LightGBM model.

Usage:
    python retrain.py              # collect missing days + retrain
    python retrain.py --from-scratch  # collect everything from scratch
    python retrain.py --collect-only  # only collect data, no training
    python retrain.py --train-only    # only train on existing data
"""

import argparse
import json
import logging
import math
import pickle
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
MODELS_DIR = DATA_DIR / "models"
LOG_DIR    = Path("logs")

FORECASTS_FILE  = DATA_DIR / "model_forecasts_day1.csv"
WU_FILE         = DATA_DIR / "wu_historical.csv"
MODEL_FILE      = MODELS_DIR / "global_model_tuned.txt"
FEATURES_FILE   = MODELS_DIR / "feature_cols.pkl"
PARAMS_FILE     = MODELS_DIR / "best_params.json"
RETRAIN_LOG     = LOG_DIR / "retrain_log.json"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("retrain")

# ─── Config ───────────────────────────────────────────────────────────────────

START_DATE = "2024-02-05"
WU_START_DATE = "2023-05-01"

CITIES = {
    "seoul":        {"lat": 37.4691,  "lon": 126.4505, "tz": "Asia/Seoul",                          "station": ("RKSI", "KR")},
    "tokyo":        {"lat": 35.7647,  "lon": 140.3864, "tz": "Asia/Tokyo",                          "station": ("RJTT", "JP")},
    "singapore":    {"lat": 1.3502,   "lon": 103.9940, "tz": "Asia/Singapore",                      "station": ("WSSS", "SG")},
    "tel-aviv":     {"lat": 32.0114,  "lon": 34.8867,  "tz": "Asia/Jerusalem",                      "station": ("LLBG", "IL")},
    "helsinki":     {"lat": 60.3172,  "lon": 24.9633,  "tz": "Europe/Helsinki",                     "station": ("EFHK", "FI")},
    "ankara":       {"lat": 40.1281,  "lon": 32.9951,  "tz": "Europe/Istanbul",                     "station": ("LTAC", "TR")},
    "toronto":      {"lat": 43.6772,  "lon": -79.6306, "tz": "America/Toronto",                     "station": ("CYYZ", "CA")},
    "sao-paulo":    {"lat": -23.4356, "lon": -46.4731, "tz": "America/Sao_Paulo",                   "station": ("SBGR", "BR")},
    "buenos-aires": {"lat": -34.8222, "lon": -58.5358, "tz": "America/Argentina/Buenos_Aires",      "station": ("SAEZ", "AR")},
    "wellington":   {"lat": -41.3272, "lon": 174.8052, "tz": "Pacific/Auckland",                    "station": ("NZWN", "NZ")},
}

MODELS = [
    "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_seamless",
    "gem_seamless", "meteofrance_seamless", "ukmo_seamless",
    "knmi_seamless", "dmi_seamless", "cma_grapes_global", "ncep_gfs_graphcast025",
]

HOURLY_CONTEXT = {
    "cloudcover_mean":           "cloud_cover",
    "precipitation_sum":         "precipitation",
    "shortwave_radiation_sum":   "shortwave_radiation",
    "relative_humidity_2m_mean": "relative_humidity_2m",
    "windspeed_10m_max":         "wind_speed_10m",
    "pressure_msl_mean":         "pressure_msl",
    "dew_point_2m_mean":         "dew_point_2m",
}

FEATURE_COLS = [
    "ecmwf_ifs025_tmax", "gfs_seamless_tmax", "icon_seamless_tmax",
    "jma_seamless_tmax", "gem_seamless_tmax", "meteofrance_seamless_tmax",
    "ukmo_seamless_tmax", "knmi_seamless_tmax", "dmi_seamless_tmax",
    "cma_grapes_global_tmax", "ncep_gfs_graphcast025_tmax",
    "cloudcover_mean", "precipitation_sum", "shortwave_radiation_sum",
    "relative_humidity_2m_mean", "windspeed_10m_max",
    "pressure_msl_mean", "dew_point_2m_mean",
    "wu_max_d_minus_2", "wu_max_d_minus_3",
    "doy_sin", "doy_cos", "city_id",
    "model_mean", "model_std", "model_min", "model_max",
]

PREV_RUNS_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

# ─── Data collection ──────────────────────────────────────────────────────────

def collect_forecasts(start: str, end: str) -> pd.DataFrame:
    """Collect previous_day1 forecasts for all cities and models."""
    log.info(f"Collecting forecasts: {start} → {end}")
    all_rows = []

    for city, cfg in CITIES.items():
        log.info(f"  {city}...")

        # 1. All 11 models tmax in one request
        try:
            r = requests.get(PREV_RUNS_BASE, params={
                "latitude": cfg["lat"], "longitude": cfg["lon"],
                "hourly": "temperature_2m_previous_day1",
                "models": ",".join(MODELS),
                "start_date": start, "end_date": end,
                "timezone": cfg["tz"],
            }, timeout=60)
            data = r.json()

            if data.get("error"):
                log.warning(f"    API error for {city}: {data.get('reason')}")
                continue

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            daily_models = {}
            for model in MODELS:
                key = f"temperature_2m_previous_day1_{model}"
                vals = hourly.get(key, [])
                for t, v in zip(times, vals):
                    day = t[:10]
                    if v is not None:
                        if day not in daily_models:
                            daily_models[day] = {}
                        daily_models[day][f"{model}_tmax"] = max(
                            daily_models[day].get(f"{model}_tmax", -999), v
                        )

            log.info(f"    tmax: {len(daily_models)} days")
            time.sleep(1)

        except Exception as e:
            log.error(f"    tmax error for {city}: {e}")
            continue

        # 2. Context variables from ecmwf
        try:
            ctx_vars = [f"{v}_previous_day1" for v in HOURLY_CONTEXT.values()]
            r2 = requests.get(PREV_RUNS_BASE, params={
                "latitude": cfg["lat"], "longitude": cfg["lon"],
                "hourly": ",".join(ctx_vars),
                "models": "ecmwf_ifs025",
                "start_date": start, "end_date": end,
                "timezone": cfg["tz"],
            }, timeout=60)
            data2 = r2.json()
            hourly2 = data2.get("hourly", {})
            times2 = hourly2.get("time", [])

            daily_ctx = {}
            for ctx_name, hourly_name in HOURLY_CONTEXT.items():
                api_key = f"{hourly_name}_previous_day1_ecmwf_ifs025"
                vals = hourly2.get(api_key, hourly2.get(f"{hourly_name}_previous_day1", []))
                for t, v in zip(times2, vals):
                    day = t[:10]
                    if v is not None:
                        if day not in daily_ctx:
                            daily_ctx[day] = {}
                        if ctx_name not in daily_ctx[day]:
                            daily_ctx[day][ctx_name] = []
                        daily_ctx[day][ctx_name].append(v)

            log.info(f"    context: {len(daily_ctx)} days")
            time.sleep(1)

        except Exception as e:
            log.error(f"    context error for {city}: {e}")
            daily_ctx = {}

        # Merge
        all_dates = sorted(set(daily_models.keys()) | set(daily_ctx.keys()))
        for day in all_dates:
            row = {"city": city, "date": day}
            if day in daily_models:
                for k, v in daily_models[day].items():
                    if v > -999:
                        row[k] = round(v, 1)
            if day in daily_ctx:
                for ctx_name, vals in daily_ctx[day].items():
                    if "sum" in ctx_name:
                        row[ctx_name] = round(sum(vals), 1)
                    elif "max" in ctx_name:
                        row[ctx_name] = round(max(vals), 1)
                    else:
                        row[ctx_name] = round(sum(vals) / len(vals), 1)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def collect_wu(start: str, end: str) -> pd.DataFrame:
    """Collect WU historical observations."""
    log.info(f"Collecting WU: {start} → {end}")
    rows = []

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    for city, cfg in CITIES.items():
        station, country = cfg["station"]
        d = start_d
        while d <= end_d:
            try:
                url = (
                    f"https://api.weather.com/v1/location/{station}:9:{country}"
                    f"/observations/historical.json"
                    f"?apiKey={WU_API_KEY}&units=m"
                    f"&startDate={d.strftime('%Y%m%d')}&endDate={d.strftime('%Y%m%d')}"
                )
                r = requests.get(url, timeout=10)
                obs = r.json().get("observations", [])
                temps = [o.get("temp") for o in obs if o.get("temp") is not None]
                if temps:
                    rows.append({"city": city, "date": d.isoformat(), "wu_max": max(temps)})
            except Exception as e:
                log.debug(f"  {city} {d}: {e}")
            d += timedelta(days=1)
            time.sleep(0.2)

        log.info(f"  {city} WU done")

    df = pd.DataFrame(rows)
    return df


# ─── Dataset management ───────────────────────────────────────────────────────

def load_or_init_forecasts() -> pd.DataFrame:
    if FORECASTS_FILE.exists():
        df = pd.read_csv(FORECASTS_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        log.info(f"Loaded forecasts: {len(df)} rows, up to {df['date'].max()}")
        return df
    log.info("No forecasts file found — will collect from scratch")
    return pd.DataFrame()


def load_or_init_wu() -> pd.DataFrame:
    if WU_FILE.exists():
        wu = pd.read_csv(WU_FILE)
        wu["date"] = pd.to_datetime(wu["date"], format="mixed").dt.date.astype(str)
        log.info(f"Loaded WU: {len(wu)} rows, up to {wu['date'].max()}")
        return wu
    log.info("No WU file found — will collect from scratch")
    return pd.DataFrame()


def update_forecasts(existing: pd.DataFrame, from_scratch: bool) -> pd.DataFrame:
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    if from_scratch or existing.empty:
        start = START_DATE
    else:
        last = existing["date"].max()
        start = (date.fromisoformat(last) + timedelta(days=1)).isoformat()

    if start > yesterday:
        log.info("Forecasts already up to date")
        return existing

    log.info(f"Collecting forecasts {start} → {yesterday}")
    new_df = collect_forecasts(start, yesterday)

    if existing.empty:
        result = new_df
    else:
        result = pd.concat([existing, new_df], ignore_index=True)
        result = result.drop_duplicates(subset=["city", "date"], keep="last")
        result = result.sort_values(["city", "date"]).reset_index(drop=True)

    result.to_csv(FORECASTS_FILE, index=False)
    log.info(f"Saved forecasts: {len(result)} rows")
    return result


def update_wu(existing: pd.DataFrame, from_scratch: bool) -> pd.DataFrame:
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    if from_scratch or existing.empty:
        start = WU_START_DATE
    else:
        last = existing["date"].max()
        start = (date.fromisoformat(last) + timedelta(days=1)).isoformat()

    if start > yesterday:
        log.info("WU already up to date")
        return existing

    log.info(f"Collecting WU {start} → {yesterday}")
    new_wu = collect_wu(start, yesterday)

    if existing.empty:
        result = new_wu
    else:
        result = pd.concat([existing, new_wu], ignore_index=True)
        result = result.drop_duplicates(subset=["city", "date"], keep="last")
        result = result.sort_values(["city", "date"]).reset_index(drop=True)

    result.to_csv(WU_FILE, index=False)
    log.info(f"Saved WU: {len(result)} rows")
    return result


# ─── Feature engineering ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, wu: pd.DataFrame) -> pd.DataFrame:
    wu_clean = wu.copy()
    wu_clean["date"] = wu_clean["date"].astype(str)
    wu_clean["wu_max"] = pd.to_numeric(wu_clean["wu_max"], errors="coerce")

    merged = df.merge(wu_clean[["city", "date", "wu_max"]], on=["city", "date"], how="inner")
    merged = merged.rename(columns={"wu_max": "target"})
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["city", "date"])

    merged["doy"] = merged["date"].dt.dayofyear
    merged["doy_sin"] = merged["doy"].apply(lambda x: math.sin(2 * math.pi * x / 365))
    merged["doy_cos"] = merged["doy"].apply(lambda x: math.cos(2 * math.pi * x / 365))
    merged["city_id"] = merged["city"].astype("category").cat.codes

    merged["wu_max_d_minus_2"] = merged.groupby("city")["target"].shift(2)
    merged["wu_max_d_minus_3"] = merged.groupby("city")["target"].shift(3)

    model_cols = [c for c in merged.columns if c.endswith("_tmax")]
    merged["model_mean"] = merged[model_cols].mean(axis=1)
    merged["model_std"]  = merged[model_cols].std(axis=1)
    merged["model_min"]  = merged[model_cols].min(axis=1)
    merged["model_max"]  = merged[model_cols].max(axis=1)

    return merged


# ─── Walk-forward CV ──────────────────────────────────────────────────────────

def make_wf_folds(df: pd.DataFrame, n_folds: int = 6):
    df = df.sort_values("date")
    min_date = df["date"].min()
    max_date = df["date"].max()
    total_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month
    test_months = max(1, total_months // (n_folds + 1))

    folds = []
    for i in range(n_folds):
        test_end_month = max_date.month - i * test_months
        test_end_year = max_date.year
        while test_end_month <= 0:
            test_end_month += 12
            test_end_year -= 1
        test_end = pd.Timestamp(year=test_end_year, month=test_end_month, day=1) + pd.offsets.MonthEnd(0)
        test_start = test_end - pd.DateOffset(months=test_months) + pd.Timedelta(days=1)
        train_end = test_start - pd.Timedelta(days=1)
        train_start = min_date

        if (train_end - train_start).days < 6 * 30:
            continue

        train_idx = df[(df["date"] >= train_start) & (df["date"] <= train_end)].index
        test_idx  = df[(df["date"] >= test_start)  & (df["date"] <= test_end)].index

        if len(train_idx) > 100 and len(test_idx) > 20:
            folds.append((train_idx, test_idx))

    return folds[::-1]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> dict:
    """Train LightGBM model and return metrics."""
    # Load params
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            params = json.load(f)
        log.info(f"Loaded params from {PARAMS_FILE}")
    else:
        log.warning("No best_params.json found — using defaults")
        params = {
            "n_estimators": 313, "learning_rate": 0.029, "max_depth": 6,
            "min_child_samples": 12, "subsample": 0.69, "colsample_bytree": 0.65,
            "reg_alpha": 0.52, "reg_lambda": 0.99, "num_leaves": 35,
        }

    # CV
    folds = make_wf_folds(df, n_folds=6)
    log.info(f"Running CV with {len(folds)} folds...")

    maes, accs = [], []
    city_results = {city: {"maes": [], "accs": []} for city in df["city"].unique()}

    for train_idx, test_idx in folds:
        X_tr = df.loc[train_idx, FEATURE_COLS]
        y_tr = df.loc[train_idx, "target"].dropna()
        X_tr = X_tr.loc[y_tr.index]
        X_te = df.loc[test_idx, FEATURE_COLS]
        y_te = df.loc[test_idx, "target"].dropna()
        X_te = X_te.loc[y_te.index]

        m = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)

        maes.append(mean_absolute_error(y_te, pred))
        accs.append((y_te.values.round() == pred.round()).mean())

        test_data = df.loc[test_idx].copy().dropna(subset=["target"])
        test_data["pred"] = m.predict(test_data[FEATURE_COLS])
        for city in test_data["city"].unique():
            ct = test_data[test_data["city"] == city]
            city_results[city]["maes"].append(mean_absolute_error(ct["target"], ct["pred"]))
            city_results[city]["accs"].append((ct["target"].values.round() == ct["pred"].values.round()).mean())

    cv_mae = float(np.mean(maes))
    cv_acc = float(np.mean(accs))
    log.info(f"CV MAE: {cv_mae:.3f}°C | CV Accuracy: {cv_acc:.1%}")

    # Per-city metrics
    city_metrics = {}
    for city, res in city_results.items():
        if res["maes"]:
            city_metrics[city] = {
                "mae": round(float(np.mean(res["maes"])), 3),
                "acc": round(float(np.mean(res["accs"])), 3),
            }
            log.info(f"  {city:15s}: MAE={city_metrics[city]['mae']:.3f} Acc={city_metrics[city]['acc']:.1%}")

    # Train final model on all data
    log.info("Training final model on all data...")
    X_all = df.dropna(subset=["target"])[FEATURE_COLS]
    y_all = df.dropna(subset=["target"])["target"]

    final_model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
    final_model.fit(X_all, y_all)

    # Save
    final_model.booster_.save_model(str(MODEL_FILE))
    pickle.dump(FEATURE_COLS, open(FEATURES_FILE, "wb"))
    log.info(f"Model saved to {MODEL_FILE}")

    return {
        "cv_mae": cv_mae,
        "cv_acc": cv_acc,
        "city_metrics": city_metrics,
        "n_train": len(y_all),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
    }


# ─── Logging ──────────────────────────────────────────────────────────────────

def write_retrain_log(metrics: dict):
    existing = []
    if RETRAIN_LOG.exists():
        try:
            existing = json.loads(RETRAIN_LOG.read_text())
        except Exception:
            pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    existing.append(entry)
    RETRAIN_LOG.write_text(json.dumps(existing, indent=2))
    log.info(f"Retrain log updated: {RETRAIN_LOG}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-only",   action="store_true")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  PolyWeather — Retrain")
    log.info("=" * 55)

    # Load existing data
    df_existing = load_or_init_forecasts()
    wu_existing = load_or_init_wu()

    # Collect
    if not args.train_only:
        df = update_forecasts(df_existing, args.from_scratch)
        wu = update_wu(wu_existing, args.from_scratch)
    else:
        df = df_existing
        wu = wu_existing

    if args.collect_only:
        log.info("Collect-only mode — skipping training")
        return

    # Build features
    log.info("Building features...")
    data = build_features(df, wu)
    log.info(f"Dataset: {len(data)} rows after merge")

    # Train
    metrics = train_model(data)

    # Log
    write_retrain_log(metrics)

    log.info("=" * 55)
    log.info(f"  Done! CV MAE={metrics['cv_mae']:.3f} Acc={metrics['cv_acc']:.1%}")
    log.info(f"  Trained on {metrics['n_train']} rows")
    log.info(f"  Data range: {metrics['date_range']}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()