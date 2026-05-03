"""
LightGBM predictor — calibrates ECMWF ensemble forecasts.

Fetches D+1 forecasts from 11 models + context variables,
runs LightGBM to get calibrated T_max, then shifts ECMWF
ensemble members by the difference.

Requires:
    data/models/global_model_tuned.pkl
    data/models/feature_cols.pkl
"""

import math
import logging
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

log = logging.getLogger("polyweather")

MODEL_PATH    = Path("data/models/global_model_tuned.pkl")
FEATURES_PATH = Path("data/models/feature_cols.pkl")

PREV_RUNS_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"

MODELS = [
    "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_seamless",
    "gem_seamless", "meteofrance_seamless", "ukmo_seamless",
    "knmi_seamless", "dmi_seamless", "cma_grapes_global", "ncep_gfs_graphcast025",
]

CONTEXT_VARS = [
    "cloudcover_mean", "precipitation_sum", "shortwave_radiation_sum",
    "relative_humidity_2m_mean", "windspeed_10m_max",
    "pressure_msl_mean", "dew_point_2m_mean",
]

CITY_ID_MAP = {
    "ankara": 0, "buenos-aires": 1, "helsinki": 2, "sao-paulo": 3,
    "seoul": 4, "singapore": 5, "tel-aviv": 6, "tokyo": 7,
    "toronto": 8, "wellington": 9,
}

WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

STATION_MAP = {
    "seoul":        ("RKSI", "KR"),
    "tokyo":        ("RJTT", "JP"),
    "singapore":    ("WSSS", "SG"),
    "tel-aviv":     ("LLBG", "IL"),
    "helsinki":     ("EFHK", "FI"),
    "ankara":       ("LTAC", "TR"),
    "toronto":      ("CYYZ", "CA"),
    "sao-paulo":    ("SBGR", "BR"),
    "buenos-aires": ("SAEZ", "AR"),
    "wellington":   ("NZWN", "NZ"),
}


# ─── Load model ────────────────────────────────────────────

_model = None
_feature_cols = None

def _load_model():
    global _model, _feature_cols
    try:
        import lightgbm as lgb
        import joblib

        txt_path = Path("data/models/global_model_tuned.txt")
        if txt_path.exists():
            _model = lgb.Booster(model_file=str(txt_path))
            _feature_cols = joblib.load(FEATURES_PATH)
            log.info(f"LightGBM loaded ({len(_feature_cols)} features)")
        else:
            log.warning(f"Model .txt not found at {txt_path}")
    except Exception as e:
        log.warning(f"Model load failed: {e}")
        import traceback
        traceback.print_exc()

_load_model()


# ─── Data fetching ─────────────────────────────────────────

def _fetch_forecasts(lat: float, lon: float, tz: str, target_date: date) -> Optional[dict]:
    """Fetch 11-model forecasts + context variables from Previous Runs API."""
    d_str = target_date.isoformat()
    result = {}

    try:
        r1 = requests.get(PREV_RUNS_BASE, params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": d_str, "end_date": d_str,
            "models": ",".join(MODELS),
            "previous_day": 1,
            "timezone": tz,
        }, timeout=(5, 15))
        d1 = r1.json().get("daily", {})
        for m in MODELS:
            key = f"temperature_2m_max_{m}"
            vals = d1.get(key, [None])
            result[f"{m}_tmax"] = vals[0] if vals else None
    except Exception as e:
        log.debug(f"Model forecasts fetch error: {e}")
        return None

    try:
        r2 = requests.get(PREV_RUNS_BASE, params={
            "latitude": lat, "longitude": lon,
            "daily": ",".join(CONTEXT_VARS),
            "start_date": d_str, "end_date": d_str,
            "models": "ecmwf_ifs025",
            "previous_day": 1,
            "timezone": tz,
        }, timeout=(5, 15))
        d2 = r2.json().get("daily", {})
        for v in CONTEXT_VARS:
            vals = d2.get(v) or d2.get(f"{v}_ecmwf_ifs025") or [None]
            result[v] = vals[0] if vals else None
    except Exception as e:
        log.debug(f"Context vars fetch error: {e}")

    return result


def _fetch_wu_max(city_key: str, d: date) -> Optional[float]:
    """Fetch WU max temperature for a given date."""
    station, country = STATION_MAP.get(city_key, (None, None))
    if not station:
        return None
    try:
        url = (
            f"https://api.weather.com/v1/location/{station}:9:{country}"
            f"/observations/historical.json"
            f"?apiKey={WU_API_KEY}&units=m"
            f"&startDate={d.strftime('%Y%m%d')}&endDate={d.strftime('%Y%m%d')}"
        )
        r = requests.get(url, timeout=(5, 10))
        obs = r.json().get("observations", [])
        temps = [o.get("temp") for o in obs if o.get("temp") is not None]
        return max(temps) if temps else None
    except Exception:
        return None


# ─── Main prediction ──────────────────────────────────────

def predict_temperature(city_key: str, target_date: date,
                         ecmwf_mean: float) -> Optional[float]:
    """
    Predict calibrated T_max using LightGBM.

    Args:
        city_key: city identifier (e.g. 'seoul')
        target_date: date to predict for
        ecmwf_mean: Markov-corrected ECMWF ensemble mean

    Returns:
        Calibrated T_max or None if unavailable
    """
    if _model is None or _feature_cols is None:
        return None

    from cities import CITIES
    cfg = CITIES[city_key]

    forecasts = _fetch_forecasts(cfg["lat"], cfg["lon"], cfg["timezone"], target_date)
    if forecasts is None:
        return None

    # WU D-2 and D-3
    wu_d2 = _fetch_wu_max(city_key, target_date - timedelta(days=2))
    wu_d3 = _fetch_wu_max(city_key, target_date - timedelta(days=3))

    # Seasonality
    doy = target_date.timetuple().tm_yday
    doy_sin = math.sin(2 * math.pi * doy / 365)
    doy_cos = math.cos(2 * math.pi * doy / 365)

    # Model ensemble stats
    model_vals = [forecasts.get(f"{m}_tmax") for m in MODELS]
    model_vals_clean = [v for v in model_vals if v is not None]
    model_mean = statistics.mean(model_vals_clean) if model_vals_clean else ecmwf_mean
    model_std  = statistics.stdev(model_vals_clean) if len(model_vals_clean) > 1 else 1.0
    model_min  = min(model_vals_clean) if model_vals_clean else ecmwf_mean - 2
    model_max  = max(model_vals_clean) if model_vals_clean else ecmwf_mean + 2

    # Build feature dict
    feat = {}
    for f in _feature_cols:
        if f.endswith("_tmax") and not f.startswith("model_"):
            feat[f] = forecasts.get(f)
        elif f in CONTEXT_VARS:
            feat[f] = forecasts.get(f)
        elif f == "wu_max_d_minus_2":
            feat[f] = wu_d2
        elif f == "wu_max_d_minus_3":
            feat[f] = wu_d3
        elif f == "doy_sin":
            feat[f] = doy_sin
        elif f == "doy_cos":
            feat[f] = doy_cos
        elif f == "city_id":
            feat[f] = CITY_ID_MAP.get(city_key, 0)
        elif f == "model_mean":
            feat[f] = model_mean
        elif f == "model_std":
            feat[f] = model_std
        elif f == "model_min":
            feat[f] = model_min
        elif f == "model_max":
            feat[f] = model_max
        else:
            feat[f] = None

    import numpy as np
    X = np.array([[feat.get(f) for f in _feature_cols]], dtype=float)

    try:
        import traceback
        pred = float(_model.predict(X, num_iteration=_model.best_iteration)[0])
        shift = pred - ecmwf_mean
        log.info(
            f"[{cfg['name']}] LightGBM: mean={ecmwf_mean:.1f}° → "
            f"calibrated={pred:.1f}° (shift={shift:+.1f}°)"
        )
        return round(pred, 1)
    except Exception as e:
        log.warning(f"[{city_key}] LightGBM predict failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def shift_ensemble_members(members: List[float], ecmwf_mean: float,
                             lgbm_pred: float) -> List[float]:
    """
    Shift ECMWF ensemble by LightGBM correction.
    Preserves std, shifts mean to lgbm_pred.
    """
    shift = lgbm_pred - ecmwf_mean
    return [round(m + shift, 1) for m in members]


# ─── Quick test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)

    for city in ["tel-aviv", "singapore", "wellington"]:
        print(f"\n  {city}")
        pred = predict_temperature(city, tomorrow, ecmwf_mean=20.0)
        print(f"  → {pred}°C")