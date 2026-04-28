"""
ECMWF Ensemble probability engine with adaptive Markov bias correction.

Uses ECMWF IFS 50-member ensemble via ensemble-api.open-meteo.com.
Probability = direct count of members in bucket / total members.

Bias correction: Exponential Moving Average (EMA) of recent ECMWF errors
vs Wunderground (Polymarket resolution source). Updates dynamically —
if yesterday ECMWF undershot by 3°, today's correction increases.

Static bias_correction in cities.py is used as fallback when no
recent error data is available.
"""

import json
import time
import logging
import statistics
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import requests

from cities import CITIES, MONTHS

log = logging.getLogger("polyweather")

ENSEMBLE_BASE = "https://ensemble-api.open-meteo.com/v1/ensemble"

REGION_MODELS = {
    "default":  ("ecmwf_ifs025", 50),
    "toronto":  ("ecmwf_ifs025", 50),
}

# EMA smoothing factor: 0.4 = responsive (adapts in 2-3 days), 0.2 = smoother
EMA_ALPHA = 0.4

# File to persist EMA state between bot restarts
EMA_STATE_FILE = Path("data/ema_bias.json")


# ─── EMA bias state ───────────────────────────────────────

def _load_ema_state() -> dict:
    """Load persisted EMA bias corrections per city."""
    if EMA_STATE_FILE.exists():
        try:
            return json.loads(EMA_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_ema_state(state: dict):
    EMA_STATE_FILE.parent.mkdir(exist_ok=True)
    EMA_STATE_FILE.write_text(json.dumps(state, indent=2))


def _fetch_yesterday_winner(city_key: str, yesterday: date) -> Optional[int]:
    """Get Polymarket winner temp for yesterday (= Wunderground value)."""
    slug = f"highest-temperature-in-{city_key}-on-{MONTHS[yesterday.month-1]}-{yesterday.day}-{yesterday.year}"
    try:
        r = requests.get(
            f"https://gamma-api.polymarket.com/events?slug={slug}",
            timeout=(5, 10)
        )
        data = r.json()
        if not data:
            return None
        for mkt in data[0].get("markets", []):
            try:
                prices = json.loads(mkt.get("outcomePrices", "[0,0]"))
                if float(prices[0]) > 0.95:
                    import re
                    m = re.search(r'(\d+)\s*°', mkt.get("question", ""))
                    if m:
                        return int(m.group(1))
            except Exception:
                continue
    except Exception as e:
        log.debug(f"[{city_key}] Failed to fetch yesterday's winner: {e}")
    return None


def _fetch_yesterday_ecmwf(city_key: str, yesterday: date) -> Optional[float]:
    """Get what ECMWF predicted (raw, no bias) for yesterday."""
    cfg = CITIES[city_key]
    try:
        url = (
            f"https://api.open-meteo.com/v1/ecmwf"
            f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
            f"&daily=temperature_2m_max&temperature_unit=celsius"
            f"&timezone={cfg['timezone']}"
            f"&start_date={yesterday.isoformat()}"
            f"&end_date={yesterday.isoformat()}"
        )
        r = requests.get(url, timeout=(5, 10))
        data = r.json()
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        if temps and temps[0] is not None:
            return round(temps[0], 1)
    except Exception as e:
        log.debug(f"[{city_key}] Failed to fetch yesterday's ECMWF: {e}")
    return None


def get_adaptive_bias(city_key: str) -> float:
    """
    Get adaptive bias correction for a city using EMA.

    1. Load persisted EMA state
    2. Fetch yesterday's ECMWF forecast and Wunderground winner
    3. Compute yesterday's error: winner - ecmwf_raw
    4. Update EMA: new_bias = alpha * yesterday_error + (1 - alpha) * old_bias
    5. Persist and return

    Falls back to static bias_correction from cities.py if no data available.
    """
    cfg = CITIES[city_key]
    static_bias = cfg.get("bias_correction", 0.0)
    state = _load_ema_state()

    city_state = state.get(city_key, {
        "ema_bias": static_bias,
        "last_updated": None,
        "last_error": None,
        "error_history": [],
    })

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    # Already updated today?
    if city_state.get("last_updated") == today.isoformat():
        return city_state["ema_bias"]

    # Fetch yesterday's data
    winner = _fetch_yesterday_winner(city_key, yesterday)
    ecmwf_raw = _fetch_yesterday_ecmwf(city_key, yesterday)

    if winner is not None and ecmwf_raw is not None:
        yesterday_error = winner - ecmwf_raw  # positive = ECMWF undershot
        old_bias = city_state.get("ema_bias", static_bias)
        new_bias = round(EMA_ALPHA * yesterday_error + (1 - EMA_ALPHA) * old_bias, 2)

        # Keep history (last 30 errors)
        history = city_state.get("error_history", [])
        history.append({"date": yesterday.isoformat(), "error": yesterday_error, "bias": new_bias})
        history = history[-30:]

        city_state.update({
            "ema_bias": new_bias,
            "last_updated": today.isoformat(),
            "last_error": yesterday_error,
            "error_history": history,
        })

        log.info(
            f"[{cfg['name']}] Markov bias: yesterday error={yesterday_error:+.1f}° "
            f"(ECMWF={ecmwf_raw:.1f}, WU={winner}) → "
            f"EMA bias={new_bias:+.1f}° (was {old_bias:+.1f}°)"
        )
    else:
        # No data — keep previous EMA or fall back to static
        if "ema_bias" not in city_state:
            city_state["ema_bias"] = static_bias
        log.debug(f"[{cfg['name']}] No yesterday data, using bias={city_state['ema_bias']:+.1f}°")

    state[city_key] = city_state
    _save_ema_state(state)

    return city_state["ema_bias"]


# ─── Data classes ──────────────────────────────────────────

@dataclass
class EnsembleForecast:
    """Real ensemble: 50 independent T_max values from ECMWF IFS."""
    city_key:      str
    target_date:   date
    model:         str
    member_highs:  List[float]
    fetched_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def n_members(self) -> int:
        return len(self.member_highs)

    @property
    def mean(self) -> float:
        return statistics.mean(self.member_highs) if self.member_highs else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.member_highs) if self.member_highs else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.member_highs) if len(self.member_highs) > 1 else 0.0

    def prob_above(self, threshold: float) -> float:
        if not self.member_highs:
            return 0.5
        above = sum(1 for t in self.member_highs if t > threshold)
        raw = above / len(self.member_highs)
        return max(0.05, min(0.95, raw))

    def prob_below(self, threshold: float) -> float:
        return 1.0 - self.prob_above(threshold)

    def prob_between(self, low: float, high: float) -> float:
        if not self.member_highs:
            return 0.5
        if high - low < 1.0:
            actual_low = low - 0.5
            actual_high = high + 0.5
        else:
            actual_low = low
            actual_high = high
        count = sum(1 for t in self.member_highs if actual_low <= t < actual_high)
        raw = count / len(self.member_highs)
        return max(0.03, min(0.95, raw))

    @property
    def agreement(self) -> float:
        if len(self.member_highs) < 2:
            return 0.5
        spread = max(self.member_highs) - min(self.member_highs)
        return max(0.3, min(1.0, 1.0 - spread / 10.0))


# ─── Fetch ensemble from Open-Meteo ───────────────────────

def fetch_ensemble(city_key: str, forecast_days: int = 4,
                   retries: int = 3) -> Dict[date, EnsembleForecast]:
    cfg = CITIES[city_key]
    model_name, _ = REGION_MODELS.get(city_key, REGION_MODELS["default"])

    url = (
        f"{ENSEMBLE_BASE}"
        f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
        f"&daily=temperature_2m_max"
        f"&temperature_unit=celsius"
        f"&timezone={cfg['timezone']}"
        f"&forecast_days={forecast_days}"
        f"&models={model_name}"
    )

    result: Dict[date, EnsembleForecast] = {}

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=(5, 15))
            data = resp.json()

            if data.get("error"):
                log.warning(f"[{cfg['name']}] API error: {data.get('reason', '?')}")
                break

            daily = data.get("daily", {})
            dates_raw = daily.get("time", [])

            member_keys = sorted(
                k for k in daily
                if k.startswith("temperature_2m_max_member")
            )

            if not member_keys:
                log.warning(f"[{cfg['name']}] No ensemble members in response")
                break

            # Adaptive Markov bias correction (EMA of recent errors)
            bias = get_adaptive_bias(city_key)

            for i, d_str in enumerate(dates_raw):
                d = date.fromisoformat(d_str)
                highs = []
                for mk in member_keys:
                    vals = daily[mk]
                    if i < len(vals) and vals[i] is not None:
                        highs.append(round(vals[i] + bias, 1))

                if highs:
                    result[d] = EnsembleForecast(
                        city_key=city_key,
                        target_date=d,
                        model=f"{model_name}+markov{bias:+.1f}",
                        member_highs=highs,
                    )

            break

        except requests.exceptions.RequestException as e:
            log.warning(f"[{cfg['name']}] request failed (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    return result


# ─── METAR observations (D+0 override) ────────────────────

def fetch_metar_temp(city_key: str) -> Optional[float]:
    station = CITIES[city_key]["station"]
    try:
        url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json"
        resp = requests.get(url, timeout=(5, 8))
        data = resp.json()
        if data and isinstance(data, list):
            temp_c = data[0].get("temp")
            if temp_c is not None:
                return round(float(temp_c), 1)
    except Exception as e:
        log.debug(f"[{CITIES[city_key]['name']}] METAR error: {e}")
    return None


# ─── Main entry point ─────────────────────────────────────

def get_forecasts(city_key: str, days_ahead: int = 4) -> Dict[date, EnsembleForecast]:
    forecasts = fetch_ensemble(city_key, forecast_days=max(days_ahead, 4))

    today = datetime.now(timezone.utc).date()
    if today in forecasts:
        observed = fetch_metar_temp(city_key)
        if observed is not None:
            fc = forecasts[today]
            updated = [max(t, observed) for t in fc.member_highs]
            forecasts[today] = EnsembleForecast(
                city_key=city_key,
                target_date=today,
                model=fc.model + "+metar",
                member_highs=updated,
                fetched_at=fc.fetched_at,
            )

    target_dates = {today + timedelta(days=i) for i in range(days_ahead)}
    return {d: fc for d, fc in forecasts.items() if d in target_dates}


# ─── Quick test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    today = datetime.now(timezone.utc).date()

    for city in ["seoul", "singapore", "toronto"]:
        print(f"\n{'='*70}")
        print(f"  {CITIES[city]['name']}")
        print(f"{'='*70}")

        # Show adaptive bias
        bias = get_adaptive_bias(city)
        static = CITIES[city].get("bias_correction", 0.0)
        print(f"  Static bias: {static:+.1f}° | Adaptive (Markov): {bias:+.1f}°")

        forecasts = get_forecasts(city, days_ahead=3)
        for d, fc in sorted(forecasts.items()):
            horizon = (d - today).days
            print(f"\n  D+{horizon} {d}: {fc.n_members} members ({fc.model})")
            print(f"    Mean: {fc.mean:.1f}C | Std: {fc.std:.1f}C | Agreement: {fc.agreement:.0%}")
            print(f"    Range: {min(fc.member_highs):.1f} - {max(fc.member_highs):.1f}C")

            mean_int = int(round(fc.mean))
            for t in range(mean_int - 3, mean_int + 4):
                prob = fc.prob_between(t, t)
                bar = "#" * int(prob * 50)
                print(f"    P({t:3d}C) = {prob:5.0%}  {bar}")

    # Show EMA state
    print(f"\n{'='*70}")
    print(f"  EMA BIAS STATE")
    print(f"{'='*70}")
    state = _load_ema_state()
    for city, cs in state.items():
        last_err = cs.get("last_error", "?")
        print(f"  {city:15s} | EMA={cs['ema_bias']:+.1f}° | last error={last_err}")