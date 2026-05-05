"""
Polymarket market scanner.

Finds weather markets via Gamma API, parses temperature buckets,
fetches real bid/ask from CLOB API for accurate pricing.
"""

import re
import json
import logging
import time
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests

from cities import CITIES, MONTHS

log = logging.getLogger("polyweather")

# ─── Data classes ──────────────────────────────────────────

@dataclass
class TempBucket:
    """A single temperature outcome in a weather market."""
    market_id:   str
    token_id:    str              # needed for CLOB orders
    question:    str
    temp_low:    float            # -999 = "or below"
    temp_high:   float            # 999  = "or higher"
    yes_price:   float            # from Gamma (approximate)
    no_price:    float
    volume:      float
    # Real CLOB data (filled by fetch_clob_depth)
    best_bid:    Optional[float] = None
    best_ask:    Optional[float] = None
    spread:      Optional[float] = None

    @property
    def unit(self) -> str:
        if "°F" in self.question or "°f" in self.question:
            return "F"
        return "C"

    def contains(self, temp: float) -> bool:
        """Does this bucket contain the given temperature?"""
        return self.temp_low <= temp <= self.temp_high


@dataclass
class WeatherEvent:
    """A Polymarket weather event (one city + one day)."""
    city_key:     str
    target_date:  date
    event_id:     str
    end_date:     str
    buckets:      List[TempBucket] = field(default_factory=list)

    @property
    def hours_to_resolution(self) -> float:
        try:
            end = datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
            return max(0.0, (end - datetime.now(timezone.utc)).total_seconds() / 3600)
        except Exception:
            return 999.0


# ─── Temperature range parser ──────────────────────────────

def parse_temp_range(question: str) -> Optional[Tuple[float, float]]:
    """Extract (low, high) from market question text."""
    if not question:
        return None

    q = question.strip()
    num = r'(-?\d+(?:\.\d+)?)'

    # "X°C or below" / "X°F or below"
    if "or below" in q.lower():
        m = re.search(num + r'\s*°[CF]\s+or below', q, re.IGNORECASE)
        if m:
            return (-999.0, float(m.group(1)))

    # "X°C or higher" / "X°F or higher"
    if "or higher" in q.lower():
        m = re.search(num + r'\s*°[CF]\s+or higher', q, re.IGNORECASE)
        if m:
            return (float(m.group(1)), 999.0)

    # "between X-Y°C" / "between X-Y°F"
    m = re.search(r'between\s+' + num + r'\s*[-–]\s*' + num + r'\s*°[CF]', q, re.IGNORECASE)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    # "be X°C on" (exact value)
    m = re.search(r'be\s+' + num + r'\s*°[CF]\s+on', q, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return (v, v)

    return None


# ─── Gamma API ─────────────────────────────────────────────

def _build_slug(city_key: str, d: date) -> str:
    """Build Polymarket event slug for a weather market."""
    month = MONTHS[d.month - 1]
    return f"highest-temperature-in-{city_key}-on-{month}-{d.day}-{d.year}"


def fetch_event(city_key: str, d: date) -> Optional[WeatherEvent]:
    """
    Fetch a weather event from Polymarket Gamma API.
    Returns WeatherEvent with all temperature buckets, or None.
    """
    slug = _build_slug(city_key, d)
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"

    try:
        resp = requests.get(url, timeout=(5, 10))
        data = resp.json()

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        event = data[0]
        event_id = str(event.get("id", ""))
        end_date = event.get("endDate", "")

        buckets = []
        for mkt in event.get("markets", []):
            question = mkt.get("question", "")
            rng = parse_temp_range(question)
            if rng is None:
                continue

            try:
                prices = json.loads(mkt.get("outcomePrices", "[0.5,0.5]"))
                yes_price = float(prices[0])
                no_price = float(prices[1]) if len(prices) > 1 else 1.0 - yes_price
            except Exception:
                continue

            # Token IDs for CLOB
            try:
                token_ids = json.loads(mkt.get("clobTokenIds", "[\"\",\"\"]"))
                yes_token = token_ids[0] if token_ids else ""
            except Exception:
                yes_token = ""

            buckets.append(TempBucket(
                market_id=str(mkt.get("id", "")),
                token_id=yes_token,
                question=question,
                temp_low=rng[0],
                temp_high=rng[1],
                yes_price=round(yes_price, 4),
                no_price=round(no_price, 4),
                volume=float(mkt.get("volume", 0)),
            ))

        if not buckets:
            return None

        buckets.sort(key=lambda b: b.temp_low)

        return WeatherEvent(
            city_key=city_key,
            target_date=d,
            event_id=event_id,
            end_date=end_date,
            buckets=buckets,
        )

    except Exception as e:
        log.warning(f"[{CITIES[city_key]['name']}] Gamma API error: {e}")
        return None


# ─── CLOB depth check ─────────────────────────────────────

def fetch_clob_depth(bucket: TempBucket) -> TempBucket:
    if not bucket.token_id:
        return bucket
    try:
        url = f"https://clob.polymarket.com/price?token_id={bucket.token_id}&side=BUY"
        resp = requests.get(url, timeout=(3, 5))
        price = resp.json().get("price")
        if price is not None:
            bucket.best_ask = float(price)

        url2 = f"https://clob.polymarket.com/price?token_id={bucket.token_id}&side=SELL"
        resp2 = requests.get(url2, timeout=(3, 5))
        sell = resp2.json().get("price")
        if sell is not None:
            bucket.best_bid = float(sell)
            if bucket.best_ask is not None:
                bucket.spread = round(bucket.best_ask - bucket.best_bid, 4)
    except Exception as e:
        log.debug(f"CLOB price error for {bucket.market_id}: {e}")
    return bucket


# ─── Convenience: scan all markets for a city ──────────────

def scan_city(city_key: str, days_ahead: int = 4) -> List[WeatherEvent]:
    """Find all active weather events for a city in the next N days."""
    today = datetime.now(timezone.utc).date()
    events = []

    for i in range(days_ahead):
        d = today + timedelta(days=i)
        event = fetch_event(city_key, d)
        if event:
            events.append(event)
        time.sleep(0.2)  # rate limit

    return events


# ─── Quick test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    today = datetime.now(timezone.utc).date()

    for city in ["seoul", "singapore", "toronto"]:
        print(f"\n{'='*60}")
        print(f"  {CITIES[city]['name']} — scanning Polymarket")
        print(f"{'='*60}")

        events = scan_city(city, days_ahead=3)
        if not events:
            print("  No markets found")
            continue

        for ev in events:
            horizon = (ev.target_date - today).days
            print(f"\n  D+{horizon} {ev.target_date} | {len(ev.buckets)} buckets | "
                  f"resolves in {ev.hours_to_resolution:.0f}h")

            for b in ev.buckets[:5]:  # show first 5
                low_s = f"{b.temp_low}" if b.temp_low > -999 else "≤"
                high_s = f"{b.temp_high}" if b.temp_high < 999 else "+"
                print(f"    {low_s}–{high_s}°{b.unit}: YES ${b.yes_price:.3f} "
                      f"| vol ${b.volume:,.0f} | {b.question[:50]}...")
