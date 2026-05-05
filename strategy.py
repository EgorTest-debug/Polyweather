"""
Trading strategy: edge detection → Kelly sizing → position management → risk controls.

This is the brain. Given ensemble forecasts and market data, it decides:
- Is there an edge? (model_prob vs market_price)
- How much to bet? (fractional Kelly)
- When to exit? (stop-loss, trailing stop, forecast-change, take-profit, auto-resolution)
"""

import re
import math
import json
import logging
from datetime import datetime, timezone, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path

import requests

from forecast import EnsembleForecast
from markets import TempBucket, WeatherEvent, fetch_clob_depth

log = logging.getLogger("polyweather")

# ─── Per-city max entry price ─────────────────────────────
# Based on CV bucket accuracy (honest walk-forward validation):
# >60%: can afford up to 50¢
# 55-60%: up to 40¢
# 50-55%: up to 35¢
# <50%: up to 25¢
CITY_MAX_ENTRY = {
    "tel-aviv":     0.50,   # 62.6% CV accuracy
    "helsinki":     0.50,   # 60.4%
    "singapore":    0.50,   # 60.0%
    "wellington":   0.50,   # 59.7%
    "ankara":       0.40,   # 57.5%
    "toronto":      0.40,   # 55.3%
    "seoul":        0.40,   # 54.9%
    "buenos-aires": 0.40,   # 54.6%
    "sao-paulo":    0.35,   # 50.2%
    "tokyo":        0.25,   # 44.9% — lowest accuracy
}

# Slippage: real fill price is typically 2¢ worse than Gamma midpoint
SLIPPAGE_ESTIMATE = 0.02

# ─── Config ────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """All tunable parameters in one place."""
    # Entry filters
    min_edge:          float = 0.10    # 10% minimum edge (after slippage)
    min_agreement:     float = 0.55    # ensemble agreement
    max_entry_price:   float = 0.50    # global cap — per-city CITY_MAX_ENTRY overrides
    min_volume:        float = 200.0   # minimum market volume $
    max_spread:        float = 0.08    # max bid-ask spread 8¢
    min_hours:         float = 12.0     # don't trade if resolves < 12h
    max_hours:         float = 36.0    # don't trade if resolves > 36h
    best_bucket_only:  bool  = True    # only 1 best bucket per city per day
    use_slippage:      bool  = True    # add 2¢ slippage to entry for EV calc

    # Sizing
    kelly_fraction:    float = 0.15    # 15% fractional Kelly
    max_bet:           float = 35.0    # hard cap per trade ($)
    min_bet:           float = 1.0     # minimum trade size ($)
    max_pct_balance:   float = 0.05    # max 5% of balance per trade

    # Exit rules
    stop_loss_pct:     float = 0.20    # 20% stop loss
    trailing_trigger:  float = 0.99    # move stop to breakeven after +20%
    take_profit_72h:   float = 0.75    # take profit at 75¢ if >48h to resolution
    take_profit_48h:   float = 0.85    # take profit at 85¢ if 24-48h

    # Risk limits
    daily_loss_limit:  float = 50.0    # stop trading after $35 loss in a day
    max_positions:     int   = 5       # max simultaneous positions
    max_per_city:      int   = 1       # max positions in one city
    drawdown_stop_pct: float = 0.25    # stop everything at -15% from peak


# ─── Signal ────────────────────────────────────────────────

@dataclass
class Signal:
    """A trading signal — the result of edge detection."""
    city_key:      str
    target_date:   date
    bucket:        TempBucket
    forecast:      EnsembleForecast

    # Computed
    model_prob:    float = 0.0
    market_price:  float = 0.0
    edge:          float = 0.0
    agreement:     float = 0.0
    direction:     str   = "yes"

    # Sizing
    kelly:         float = 0.0
    bet_size:      float = 0.0

    @property
    def passes(self) -> bool:
        return self.edge > 0 and self.bet_size > 0


# ─── Position ──────────────────────────────────────────────

@dataclass
class Position:
    """An open or closed position."""
    id:            str
    city_key:      str
    target_date:   str
    question:      str
    token_id:      str

    entry_price:   float
    shares:        float
    cost:          float

    model_prob:    float
    edge:          float
    agreement:     float

    stop_price:    float
    opened_at:     str
    status:        str = "open"

    exit_price:    Optional[float] = None
    pnl:           Optional[float] = None
    close_reason:  Optional[str]   = None
    closed_at:     Optional[str]   = None


# ─── State persistence ─────────────────────────────────────

STATE_FILE = Path("data/state.json")

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "balance": 420.0,
        "starting_balance": 420.0,
        "peak_balance": 420.0,
        "positions": {},
        "closed_trades": [],
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "daily_pnl": {},
    }

def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ─── Math ──────────────────────────────────────────────────

def calc_kelly(prob: float, price: float) -> float:
    if price <= 0 or price >= 1 or prob <= 0:
        return 0.0
    b = 1.0 / price - 1.0
    f = (prob * b - (1 - prob)) / b
    return max(0.0, f)

def calc_ev(prob: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    return prob * (1.0 / price - 1.0) - (1.0 - prob)


# ─── Edge detection (Layer 4) ──────────────────────────────

def detect_signals(
    event: WeatherEvent,
    forecasts: Dict[date, EnsembleForecast],
    config: StrategyConfig,
    balance: float,
) -> List[Signal]:
    fc = forecasts.get(event.target_date)
    if fc is None or fc.n_members < 5:
        return []

    hours = event.hours_to_resolution
    if hours < config.min_hours or hours > config.max_hours:
        return []

    # Per-city max entry price based on CV accuracy
    city_max_entry = CITY_MAX_ENTRY.get(event.city_key, config.max_entry_price)

    signals = []

    for bucket in event.buckets:
        if bucket.volume < config.min_volume:
            continue

        if bucket.temp_low == -999:
            model_prob = fc.prob_below(bucket.temp_high)
        elif bucket.temp_high == 999:
            model_prob = fc.prob_above(bucket.temp_low)
        else:
            model_prob = fc.prob_between(bucket.temp_low, bucket.temp_high)

        fetch_clob_depth(bucket)

        # Only real CLOB best_ask — no Gamma fallback
        # Entry price logic:
        # 1. If CLOB has real asks (< 0.90) — use best_ask (most accurate)
        # 2. Otherwise use Gamma + slippage (conservative estimate)
        if bucket.best_ask and bucket.best_ask < 0.90:
            market_price = bucket.best_ask
        else:
            market_price = bucket.yes_price + SLIPPAGE_ESTIMATE

        # Per-city price cap
        if market_price > city_max_entry:
            continue

        effective_price = market_price + (SLIPPAGE_ESTIMATE if config.use_slippage else 0)

        if (bucket.spread is not None
            and bucket.spread < 0.90
            and bucket.spread > config.max_spread):
            continue

        # Edge on effective price
        edge = model_prob - effective_price
        if edge < config.min_edge:
            continue

        if fc.agreement < config.min_agreement:
            continue

        kelly_raw = calc_kelly(model_prob, effective_price)
        kelly = kelly_raw * config.kelly_fraction
        bet_raw = kelly * balance
        bet = round(min(bet_raw, config.max_bet, balance * config.max_pct_balance), 2)
        if bet < config.min_bet:
            continue

        signals.append(Signal(
            city_key=event.city_key,
            target_date=event.target_date,
            bucket=bucket,
            forecast=fc,
            model_prob=round(model_prob, 4),
            market_price=round(market_price, 4),
            edge=round(edge, 4),
            agreement=round(fc.agreement, 4),
            direction="yes",
            kelly=round(kelly, 4),
            bet_size=bet,
        ))

    signals.sort(key=lambda s: s.edge * s.model_prob, reverse=True)

    if config.best_bucket_only and signals:
        signals = [signals[0]]

    return signals


# ─── Auto-resolution check ────────────────────────────────

def check_resolutions(positions: Dict[str, dict]) -> List[dict]:
    """
    Check if any open positions have been resolved on Polymarket.
    Queries Gamma API for each market — if closed and a winner is determined,
    returns resolution info.

    Returns list of:
        {"market_id": ..., "won": True/False, "winner_temp": int, "pnl": float}
    """
    resolved = []

    for mid, pos in list(positions.items()):
        if pos.get("status") != "open":
            continue

        try:
            url = f"https://gamma-api.polymarket.com/markets/{mid}"
            r = requests.get(url, timeout=(5, 8))
            mdata = r.json()

            # Check if market is closed/resolved
            if not mdata.get("closed", False):
                continue

            # Check resolvedTo field
            resolved_to = mdata.get("resolvedTo", "")

            # Also check outcome prices — if YES=1.0 this market won
            try:
                prices = json.loads(mdata.get("outcomePrices", "[0,0]"))
                yes_price = float(prices[0])
            except Exception:
                yes_price = 0.0

            won = (resolved_to == "Yes") or (yes_price > 0.95)

            if won:
                # WIN: shares pay out $1 each
                pnl = round(pos["shares"] * 1.0 - pos["cost"], 2)
            else:
                # LOSS: shares worth $0
                pnl = round(-pos["cost"], 2)

            # Try to extract winner temp from other markets in the event
            winner_temp = None
            question = mdata.get("question", "")
            m = re.search(r'(\d+)\s*°', question)
            if m:
                winner_temp = int(m.group(1))

            resolved.append({
                "market_id":    mid,
                "won":          won,
                "winner_temp":  winner_temp,
                "pnl":          pnl,
                "exit_price":   1.0 if won else 0.0,
                "reason":       "resolved_win" if won else "resolved_loss",
            })

            log.info(
                f"[Resolution] {pos.get('city_key','?')} {pos.get('target_date','?')} "
                f"bucket {pos.get('bucket_low','?')}-{pos.get('bucket_high','?')}° → "
                f"{'WIN' if won else 'LOSS'} (PnL: ${pnl:+.2f})"
            )

        except Exception as e:
            log.debug(f"Resolution check failed for {mid}: {e}")
            continue

    return resolved


# ─── Position management (Layer 6) ─────────────────────────

def check_exits(
    positions: Dict[str, dict],
    config: StrategyConfig,
    forecasts: Dict[date, EnsembleForecast] = None,
) -> List[dict]:
    """
    Check all open positions for exit conditions:
    1. Auto-resolution (market closed on Polymarket) — highest priority
    2. Stop loss
    3. Trailing stop (breakeven after +20%)
    4. Take profit (based on hours to resolution)
    5. Forecast changed (ensemble shifted out of bucket)
    """
    exits = []

    # ── First: check auto-resolutions ──────────────────
    resolutions = check_resolutions(positions)
    exits.extend(resolutions)
    resolved_ids = {r["market_id"] for r in resolutions}

    # ── Then: check price-based exits for non-resolved positions ──
    for mid, pos in list(positions.items()):
        if pos.get("status") != "open":
            continue
        if mid in resolved_ids:
            continue

        # Fetch current price
        current_price = None
        try:
            url = f"https://gamma-api.polymarket.com/markets/{mid}"
            r = requests.get(url, timeout=(3, 5))
            mdata = r.json()
            prices = json.loads(mdata.get("outcomePrices", "[0.5,0.5]"))
            current_price = float(prices[0])
        except Exception:
            continue

        entry = pos["entry_price"]
        stop = pos.get("stop_price", entry * (1 - config.stop_loss_pct))
        reason = None

        # Trailing stop: if up 20%+, move stop to breakeven
        if current_price >= entry * (1 + config.trailing_trigger):
            if stop < entry:
                pos["stop_price"] = entry
                stop = entry

        # Stop loss / trailing stop
        if current_price <= stop:
            reason = "stop_loss" if current_price < entry else "trailing_stop"

        # Take profit (based on time)
        if reason is None:
            try:
                end = datetime.fromisoformat(
                    pos.get("end_date", "2099-01-01").replace("Z", "+00:00")
                )
                hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600
            except Exception:
                hours_left = 999

            if hours_left > 48 and current_price >= config.take_profit_72h:
                reason = "take_profit"
            elif 24 < hours_left <= 48 and current_price >= config.take_profit_48h:
                reason = "take_profit"

        # Forecast changed: model shifted out of bucket
        if reason is None and forecasts:
            target_d = date.fromisoformat(pos["target_date"])
            fc = forecasts.get((pos.get("city_key"), target_d)) or forecasts.get(target_d)
            if fc:
                low = pos.get("bucket_low", -999)
                high = pos.get("bucket_high", 999)
                if low != -999 and high != 999:
                    mid_bucket = (low + high) / 2
                    diff = abs(fc.mean - mid_bucket)
                    threshold = (high - low) + 2
                    if diff > threshold:
                        reason = "forecast_changed"

        if reason:
            pnl = round((current_price - entry) * pos["shares"], 2)
            exits.append({
                "market_id":    mid,
                "exit_price":   current_price,
                "pnl":          pnl,
                "reason":       reason,
            })

    return exits


# ─── Risk engine (Layer 7) ─────────────────────────────────

def check_risk(state: dict, config: StrategyConfig) -> dict:
    balance = state["balance"]
    peak = state.get("peak_balance", balance)

    today_str = datetime.now(timezone.utc).date().isoformat()
    daily_pnl = state.get("daily_pnl", {}).get(today_str, 0.0)

    open_count = sum(
        1 for p in state.get("positions", {}).values()
        if p.get("status") == "open"
    )

    drawdown = (peak - balance) / peak if peak > 0 else 0

    can_trade = True
    reasons = []

    if daily_pnl <= -config.daily_loss_limit:
        can_trade = False
        reasons.append(f"daily loss ${abs(daily_pnl):.2f} >= limit ${config.daily_loss_limit}")

    if open_count >= config.max_positions:
        can_trade = False
        reasons.append(f"{open_count} positions >= max {config.max_positions}")

    if drawdown >= config.drawdown_stop_pct:
        can_trade = False
        reasons.append(f"drawdown {drawdown:.1%} >= stop {config.drawdown_stop_pct:.0%}")

    return {
        "can_trade":    can_trade,
        "reasons":      reasons,
        "balance":      balance,
        "daily_pnl":    daily_pnl,
        "open_count":   open_count,
        "drawdown":     drawdown,
        "peak":         peak,
    }


# ─── Quick test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = StrategyConfig()
    state = load_state()
    risk = check_risk(state, config)

    print(f"\nBalance: ${state['balance']:.2f}")
    print(f"Can trade: {risk['can_trade']}")
    if risk['reasons']:
        for r in risk['reasons']:
            print(f"  warning {r}")
    print(f"Config: min_edge={config.min_edge}, kelly={config.kelly_fraction}, "
          f"max_bet=${config.max_bet}")