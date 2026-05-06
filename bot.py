#!/usr/bin/env python3
"""
PolyWeather — weather trading bot for Polymarket.

Usage:
    python bot.py                    # paper mode, continuous (30 min)
    python bot.py --live             # real trades (needs .env)
    python bot.py --status           # show balance + positions
    python bot.py --once             # single scan, no loop
    python bot.py --test-forecast    # test forecast engine only
    python bot.py --interval 60      # scan every 60 min
"""

import sys
import time
import json
import logging
import argparse
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

from cities import CITIES
from forecast import get_forecasts, EnsembleForecast
from markets import scan_city, WeatherEvent, fetch_clob_depth
from strategy import (
    StrategyConfig, Signal, detect_signals,
    check_exits, check_risk, load_state, save_state,
)
from executor import ClobExecutor, PaperExecutor

# ─── Logging (console + file) ─────────────────────────────

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir / f"polyweather_{datetime.now().strftime('%Y-%m-%d')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("polyweather")

# ─── Display helpers ───────────────────────────────────────

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"
D = "\033[90m"; B = "\033[1m"; X = "\033[0m"

def ok(m):   print(f"  {G}✅ {m}{X}"); log.info(m)
def warn(m): print(f"  {Y}⚠  {m}{X}"); log.warning(m)
def skip(m): print(f"  {D}⏸  {m}{X}")
def err(m):  print(f"  {R}❌ {m}{X}"); log.error(m)

# ─── Core scan cycle ──────────────────────────────────────

def run_scan(executor, config: StrategyConfig, state: dict) -> dict:
    """One full scan cycle across all cities."""
    today = datetime.now(timezone.utc).date()
    balance = state["balance"]
    positions = state.get("positions", {})
    new_trades = 0
    exits_found = 0

    # ── Check exits (auto-resolution + price-based) ────
    print(f"\n{B}📤 Checking exits...{X}")
    all_forecasts = {}

    open_cities = set(p["city_key"] for p in positions.values() if p.get("status") == "open")
    for city in open_cities:
        fcs = get_forecasts(city, days_ahead=2)
        for d, fc in fcs.items():
            all_forecasts[(city, d)] = fc

    exits = check_exits(positions, config, all_forecasts)
    for ex in exits:
        mid = ex["market_id"]
        pos = positions[mid]
        pnl = ex["pnl"]
        reason = ex["reason"]

        balance += pos["cost"] + pnl
        pos["status"] = "closed"
        pos["exit_price"] = ex["exit_price"]
        pos["pnl"] = pnl
        pos["close_reason"] = reason
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()

        state["closed_trades"].append(pos)
        if pnl > 0:
            state["wins"] += 1
        else:
            state["losses"] += 1

        today_str = today.isoformat()
        daily = state.get("daily_pnl", {})
        daily[today_str] = daily.get(today_str, 0) + pnl
        state["daily_pnl"] = daily

        exits_found += 1
        color = G if pnl >= 0 else R
        sym = "+" if pnl >= 0 else ""
        city_name = CITIES.get(pos.get("city_key", ""), {}).get("name", "?")
        bucket = f"{pos.get('bucket_low', '?')}-{pos.get('bucket_high', '?')}°"
        ok(f"EXIT [{reason}] {city_name} {pos.get('target_date','')} {bucket} | {color}{sym}${pnl:.2f}{X}")

    # Remove closed positions
    positions = {k: v for k, v in positions.items() if v.get("status") == "open"}

    if exits_found == 0:
        skip("No exit signals")

    # ── Scan for entries ───────────────────────────────
    print(f"\n{B}🔍 Scanning markets...{X}")

    state["balance"] = round(balance, 2)
    state["positions"] = positions

    # Build cooldown set — market_ids already in closed_trades (any reason)
    # Prevents re-entering same bucket after stop_loss
    closed_market_ids = {t.get("market_id") for t in state.get("closed_trades", []) if t.get("market_id")}

    # ── Pass 1: collect ALL signals across all cities ──
    all_signals = []
    for city_key in CITIES:
        cfg = CITIES[city_key]
        print(f"\n  {B}📍 {cfg['name']}{X} ", end="", flush=True)

        forecasts = get_forecasts(city_key, days_ahead=2)
        if not forecasts:
            print(f"{R}no forecast{X}")
            continue

        events = scan_city(city_key, days_ahead=2)
        if not events:
            print(f"{D}no markets{X}")
            continue

        print(f"{G}{len(events)} events{X}")

        for event in events:
            signals = detect_signals(event, forecasts, config, balance)
            for sig in signals:
                # Skip if already open or in cooldown
                if sig.bucket.market_id in positions:
                    continue
                if sig.bucket.market_id in closed_market_ids:
                    continue
                sig._end_date = event.end_date
                all_signals.append(sig)

        time.sleep(0.3)

    # ── Pass 2: rank by score and enter best signals ───
    all_signals.sort(
        key=lambda s: s.edge * s.model_prob * s.agreement,
        reverse=True
    )

    if all_signals:
        print(f"\n{B}📊 Ranked signals ({len(all_signals)} total):{X}")
        for s in all_signals:
            b = s.bucket
            low_s = f"{b.temp_low:.0f}" if b.temp_low > -999 else "≤"
            high_s = f"{b.temp_high:.0f}" if b.temp_high < 999 else "+"
            score = s.edge * s.model_prob * s.agreement
            print(f"  {CITIES[s.city_key]['name']:12s} {low_s}–{high_s}° | "
                  f"score={score:.3f} | edge={s.edge:+.0%} | "
                  f"prob={s.model_prob:.0%} | agree={s.agreement:.0%} | "
                  f"bet=${s.bet_size:.2f}")
        sys.stdout.flush()

    cities_entered = set()
    for sig in all_signals:
        risk = check_risk(state, config)
        if not risk["can_trade"]:
            warn(f"RISK BLOCK: {', '.join(risk['reasons'])}")
            break

        # One position per city per scan
        if sig.city_key in cities_entered:
            continue

        b = sig.bucket
        today = datetime.now(timezone.utc).date()
        horizon = (sig.target_date - today).days
        low_s = f"{b.temp_low:.0f}" if b.temp_low > -999 else "≤"
        high_s = f"{b.temp_high:.0f}" if b.temp_high < 999 else "+"

        print(f"\n    {G}{B}SIGNAL{X} D+{horizon} | "
              f"{low_s}–{high_s}°{b.unit} | "
              f"edge {sig.edge:+.0%} | "
              f"model {sig.model_prob:.0%} vs mkt ${sig.market_price:.3f} | "
              f"bet ${sig.bet_size:.2f}")
        print(f"    {D}ensemble: {sig.forecast.n_members} members, "
              f"mean {sig.forecast.mean:.1f}°, std {sig.forecast.std:.1f}°, "
              f"agreement {sig.agreement:.0%}{X}")

        order = executor.buy_yes(
            token_id=b.token_id,
            price=sig.market_price,
            size=sig.bet_size,
        )

        balance -= sig.bet_size
        positions[b.market_id] = {
            "city_key":      sig.city_key,
            "target_date":   sig.target_date.isoformat(),
            "question":      b.question,
            "token_id":      b.token_id,
            "entry_price":   sig.market_price,
            "shares":        round(sig.bet_size / sig.market_price, 2),
            "cost":          sig.bet_size,
            "model_prob":    sig.model_prob,
            "edge":          sig.edge,
            "agreement":     sig.agreement,
            "bucket_low":    b.temp_low,
            "bucket_high":   b.temp_high,
            "stop_price":    round(sig.market_price * (1 - config.stop_loss_pct), 4),
            "end_date":      getattr(sig, '_end_date', ''),
            "opened_at":     datetime.now(timezone.utc).isoformat(),
            "status":        "open",
            "market_id":     b.market_id,
        }
        state["total_trades"] += 1
        new_trades += 1
        cities_entered.add(sig.city_key)

        state["balance"] = round(balance, 2)
        state["positions"] = positions

        ok(f"Position opened | ${sig.bet_size:.2f}")

    # ── Save state ─────────────────────────────────────
    state["balance"] = round(balance, 2)
    state["positions"] = positions
    state["peak_balance"] = max(state.get("peak_balance", balance), balance)
    save_state(state)

    # ── Summary ────────────────────────────────────────
    total_return = (balance - state["starting_balance"]) / state["starting_balance"] * 100
    ret_color = G if total_return >= 0 else R
    ret_sign = "+" if total_return >= 0 else ""
    win_rate = state["wins"] / max(state["wins"] + state["losses"], 1) * 100

    print(f"\n{'='*55}")
    print(f"  {B}Balance:{X}  ${balance:.2f}  "
          f"({ret_color}{ret_sign}{total_return:.1f}%{X})")
    print(f"  {B}Trades:{X}   {state['total_trades']} total | "
          f"W:{state['wins']} L:{state['losses']} ({win_rate:.0f}%) | "
          f"this scan: +{new_trades} new, {exits_found} exits")
    print(f"  {B}Open:{X}     {sum(1 for p in positions.values() if p.get('status')=='open')}")
    print(f"{'='*55}")

    return state


# ─── Status display ────────────────────────────────────────

def show_status():
    state = load_state()
    bal = state["balance"]
    start = state["starting_balance"]
    ret = (bal - start) / start * 100
    win_rate = state["wins"] / max(state["wins"] + state["losses"], 1) * 100

    print(f"\n{B}{'='*55}{X}")
    print(f"  {B}POLYWEATHER — STATUS{X}")
    print(f"{B}{'='*55}{X}")
    print(f"  Balance:  ${bal:.2f} (start ${start:.2f}, {'+' if ret>=0 else ''}{ret:.1f}%)")
    print(f"  Trades:   {state['total_trades']} | W: {state['wins']} | L: {state['losses']} ({win_rate:.0f}%)")

    positions = state.get("positions", {})
    open_pos = {k: v for k, v in positions.items() if v.get("status") == "open"}
    print(f"  Open:     {len(open_pos)}")

    if open_pos:
        print(f"\n  Open positions:")
        for mid, pos in open_pos.items():
            city = CITIES.get(pos["city_key"], {}).get("name", pos["city_key"])
            low = pos.get("bucket_low", "?")
            high = pos.get("bucket_high", "?")
            print(f"    {city:<14} {pos['target_date']} | "
                  f"{low}–{high}° | entry ${pos['entry_price']:.3f} | "
                  f"${pos['cost']:.2f} | edge {pos['edge']:+.0%}")

    closed = state.get("closed_trades", [])
    if closed:
        recent = closed[-5:]
        print(f"\n  Recent closed ({len(closed)} total):")
        for pos in recent:
            city = CITIES.get(pos.get("city_key", ""), {}).get("name", "?")
            pnl = pos.get("pnl", 0)
            reason = pos.get("close_reason", "?")
            color = G if pnl >= 0 else R
            sym = "+" if pnl >= 0 else ""
            print(f"    {city:<14} {pos.get('target_date','')} | "
                  f"{reason:<16} | {color}{sym}${pnl:.2f}{X}")

    print(f"{B}{'='*55}{X}\n")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PolyWeather — weather trading bot")
    parser.add_argument("--live", action="store_true", help="Real trades (needs .env)")
    parser.add_argument("--status", action="store_true", help="Show balance + positions")
    parser.add_argument("--once", action="store_true", help="Single scan, no loop")
    parser.add_argument("--test-forecast", action="store_true", help="Test forecast engine")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in minutes (default 60)")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.test_forecast:
        from forecast import get_forecasts
        for city in ["seoul", "singapore", "toronto"]:
            fcs = get_forecasts(city, days_ahead=3)
            for d, fc in sorted(fcs.items()):
                print(f"{city:>10} {d} | {fc.n_members} members | "
                      f"mean {fc.mean:.1f}° ± {fc.std:.1f}° | "
                      f"P(>20°)={fc.prob_above(20):.0%} | "
                      f"agree {fc.agreement:.0%}")
        return

    if args.live:
        try:
            executor = ClobExecutor.from_env()
        except Exception as e:
            err(f"Cannot start live mode: {e}")
            return
    else:
        executor = ClobExecutor.paper()

    config = StrategyConfig()
    state = load_state()

    mode = f"{G}LIVE{X}" if args.live else f"{Y}PAPER{X}"
    print(f"\n{B}{'='*55}{X}")
    print(f"  {B}{C}🌤  PolyWeather{X}")
    print(f"{B}{'='*55}{X}")
    print(f"  Mode:     {mode}")
    print(f"  Cities:   {len(CITIES)}")
    print(f"  Balance:  ${state['balance']:.2f}")
    print(f"  Interval: {args.interval} min")
    print(f"  Forecast: ECMWF ensemble + LightGBM calibration")
    print(f"  Logging:  logs/polyweather_*.log")
    if not args.live:
        print(f"  {Y}Use --live to place real trades{X}")
    print(f"  Ctrl+C to stop\n")

    if args.once:
        state = run_scan(executor, config, state)
        return

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{D}[{now}] Starting scan...{X}")
            state = run_scan(executor, config, state)
        except KeyboardInterrupt:
            print(f"\n  Saving state and exiting...")
            save_state(state)
            print(f"  Done. Bye!")
            break
        except Exception as e:
            err(f"Scan error: {e}")
            import traceback
            traceback.print_exc()

        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print(f"\n  Saving state and exiting...")
            save_state(state)
            print(f"  Done. Bye!")
            break


if __name__ == "__main__":
    main()