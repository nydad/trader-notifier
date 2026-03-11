#!/usr/bin/env python3
"""Live signal monitor — runs during market hours, sends Discord alerts on signal change.

Checks indicators every N minutes. When significant change detected:
  → Re-runs Bayesian engine
  → If direction changes or confidence crosses threshold → Discord alert

Triggers:
  - USD/KRW moves > 0.3% from last check
  - S&P500 futures move > 0.5%
  - WTI moves > 2%
  - VIX spikes > 5%
  - Direction flips (LONG↔SHORT)
  - News urgency → HIGH/CRITICAL

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/live_monitor.py                    # 5분 간격
    PYTHONPATH=src python scripts/live_monitor.py --interval 3       # 3분 간격
    PYTHONPATH=src python scripts/live_monitor.py --threshold 0.6    # 신뢰도 60% 이상만
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger("live_monitor")
KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1481273084916011008"  # #코스피
API_BASE = "https://discord.com/api/v10"

# Tickers to monitor (subset of premarket — fastest to fetch)
MONITOR_TICKERS = {
    "usd_krw":    {"yf": "KRW=X",    "label": "USD/KRW",  "dir": -1, "weight": 2.5, "cat": "direction_driver", "trigger_pct": 0.003},
    "sp500_fut":  {"yf": "ES=F",     "label": "S&P선물",   "dir": +1, "weight": 2.0, "cat": "direction_driver", "trigger_pct": 0.005},
    "nasdaq_fut": {"yf": "NQ=F",     "label": "NQ선물",    "dir": +1, "weight": 1.5, "cat": "direction_driver", "trigger_pct": 0.005},
    "wti_crude":  {"yf": "CL=F",     "label": "WTI",      "dir": +1, "weight": 1.2, "cat": "direction_driver", "trigger_pct": 0.020},
    "dxy":        {"yf": "DX-Y.NYB", "label": "DXY",      "dir": -1, "weight": 1.0, "cat": "direction_driver", "trigger_pct": 0.005},
    "vix":        {"yf": "^VIX",     "label": "VIX",      "dir": -1, "weight": 1.0, "cat": "sentiment",        "trigger_pct": 0.050},
}

MARKET_OPEN = (9, 0)    # 09:00 KST
MARKET_CLOSE = (15, 30)  # 15:30 KST


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

class MonitorState:
    """Tracks previous state to detect meaningful changes."""

    def __init__(self):
        self.last_direction = None        # "LONG" / "SHORT" / "NEUTRAL"
        self.last_prices: dict[str, float] = {}
        self.last_signal_time: datetime | None = None
        self.signal_count = 0
        self.cooldown_seconds = 120       # min 2 min between alerts

    def price_changed(self, key: str, current_price: float, trigger_pct: float) -> bool:
        """Check if price moved enough from last recorded price."""
        if key not in self.last_prices:
            self.last_prices[key] = current_price
            return False
        prev = self.last_prices[key]
        if prev == 0:
            return False
        change = abs(current_price - prev) / prev
        return change >= trigger_pct

    def direction_changed(self, new_direction: str) -> bool:
        """Check if signal direction flipped."""
        if self.last_direction is None:
            return True  # first run
        return self.last_direction != new_direction and new_direction != "NEUTRAL"

    def cooldown_passed(self) -> bool:
        if self.last_signal_time is None:
            return True
        elapsed = (datetime.now(KST) - self.last_signal_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def update(self, direction: str, prices: dict[str, float]):
        self.last_direction = direction
        self.last_prices.update(prices)
        self.last_signal_time = datetime.now(KST)
        self.signal_count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_market_hours() -> bool:
    """Check if KRX is currently open."""
    now = datetime.now(KST)
    if now.weekday() >= 5:  # weekend
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def fetch_quotes() -> dict[str, dict]:
    """Fetch latest quotes for monitored tickers."""
    import yfinance as yf

    results = {}
    for key, info in MONITOR_TICKERS.items():
        try:
            _time.sleep(0.2)
            t = yf.Ticker(info["yf"])

            try:
                fi = t.fast_info
                current = fi.get("lastPrice") or fi.get("last_price")
                prev = fi.get("previousClose") or fi.get("previous_close")
            except Exception:
                current, prev = None, None

            if current is None or prev is None:
                hist = t.history(period="5d")
                if hist.empty or len(hist) < 2:
                    continue
                col = "Close" if "Close" in hist.columns else "close"
                current = float(hist[col].iloc[-1])
                prev = float(hist[col].iloc[-2])

            # 30d stats for z-score
            hist30 = t.history(period="1mo")
            if hist30 is not None and len(hist30) > 5:
                col = "Close" if "Close" in hist30.columns else "close"
                rets = hist30[col].pct_change().dropna()
                mean_r, std_r = float(rets.mean()), float(rets.std())
            else:
                mean_r, std_r = 0.0, 0.01

            change = (current - prev) / prev if prev else 0.0
            results[key] = {
                "current": float(current),
                "prev_close": float(prev),
                "change_pct": float(change),
                "mean_ret": mean_r,
                "std_ret": std_r,
                **info,
            }
        except Exception as e:
            logger.debug(f"{info['yf']} failed: {e}")

    return results


def run_bayesian(data: dict) -> dict:
    """Run Bayesian engine on current data. Returns result dict."""
    from kospi_corr.engine.bayesian import BayesianEngine, BayesianSignal
    from kospi_corr.domain.types import SignalDirection

    signals = []
    for key, d in data.items():
        ret = d["change_pct"]
        std = d["std_ret"] if d["std_ret"] > 0 else 0.01
        z = (ret - d["mean_ret"]) / std
        z_dir = z * d["dir"]
        prob = float(np.clip(1 / (1 + np.exp(-z_dir)), 0.01, 0.99))

        signals.append(BayesianSignal(
            name=key, probability=prob,
            weight=d["weight"], category=d["cat"],
            age_seconds=0.0, tau_seconds=14400.0,
        ))

    engine = BayesianEngine(prior=0.5, neutral_band=0.05)
    sp_ret = data.get("sp500_fut", {}).get("change_pct", 0.0)
    result = engine.compute(signals, vkospi=18.0, trend_slope=sp_ret)

    dir_map = {
        SignalDirection.LONG: "LONG",
        SignalDirection.SHORT: "SHORT",
        SignalDirection.NEUTRAL: "NEUTRAL",
    }

    return {
        "direction": dir_map[result.direction],
        "long_prob": result.long_probability,
        "short_prob": result.short_probability,
        "confidence": result.confidence,
        "key_signals": result.key_signals,
    }


def send_discord(content: str = "", embed: dict | None = None) -> bool:
    headers = {"Authorization": f"Bot {BOT_TOKEN}", "Content-Type": "application/json"}
    url = f"{API_BASE}/channels/{CHANNEL_ID}/messages"
    payload = {}
    if content:
        payload["content"] = content[:2000]
    if embed:
        payload["embeds"] = [embed]
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.error(f"Discord send failed: {e}")
        return False


def build_signal_embed(sig: dict, data: dict, trigger_reason: str) -> dict:
    """Build Discord embed for a live signal alert."""
    color_map = {"LONG": 0x00FF00, "SHORT": 0xFF0000, "NEUTRAL": 0xFFFF00}
    emoji_map = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "🟡"}
    dir_kr = {"LONG": "매수(LONG)", "SHORT": "매도(SHORT)", "NEUTRAL": "관망"}

    d = sig["direction"]
    now = datetime.now(KST)

    fields = []
    # Current prices
    for key in ["usd_krw", "sp500_fut", "nasdaq_fut", "wti_crude", "dxy", "vix"]:
        if key in data:
            dd = data[key]
            ret = dd["change_pct"]
            arrow = "📈" if ret > 0 else "📉" if ret < 0 else "➡️"
            fields.append({
                "name": dd["label"],
                "value": f"{arrow} {dd['current']:,.2f} ({ret:+.2%})",
                "inline": True,
            })

    fields.append({
        "name": f"{emoji_map[d]} LONG / SHORT",
        "value": f"**{sig['long_prob']:.1%}** / {sig['short_prob']:.1%}\n신뢰도: {sig['confidence']:.0%}",
        "inline": False,
    })

    # Top 3 contributors
    if sig["key_signals"]:
        top3 = sig["key_signals"][:3]
        contrib_text = "\n".join(
            f"{'↑' if s['contribution'] > 0 else '↓'} {s['name']}: {s['contribution']:+.3f}"
            for s in top3
        )
        fields.append({"name": "주요 기여", "value": contrib_text, "inline": False})

    fields.append({"name": "트리거", "value": trigger_reason, "inline": False})

    return {
        "title": f"📊 실시간 시그널 — {dir_kr[d]}",
        "color": color_map.get(d, 0xFFFF00),
        "fields": fields,
        "footer": {"text": f"KOSPI Signal Bot | {now.strftime('%H:%M:%S KST')}"},
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def monitor_loop(interval_min: int = 5, confidence_threshold: float = 0.55):
    state = MonitorState()
    logger.info(f"Live monitor started. Interval: {interval_min}min, threshold: {confidence_threshold:.0%}")
    send_discord(f"📊 KOSPI 실시간 모니터 시작 ({interval_min}분 간격, 신뢰도 {confidence_threshold:.0%} 이상 알림)")

    while True:
        try:
            now = datetime.now(KST)

            # Only run during market hours (with 30min pre-market buffer)
            pre_market_start = (8, 30)
            t = (now.hour, now.minute)
            if now.weekday() >= 5:
                logger.info("Weekend — sleeping 1h")
                _time.sleep(3600)
                continue

            if t < pre_market_start or t > (15, 35):
                logger.debug(f"{now.strftime('%H:%M')} — outside market hours, sleeping")
                _time.sleep(300)
                continue

            # Fetch data
            logger.info(f"Fetching quotes... ({now.strftime('%H:%M:%S')})")
            data = fetch_quotes()
            if not data:
                logger.warning("No data fetched")
                _time.sleep(60)
                continue

            # Check for significant changes
            triggers = []
            current_prices = {}
            for key, d in data.items():
                current_prices[key] = d["current"]
                if state.price_changed(key, d["current"], d["trigger_pct"]):
                    triggers.append(f"{d['label']} 변동 ({d['change_pct']:+.2%})")

            # Run Bayesian
            sig = run_bayesian(data)

            # Check direction change
            if state.direction_changed(sig["direction"]):
                triggers.append(f"방향 전환 → {sig['direction']}")

            # Determine if we should alert
            should_alert = False
            trigger_reason = ""

            if triggers and state.cooldown_passed():
                if sig["confidence"] >= confidence_threshold:
                    should_alert = True
                    trigger_reason = " | ".join(triggers)

            # Log current state
            logger.info(
                f"{sig['direction']} L={sig['long_prob']:.1%} S={sig['short_prob']:.1%} "
                f"conf={sig['confidence']:.0%} triggers={len(triggers)}"
            )

            if should_alert:
                logger.info(f"ALERT: {trigger_reason}")
                embed = build_signal_embed(sig, data, trigger_reason)
                if send_discord(embed=embed):
                    state.update(sig["direction"], current_prices)
                    logger.info(f"Signal #{state.signal_count} sent to Discord")
                else:
                    logger.warning("Discord send failed")
            else:
                # Still update prices even if no alert
                state.last_prices.update(current_prices)
                if state.last_direction is None:
                    state.last_direction = sig["direction"]

        except KeyboardInterrupt:
            logger.info("Monitor stopped")
            send_discord("🛑 실시간 모니터 종료")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}", exc_info=True)

        _time.sleep(interval_min * 60)


def main():
    parser = argparse.ArgumentParser(description="KOSPI Live Signal Monitor")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in minutes (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Min confidence for alert (default: 0.55)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    monitor_loop(interval_min=args.interval, confidence_threshold=args.threshold)


if __name__ == "__main__":
    main()
