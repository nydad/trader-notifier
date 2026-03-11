#!/usr/bin/env python3
"""Discord loop polling — checks #코스피 channel every N seconds for commands.

Supported commands (type in Discord):
  !장전체크  or  !premarket   → run premarket check, post result
  !시그널    or  !signal      → run signal engine, post result
  !뉴스     or  !news        → check news keywords
  !상관분석  or  !correlate   → run correlation (takes a while)
  !상태     or  !status      → show system status
  !도움말   or  !help        → show available commands

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/discord_loop.py
    PYTHONPATH=src python scripts/discord_loop.py --interval 60   # 60 seconds
"""
from __future__ import annotations

import argparse
import io
import os
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger("discord_loop")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1481273084916011008"   # #코스피
ADMIN_USER_ID = "1466390278771576894"
API_BASE = "https://discord.com/api/v10"

HEADERS = {
    "Authorization": f"Bot {BOT_TOKEN}",
    "Content-Type": "application/json",
}

COMMANDS = {
    "!장전체크": "premarket",
    "!premarket": "premarket",
    "!시그널": "signal",
    "!signal": "signal",
    "!뉴스": "news",
    "!news": "news",
    "!상관분석": "correlate",
    "!correlate": "correlate",
    "!상태": "status",
    "!status": "status",
    "!도움말": "help",
    "!help": "help",
}


# ---------------------------------------------------------------------------
# Discord API helpers
# ---------------------------------------------------------------------------

def get_messages(limit: int = 10, after: str | None = None) -> list[dict]:
    """Fetch recent messages from #코스피 channel."""
    url = f"{API_BASE}/channels/{CHANNEL_ID}/messages?limit={limit}"
    if after:
        url += f"&after={after}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"GET messages failed: {resp.status_code}")
    except Exception as e:
        logger.error(f"GET messages error: {e}")
    return []


def send_message(content: str, embed: dict | None = None) -> bool:
    """Send message to #코스피 channel."""
    url = f"{API_BASE}/channels/{CHANNEL_ID}/messages"
    payload: dict = {}
    if content:
        payload["content"] = content[:2000]
    if embed:
        payload["embeds"] = [embed]
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.error(f"Send failed: {e}")
        return False


def send_typing():
    """Show typing indicator."""
    try:
        requests.post(f"{API_BASE}/channels/{CHANNEL_ID}/typing", headers=HEADERS, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def handle_premarket() -> str:
    """Run premarket check and return formatted result."""
    send_typing()
    try:
        # Import and run directly for richer output
        import numpy as np
        from scripts.premarket import _fetch_realtime, REALTIME_TICKERS, _interpret, _build_discord_embed

        data = _fetch_realtime(REALTIME_TICKERS)
        if not data:
            return "데이터 수집 실패"

        # Build text summary
        lines = ["**장 시작 전 체크 (실시간)**\n"]
        sorted_items = sorted(data.items(), key=lambda x: x[1]["priority"])
        for key, d in sorted_items:
            ret = d["change_pct"]
            emoji = "🟢" if ret * d["dir"] > 0.001 else ("🔴" if ret * d["dir"] < -0.001 else "🟡")
            lines.append(f"{emoji} **{d['label']}**: {d['current']:,.2f} ({ret:+.2%})")
            lines.append(f"   → {_interpret(key, d, data)}")

        # Run Bayesian
        from kospi_corr.engine.bayesian import BayesianEngine, BayesianSignal
        from kospi_corr.domain.types import SignalDirection

        signals = []
        for key, d in data.items():
            ret = d["change_pct"]
            std = d["std_ret"] if d["std_ret"] > 0 else 0.01
            z = (ret - d["mean_ret"]) / std
            z_dir = z * d["dir"]
            prob = float(1 / (1 + np.exp(-z_dir)))
            prob = float(np.clip(prob, 0.01, 0.99))
            signals.append(BayesianSignal(
                name=key, probability=prob,
                weight=d["weight"], category=d["cat"],
                age_seconds=0.0, tau_seconds=14400.0,
            ))

        engine = BayesianEngine(prior=0.5, neutral_band=0.05)
        result = engine.compute(signals, vkospi=18.0, trend_slope=data.get("sp500", {}).get("change_pct", 0.0))

        dlabels = {SignalDirection.LONG: "🟢 LONG (매수)", SignalDirection.SHORT: "🔴 SHORT (매도)", SignalDirection.NEUTRAL: "🟡 NEUTRAL (관망)"}
        lines.append(f"\n**종합 판정: {dlabels.get(result.direction, 'UNKNOWN')}**")
        lines.append(f"LONG {result.long_probability:.1%} / SHORT {result.short_probability:.1%} (신뢰도 {result.confidence:.0%})")

        # Top signals
        if result.key_signals:
            lines.append("\n**기여도 Top 5:**")
            for s in result.key_signals[:5]:
                arrow = "↑" if s["contribution"] > 0 else "↓"
                lines.append(f"  {arrow} {s['name']}: P(LONG)={s['probability']:.0%}, 기여={s['contribution']:+.3f}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Premarket failed: {e}", exc_info=True)
        return f"장전체크 실패: {e}"


def handle_news() -> str:
    """Check news keywords."""
    send_typing()
    try:
        from kospi_corr.collectors.news import NewsCollector
        news = NewsCollector()
        sig = news.collect_signal()

        lines = [f"**뉴스 모니터링** (Urgency: {sig.urgency_level}, Sentiment: {sig.sentiment_score:.2f})\n"]
        hits = {k: v for k, v in sig.keyword_hits.items() if v > 0}
        if hits:
            lines.append(f"Keywords: {', '.join(f'{k}={v}' for k, v in hits.items())}")

        for art in sig.top_articles[:7]:
            kw = ", ".join(art.matched_keywords[:3])
            lines.append(f"• [{art.source}] {art.title} ({kw})")

        return "\n".join(lines)
    except Exception as e:
        return f"뉴스 체크 실패: {e}"


def handle_signal() -> str:
    """Run signal engine."""
    send_typing()
    # Delegate to premarket for now (same engine)
    return handle_premarket()


def handle_status() -> str:
    """System status."""
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    return (
        f"**시스템 상태**\n"
        f"시간: {now.strftime('%Y-%m-%d %H:%M KST')}\n"
        f"엔진: Bayesian Log-Odds Pooling\n"
        f"데이터: yfinance 실시간 호가\n"
        f"뉴스: 연합뉴스/한경/매경 RSS\n"
        f"채널: #코스피\n"
        f"루프 상태: 실행 중"
    )


def handle_help() -> str:
    return (
        "**사용 가능한 명령어:**\n"
        "`!장전체크` — 장 시작 전 체크 (환율→선물→유가→뉴스)\n"
        "`!시그널` — Bayesian 시그널 확인\n"
        "`!뉴스` — 뉴스 키워드 모니터링\n"
        "`!상태` — 시스템 상태 확인\n"
        "`!도움말` — 이 메시지"
    )


HANDLERS = {
    "premarket": handle_premarket,
    "signal": handle_signal,
    "news": handle_news,
    "correlate": lambda: "상관분석은 시간이 오래 걸립니다. CLI에서 실행하세요:\n`PYTHONPATH=src python scripts/analyze.py`",
    "status": handle_status,
    "help": handle_help,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(interval: int = 30):
    """Poll Discord every `interval` seconds for commands."""
    logger.info(f"Discord loop started. Polling every {interval}s. Channel: #{CHANNEL_ID}")
    send_message("🤖 KOSPI Signal Bot 루프 시작. `!도움말`로 명령어 확인.")

    last_message_id = None

    # Get the latest message ID to avoid processing old messages
    msgs = get_messages(limit=1)
    if msgs:
        last_message_id = msgs[0]["id"]
        logger.info(f"Starting from message ID: {last_message_id}")

    while True:
        try:
            # Fetch new messages since last seen
            msgs = get_messages(limit=10, after=last_message_id)
            if msgs:
                # Discord returns newest first, reverse for chronological order
                msgs.reverse()
                for msg in msgs:
                    msg_id = msg["id"]
                    author = msg.get("author", {})
                    author_id = author.get("id", "")
                    content = msg.get("content", "").strip()
                    is_bot = author.get("bot", False)

                    # Skip bot messages
                    if is_bot:
                        last_message_id = msg_id
                        continue

                    # Check if it's a command
                    cmd_word = content.split()[0].lower() if content else ""
                    if cmd_word in COMMANDS:
                        cmd_name = COMMANDS[cmd_word]
                        handler = HANDLERS.get(cmd_name)
                        if handler:
                            logger.info(f"Command: {cmd_word} from {author.get('username', '?')}")
                            try:
                                response = handler()
                                send_message(response)
                            except Exception as e:
                                send_message(f"명령 실행 실패: {e}")
                                logger.error(f"Handler error: {e}", exc_info=True)

                    last_message_id = msg_id

        except KeyboardInterrupt:
            logger.info("Loop stopped by user")
            send_message("🛑 Bot 루프 종료.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Discord command polling loop")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_loop(interval=args.interval)


if __name__ == "__main__":
    main()
