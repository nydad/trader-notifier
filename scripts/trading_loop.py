#!/usr/bin/env python3
"""Unified KOSPI ETF trading loop — signals, reports, position tracking, Discord.

Replaces: live_monitor.py, discord_loop.py, signal.py

Features:
  - 5분 주기: Bayesian 시그널 + 역사적 패턴 → Discord 전송
  - 30분 주기: 뉴스 + 외국인/기관 동향 + 종합 리포트
  - 포지션 추적: 매수 보고 → 매도 시그널 + 전략 안내
  - Discord 명령 처리: !장전체크, !시그널, !뉴스, !포지션, !도움말

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/trading_loop.py
    PYTHONPATH=src python scripts/trading_loop.py --signal-interval 3   # 3분 시그널
    PYTHONPATH=src python scripts/trading_loop.py --no-pattern          # 패턴분석 OFF
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger("trading_loop")
KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1481273084916011008"
ADMIN_USER_ID = "1466390278771576894"
API_BASE = "https://discord.com/api/v10"

MONITOR_TICKERS = {
    "usd_krw":    {"yf": "KRW=X",    "label": "USD/KRW",  "dir": -1, "weight": 2.5, "cat": "direction_driver"},
    "sp500_fut":  {"yf": "ES=F",     "label": "S&P선물",   "dir": +1, "weight": 2.0, "cat": "direction_driver"},
    "nasdaq_fut": {"yf": "NQ=F",     "label": "NQ선물",    "dir": +1, "weight": 1.5, "cat": "direction_driver"},
    "wti_crude":  {"yf": "CL=F",     "label": "WTI",      "dir": +1, "weight": 1.2, "cat": "direction_driver"},
    "dxy":        {"yf": "DX-Y.NYB", "label": "DXY",      "dir": -1, "weight": 1.0, "cat": "direction_driver"},
    "vix":        {"yf": "^VIX",     "label": "VIX",      "dir": -1, "weight": 1.0, "cat": "sentiment"},
}

# ETF aliases for position parsing
ETF_ALIASES: dict[str, dict] = {
    "인버스":       {"name": "KODEX 200선물인버스2X",   "code": "252670", "dir": "short"},
    "인버스2x":     {"name": "KODEX 200선물인버스2X",   "code": "252670", "dir": "short"},
    "코스닥레버":   {"name": "KODEX 코스닥150레버리지",  "code": "233740", "dir": "long"},
    "코스닥인버스": {"name": "KODEX 코스닥150선물인버스", "code": "251340", "dir": "short"},
    "레버리지":     {"name": "KODEX 레버리지",          "code": "122630", "dir": "long"},
    "반도체레버":   {"name": "KODEX 반도체레버리지",     "code": "091170", "dir": "long"},
    "반도체":       {"name": "KODEX 반도체레버리지",     "code": "091170", "dir": "long"},
    "방산레버":     {"name": "KODEX 방산TOP10레버리지",  "code": "472170", "dir": "long"},
    "방산":         {"name": "KODEX 방산TOP10레버리지",  "code": "472170", "dir": "long"},
}


# ---------------------------------------------------------------------------
# Position tracker
# ---------------------------------------------------------------------------

@dataclass
class Position:
    etf_name: str
    code: str
    direction: str     # "long" or "short"
    quantity: int
    price: float
    entry_time: datetime
    signal_at_entry: str   # signal direction when entered

    def is_opposing_signal(self, current_signal: str) -> bool:
        """Check if current signal opposes this position."""
        if self.direction == "short":
            return current_signal == "LONG"
        return current_signal == "SHORT"


class PositionTracker:
    """Tracks user's open positions for sell signal management."""

    def __init__(self):
        self.positions: list[Position] = []

    def add(self, etf_name: str, code: str, direction: str,
            qty: int, price: float, current_signal: str) -> Position:
        pos = Position(
            etf_name=etf_name, code=code, direction=direction,
            quantity=qty, price=price,
            entry_time=datetime.now(KST),
            signal_at_entry=current_signal,
        )
        self.positions.append(pos)
        return pos

    def remove_by_name(self, name_part: str) -> Position | None:
        for i, p in enumerate(self.positions):
            if name_part in p.etf_name:
                return self.positions.pop(i)
        return None

    def clear(self):
        self.positions.clear()

    def check_sell_signals(self, current_signal: str) -> list[tuple[Position, str]]:
        """Check all positions for sell triggers. Returns (position, reason) pairs."""
        alerts = []
        for pos in self.positions:
            if pos.is_opposing_signal(current_signal):
                if pos.direction == "short":
                    reason = f"LONG 시그널 전환 → {pos.etf_name} 매도 고려"
                else:
                    reason = f"SHORT 시그널 전환 → {pos.etf_name} 매도 고려"
                alerts.append((pos, reason))
        return alerts

    def format_positions(self) -> str:
        if not self.positions:
            return "보유 포지션 없음"
        lines = []
        for p in self.positions:
            d = "숏" if p.direction == "short" else "롱"
            lines.append(
                f"• {p.etf_name} ({d}) {p.quantity:,}주 @{p.price:,.0f}원 "
                f"[진입: {p.entry_time.strftime('%H:%M')}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

def parse_user_message(text: str) -> dict | None:
    """Parse user Discord message for trading actions.

    Returns dict with keys: action, etf, code, direction, quantity, price
    or None if not a trading message.
    """
    text_lower = text.strip().lower().replace(" ", "")

    # Detect buy/sell action
    is_buy = bool(re.search(r"매수|진입|샀|삼|롱진입|숏진입", text_lower))
    is_sell = bool(re.search(r"매도|팔|익절|손절|청산|탈출|매도완료|전량매도", text_lower))

    if not is_buy and not is_sell:
        return None

    action = "buy" if is_buy else "sell"

    # Find ETF
    etf_info = None
    for alias, info in ETF_ALIASES.items():
        if alias in text_lower:
            etf_info = info
            break

    if not etf_info and action == "sell":
        # For sell, might just say "매도 완료" without specifying ETF
        return {"action": "sell_all", "etf": None, "code": None,
                "direction": None, "quantity": 0, "price": 0}

    if not etf_info:
        return None

    # Extract price
    price_match = re.search(r"(\d[\d,]*)원", text)
    price = float(price_match.group(1).replace(",", "")) if price_match else 0

    # Extract quantity
    qty = 0
    qty_match = re.search(r"(\d[\d,]*)\s*주", text)
    if qty_match:
        qty = int(qty_match.group(1).replace(",", ""))
    # Handle 만주 shorthand
    man_match = re.search(r"(\d+)\s*만\s*주?", text)
    if man_match:
        qty = int(man_match.group(1)) * 10000

    return {
        "action": action,
        "etf": etf_info["name"],
        "code": etf_info["code"],
        "direction": etf_info["dir"],
        "quantity": qty,
        "price": price,
    }


def generate_strategy(pos: Position, signal_dir: str, confidence: float) -> str:
    """Generate sell strategy text for a position."""
    lines = []
    lines.append(f"**{pos.etf_name}** {pos.quantity:,}주 @{pos.price:,.0f}원")

    if pos.direction == "short":
        # 인버스 ETF — profits when KOSPI falls
        lines.append(f"포지션: 숏 (KOSPI 하락 베팅)")
        if signal_dir == "SHORT":
            lines.append(f"현재 시그널: SHORT {confidence:.0%} → **보유 유리**")
            target = pos.price * 1.02
            stop = pos.price * 0.988
        else:
            lines.append(f"현재 시그널: LONG → **매도 고려!**")
            target = pos.price * 1.005  # tight target
            stop = pos.price * 0.995
    else:
        # 레버리지 ETF — profits when underlying rises
        lines.append(f"포지션: 롱 (상승 베팅)")
        if signal_dir == "LONG":
            lines.append(f"현재 시그널: LONG {confidence:.0%} → **보유 유리**")
            target = pos.price * 1.025
            stop = pos.price * 0.985
        else:
            lines.append(f"현재 시그널: SHORT → **매도 고려!**")
            target = pos.price * 1.005
            stop = pos.price * 0.995

    lines.append(f"\n**매매 전략:**")
    lines.append(f"🎯 목표가: {target:,.0f}원 ({(target/pos.price-1)*100:+.1f}%)")
    lines.append(f"🛑 손절가: {stop:,.0f}원 ({(stop/pos.price-1)*100:+.1f}%)")
    lines.append(f"\n**매도 조건:**")
    if pos.direction == "short":
        lines.append("1. LONG 시그널 전환 → 즉시 매도")
        lines.append("2. USD/KRW -0.3% 이상 하락 → 매도 준비")
        lines.append("3. 목표가 도달 → 분할 매도")
        lines.append("4. 장 마감 30분 전 → 미청산 시 매도 고려")
        lines.append("\n**핵심 모니터:**")
        lines.append("- USD/KRW (가장 중요): 상승 유지 = 보유")
        lines.append("- S&P선물: 하락 유지 = 보유")
    else:
        lines.append("1. SHORT 시그널 전환 → 즉시 매도")
        lines.append("2. VIX 급등(+5%) → 매도 준비")
        lines.append("3. 목표가 도달 → 분할 매도")
        lines.append("4. 장 마감 30분 전 → 미청산 시 매도 고려")
        lines.append("\n**핵심 모니터:**")
        lines.append("- S&P선물/NQ선물: 상승 유지 = 보유")
        lines.append("- VIX: 하락 유지 = 보유")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_quotes() -> dict[str, dict]:
    """Fetch latest quotes for monitored tickers via yfinance."""
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

            hist30 = t.history(period="1mo")
            if hist30 is not None and len(hist30) > 5:
                col = "Close" if "Close" in hist30.columns else "close"
                rets = hist30[col].pct_change().dropna()
                mean_r, std_r = float(rets.mean()), float(rets.std())
            else:
                mean_r, std_r = 0.0, 0.01

            change = (current - prev) / prev if prev else 0.0
            results[key] = {
                "current": float(current), "prev_close": float(prev),
                "change_pct": float(change), "mean_ret": mean_r, "std_ret": std_r,
                **info,
            }
        except Exception as e:
            logger.debug(f"{info['yf']} failed: {e}")

    return results


def run_bayesian(data: dict) -> dict:
    """Run Bayesian engine on current data."""
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

    # Fetch real VKOSPI
    try:
        from kospi_corr.data.providers.naver_scraper import get_vkospi_level
        vkospi_val, _ = get_vkospi_level()
    except Exception:
        vkospi_val = 25.0

    result = engine.compute(signals, vkospi=vkospi_val, trend_slope=sp_ret)

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


# ---------------------------------------------------------------------------
# Discord helpers
# ---------------------------------------------------------------------------

_DISCORD_HEADERS = {
    "Authorization": f"Bot {BOT_TOKEN}",
    "Content-Type": "application/json",
}


def send_discord(content: str = "", embed: dict | None = None) -> bool:
    url = f"{API_BASE}/channels/{CHANNEL_ID}/messages"
    payload = {}
    if content:
        payload["content"] = content[:2000]
    if embed:
        payload["embeds"] = [embed]
    try:
        resp = requests.post(url, json=payload, headers=_DISCORD_HEADERS, timeout=10)
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.error(f"Discord send failed: {e}")
        return False


def get_discord_messages(limit: int = 10, after: str | None = None) -> list[dict]:
    url = f"{API_BASE}/channels/{CHANNEL_ID}/messages?limit={limit}"
    if after:
        url += f"&after={after}"
    try:
        resp = requests.get(url, headers=_DISCORD_HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Discord get failed: {e}")
    return []


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------

def build_signal_embed(sig: dict, data: dict, pattern_text: str = "") -> dict:
    """Build 5-min signal embed."""
    d = sig["direction"]
    color_map = {"LONG": 0x00FF00, "SHORT": 0xFF0000, "NEUTRAL": 0xFFFF00}
    emoji_map = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "🟡"}
    dir_kr = {"LONG": "매수(LONG)", "SHORT": "매도(SHORT)", "NEUTRAL": "관망"}
    now = datetime.now(KST)

    fields = []

    # Indicator prices
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

    # Signal result
    fields.append({
        "name": f"{emoji_map[d]} LONG / SHORT",
        "value": f"**{sig['long_prob']:.1%}** / {sig['short_prob']:.1%}\n신뢰도: {sig['confidence']:.0%}",
        "inline": False,
    })

    # Top contributors
    if sig["key_signals"]:
        top = sig["key_signals"][:4]
        contrib = "\n".join(
            f"{'↑' if s['contribution']>0 else '↓'} {s['name']}: "
            f"P(L)={s['probability']:.0%}, 기여={s['contribution']:+.3f}"
            for s in top
        )
        fields.append({"name": "주요 기여", "value": contrib, "inline": False})

    # Pattern analysis
    if pattern_text:
        fields.append({"name": "📜 역사적 패턴", "value": pattern_text[:400], "inline": False})

    # Trading interpretation
    if d == "SHORT":
        interp = "→ 인버스2X 매수 유리 (KOSPI 하락 예상)"
    elif d == "LONG":
        interp = "→ 레버리지 ETF 매수 유리 (KOSPI 상승 예상)"
    else:
        interp = "→ 관망 (시그널 약함)"
    fields.append({"name": "💡 매매 해석", "value": interp, "inline": False})

    return {
        "title": f"📊 시그널 | {dir_kr[d]} | {now.strftime('%H:%M')} KST",
        "color": color_map.get(d, 0xFFFF00),
        "fields": fields,
        "footer": {"text": f"KOSPI Signal Bot | 5분 자동"},
    }


def build_report_embed(sig: dict, data: dict, news_text: str,
                       investor_text: str, pattern_text: str,
                       position_text: str, considerations: list[str]) -> dict:
    """Build 30-min comprehensive report embed."""
    d = sig["direction"]
    color_map = {"LONG": 0x00FF00, "SHORT": 0xFF0000, "NEUTRAL": 0xFFFF00}
    now = datetime.now(KST)

    fields = []

    # Market snapshot (compact)
    snapshot_lines = []
    for key in ["usd_krw", "sp500_fut", "nasdaq_fut", "wti_crude", "vix"]:
        if key in data:
            dd = data[key]
            ret = dd["change_pct"]
            arrow = "↗" if ret > 0 else "↘" if ret < 0 else "→"
            snapshot_lines.append(f"{dd['label']} {dd['current']:,.2f} ({ret:+.2%}) {arrow}")
    fields.append({
        "name": "📊 시장 현황",
        "value": "\n".join(snapshot_lines),
        "inline": False,
    })

    # Signal summary
    fields.append({
        "name": f"시그널: {'🟢' if d == 'LONG' else '🔴' if d == 'SHORT' else '🟡'} {d}",
        "value": f"LONG {sig['long_prob']:.1%} / SHORT {sig['short_prob']:.1%} (신뢰도 {sig['confidence']:.0%})",
        "inline": False,
    })

    # Investor flow
    if investor_text:
        fields.append({"name": "👤 투자자 동향", "value": investor_text[:300], "inline": False})

    # News
    if news_text:
        fields.append({"name": "📰 뉴스", "value": news_text[:400], "inline": False})

    # Pattern
    if pattern_text:
        fields.append({"name": "📜 역사적 패턴", "value": pattern_text[:400], "inline": False})

    # Considerations
    if considerations:
        cons_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(considerations[:5]))
        fields.append({"name": "💡 고려사항", "value": cons_text, "inline": False})

    # Positions
    if position_text and position_text != "보유 포지션 없음":
        fields.append({"name": "📍 포지션", "value": position_text[:300], "inline": False})

    return {
        "title": f"📰 30분 시장 리포트 | {now.strftime('%H:%M')} KST",
        "color": color_map.get(d, 0xFFFF00),
        "fields": fields,
        "footer": {"text": "KOSPI Signal Bot | 30분 자동"},
    }


def build_position_embed(pos: Position, strategy: str) -> dict:
    """Build position alert embed."""
    color = 0xFF4444 if pos.direction == "short" else 0x44FF44
    emoji = "🔴" if pos.direction == "short" else "🟢"
    return {
        "title": f"{emoji} 포지션 등록 — {pos.etf_name}",
        "description": strategy,
        "color": color,
        "footer": {"text": f"진입 {pos.entry_time.strftime('%H:%M KST')} | KOSPI Signal Bot"},
    }


# ---------------------------------------------------------------------------
# Consideration generator
# ---------------------------------------------------------------------------

def generate_considerations(sig: dict, data: dict, investor: dict,
                            prev_data: dict | None) -> list[str]:
    """Generate actionable considerations based on all available data."""
    cons = []
    d = sig["direction"]

    # FX trend
    fx = data.get("usd_krw", {})
    if fx:
        ret = fx.get("change_pct", 0)
        if ret > 0.003:
            cons.append(f"원화 약세 ({ret:+.2%}) → 인버스 포지션 유리")
        elif ret < -0.003:
            cons.append(f"원화 강세 ({ret:+.2%}) → 레버리지 포지션 유리")

    # US market
    sp = data.get("sp500_fut", {})
    nq = data.get("nasdaq_fut", {})
    if sp and nq:
        sp_ret = sp.get("change_pct", 0)
        nq_ret = nq.get("change_pct", 0)
        if sp_ret < -0.005 and nq_ret < -0.005:
            cons.append(f"미국 선물 동반 하락 (S&P {sp_ret:+.2%}, NQ {nq_ret:+.2%}) → 코스피 하방 압력")
        elif sp_ret > 0.005 and nq_ret > 0.005:
            cons.append(f"미국 선물 동반 상승 → 코스피 상방 압력")

    # VIX
    vix = data.get("vix", {})
    if vix:
        vix_ret = vix.get("change_pct", 0)
        vix_val = vix.get("current", 0)
        if vix_ret > 0.05:
            cons.append(f"VIX 급등({vix_ret:+.1%}, 현재 {vix_val:.1f}) → 변동성 확대 주의")
        elif vix_val > 25:
            cons.append(f"VIX 높은 수준({vix_val:.1f}) → 급변동 가능성")

    # Oil
    oil = data.get("wti_crude", {})
    if oil:
        oil_ret = oil.get("change_pct", 0)
        if abs(oil_ret) > 0.02:
            cons.append(f"유가 급변({oil_ret:+.2%}) → 환율 2차 영향 주시")

    # Investor flow
    if investor:
        foreign = investor.get("foreign", 0)
        inst = investor.get("institution", 0)
        if foreign > 1000:
            cons.append(f"외국인 순매수 {foreign:+,.0f}억 → 상승 지지")
        elif foreign < -1000:
            cons.append(f"외국인 순매도 {abs(foreign):,.0f}억 → 하락 압력")
        if inst > 2000:
            cons.append(f"기관 대규모 매수 {inst:+,.0f}억 → 하방 지지")

    # DXY
    dxy = data.get("dxy", {})
    if dxy:
        dxy_ret = dxy.get("change_pct", 0)
        if dxy_ret > 0.003:
            cons.append(f"달러 강세({dxy_ret:+.2%}) → 원화 약세 압력 지속")

    # Direction change from previous
    if prev_data:
        prev_sig = prev_data.get("_prev_direction")
        if prev_sig and prev_sig != d and d != "NEUTRAL":
            cons.append(f"⚠️ 방향 전환! {prev_sig} → {d}")

    # Signal strength
    conf = sig.get("confidence", 0)
    if conf > 0.8:
        cons.append(f"시그널 신뢰도 높음({conf:.0%}) → 적극 대응")
    elif conf < 0.5:
        cons.append(f"시그널 신뢰도 낮음({conf:.0%}) → 소량 진입 또는 관망")

    return cons


# ---------------------------------------------------------------------------
# Discord command handlers
# ---------------------------------------------------------------------------

def handle_help() -> str:
    return (
        "**사용 가능한 명령어:**\n"
        "`!시그널` — 현재 시그널 (즉시)\n"
        "`!장전체크` — 장 시작 전 종합 체크\n"
        "`!뉴스` — 뉴스 키워드 모니터링\n"
        "`!포지션` — 보유 포지션 확인\n"
        "`!청산` — 모든 포지션 초기화\n"
        "`!상태` — 시스템 상태\n"
        "`!도움말` — 이 메시지\n\n"
        "**매수/매도 입력:**\n"
        "`인버스 250원 10만주 매수` — 포지션 등록\n"
        "`반도체레버 매수` — 가격/수량 없이도 OK\n"
        "`매도 완료` / `인버스 매도` — 포지션 해제"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class TradingLoop:
    """Unified trading loop: signals + reports + positions + Discord."""

    def __init__(
        self,
        signal_interval: int = 5,
        report_interval: int = 30,
        poll_interval: int = 15,
        enable_patterns: bool = True,
    ):
        self.signal_interval = signal_interval * 60  # seconds
        self.report_interval = report_interval * 60
        self.poll_interval = poll_interval

        self.positions = PositionTracker()
        self.last_signal_time: float = 0
        self.last_report_time: float = 0
        self.last_direction: str | None = None
        self.last_data: dict = {}
        self.last_signal: dict = {}
        self.last_message_id: str | None = None
        self.investor_data: dict = {}

        self.pattern_matcher = None
        if enable_patterns:
            try:
                from kospi_corr.analysis.pattern_matcher import PatternMatcher
                self.pattern_matcher = PatternMatcher(lookback_years=2)
                logger.info("PatternMatcher initialized")
            except Exception as e:
                logger.warning(f"PatternMatcher init failed: {e}")

    def run(self):
        logger.info(
            f"Trading loop started: signal={self.signal_interval//60}min, "
            f"report={self.report_interval//60}min, poll={self.poll_interval}s"
        )
        send_discord(
            f"🤖 Trading Bot 시작\n"
            f"시그널: {self.signal_interval//60}분 | 리포트: {self.report_interval//60}분\n"
            f"`!도움말`로 명령어 확인"
        )

        # Initialize last_message_id
        msgs = get_discord_messages(limit=1)
        if msgs:
            self.last_message_id = msgs[0]["id"]

        # Pre-load investor data
        self._refresh_investor_data()

        while True:
            try:
                now_ts = _time.time()
                now = datetime.now(KST)
                t = (now.hour, now.minute)

                # Weekend check
                if now.weekday() >= 5:
                    logger.debug("Weekend — sleeping 30min")
                    _time.sleep(1800)
                    continue

                # Market hours: 08:00 ~ 20:00 KST (NXT 거래 포함)
                if t < (8, 0) or t > (20, 0):
                    logger.debug(f"{now.strftime('%H:%M')} outside market hours")
                    _time.sleep(300)
                    continue

                # 1. Poll Discord messages (every iteration)
                self._process_discord()

                # 2. Signal check (every signal_interval)
                if now_ts - self.last_signal_time >= self.signal_interval:
                    self._send_signal()
                    self.last_signal_time = now_ts

                # 3. Comprehensive report (every report_interval)
                if now_ts - self.last_report_time >= self.report_interval:
                    self._send_report()
                    self.last_report_time = now_ts

            except KeyboardInterrupt:
                logger.info("Trading loop stopped")
                send_discord("🛑 Trading Bot 종료")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)

            _time.sleep(self.poll_interval)

    # ----- Signal -----

    def _send_signal(self):
        logger.info("Fetching signal...")
        data = fetch_quotes()
        if not data:
            logger.warning("No data fetched")
            return

        sig = run_bayesian(data)
        self.last_data = data
        self.last_signal = sig

        # Pattern analysis
        pattern_text = ""
        if self.pattern_matcher:
            try:
                conditions = {
                    k: data[k]["change_pct"]
                    for k in ["usd_krw", "sp500_fut", "nasdaq_fut", "wti_crude", "vix", "dxy"]
                    if k in data
                }
                # Map to pattern matcher column names
                mapped = {}
                rename = {"sp500_fut": "sp500", "nasdaq_fut": "nasdaq", "wti_crude": "wti"}
                for k, v in conditions.items():
                    mapped[rename.get(k, k)] = v

                patterns = self.pattern_matcher.find_patterns(mapped)
                if patterns:
                    pattern_text = self.pattern_matcher.format_patterns(patterns)
            except Exception as e:
                logger.debug(f"Pattern analysis failed: {e}")

        # Build and send signal embed
        embed = build_signal_embed(sig, data, pattern_text)
        if send_discord(embed=embed):
            logger.info(f"Signal sent: {sig['direction']} L={sig['long_prob']:.1%}")
        else:
            logger.warning("Signal send failed")

        # Check positions for sell alerts
        if sig["direction"] != "NEUTRAL":
            sell_alerts = self.positions.check_sell_signals(sig["direction"])
            for pos, reason in sell_alerts:
                strategy = generate_strategy(pos, sig["direction"], sig["confidence"])
                alert_embed = {
                    "title": f"⚠️ 매도 시그널 — {pos.etf_name}",
                    "description": f"**{reason}**\n\n{strategy}",
                    "color": 0xFF6600,
                    "footer": {"text": "포지션 매도 알림"},
                }
                send_discord(embed=alert_embed)

        # Track direction change
        if self.last_direction and self.last_direction != sig["direction"] and sig["direction"] != "NEUTRAL":
            send_discord(f"⚠️ **방향 전환**: {self.last_direction} → {sig['direction']}")
        self.last_direction = sig["direction"]

    # ----- Report -----

    def _send_report(self):
        logger.info("Building 30-min report...")
        data = self.last_data if self.last_data else fetch_quotes()
        sig = self.last_signal if self.last_signal else run_bayesian(data) if data else {}

        if not data or not sig:
            return

        # News
        news_text = ""
        try:
            from kospi_corr.collectors.news import NewsCollector
            nc = NewsCollector()
            ns = nc.collect_signal()
            urgency = ns.urgency_level
            news_lines = [f"Urgency: {urgency} | Sentiment: {ns.sentiment_score:.2f}"]
            hits = {k: v for k, v in ns.keyword_hits.items() if v > 0}
            if hits:
                news_lines.append(f"Keywords: {', '.join(f'{k}={v}' for k, v in hits.items())}")
            for art in ns.top_articles[:4]:
                kw = ", ".join(art.matched_keywords[:2])
                news_lines.append(f"• [{art.source}] {art.title} ({kw})")
            news_text = "\n".join(news_lines)
        except Exception as e:
            logger.debug(f"News failed: {e}")

        # Investor data
        self._refresh_investor_data()
        investor_text = ""
        if self.investor_data:
            try:
                from kospi_corr.data.providers.naver_scraper import interpret_investor_flow
                investor_text = interpret_investor_flow(self.investor_data)
            except Exception as e:
                logger.debug(f"Investor interpret failed: {e}")

        # Patterns
        pattern_text = ""
        if self.pattern_matcher:
            try:
                conditions = {
                    k: data[k]["change_pct"]
                    for k in ["usd_krw", "sp500_fut", "nasdaq_fut", "wti_crude", "vix", "dxy"]
                    if k in data
                }
                mapped = {}
                rename = {"sp500_fut": "sp500", "nasdaq_fut": "nasdaq", "wti_crude": "wti"}
                for k, v in conditions.items():
                    mapped[rename.get(k, k)] = v

                patterns = self.pattern_matcher.find_patterns(mapped)
                if patterns:
                    pattern_text = self.pattern_matcher.format_patterns(patterns[:3])
            except Exception as e:
                logger.debug(f"Pattern failed: {e}")

        # Considerations
        considerations = generate_considerations(
            sig, data, self.investor_data, self.last_data
        )

        # Position text
        position_text = self.positions.format_positions()

        # Build and send report
        embed = build_report_embed(
            sig, data, news_text, investor_text,
            pattern_text, position_text, considerations,
        )
        if send_discord(embed=embed):
            logger.info("30-min report sent")

    # ----- Discord processing -----

    def _process_discord(self):
        msgs = get_discord_messages(limit=10, after=self.last_message_id)
        if not msgs:
            return

        msgs.reverse()  # chronological order
        for msg in msgs:
            msg_id = msg["id"]
            author = msg.get("author", {})
            is_bot = author.get("bot", False)
            content = msg.get("content", "").strip()

            if is_bot:
                self.last_message_id = msg_id
                continue

            if not content:
                self.last_message_id = msg_id
                continue

            # Check for commands
            cmd = content.split()[0].lower()
            if cmd in ("!시그널", "!signal"):
                self._send_signal()
            elif cmd in ("!장전체크", "!premarket"):
                self._send_signal()  # same as signal in loop mode
            elif cmd in ("!뉴스", "!news"):
                self._handle_news_command()
            elif cmd in ("!포지션", "!position"):
                send_discord(self.positions.format_positions() or "보유 포지션 없음")
            elif cmd in ("!청산", "!clear"):
                self.positions.clear()
                send_discord("포지션 초기화 완료")
            elif cmd in ("!상태", "!status"):
                self._handle_status_command()
            elif cmd in ("!도움말", "!help"):
                send_discord(handle_help())
            else:
                # Check for buy/sell message
                parsed = parse_user_message(content)
                if parsed:
                    self._handle_trade_message(parsed, content)

            self.last_message_id = msg_id

    def _handle_trade_message(self, parsed: dict, original_text: str):
        sig = self.last_signal or {}
        sig_dir = sig.get("direction", "NEUTRAL")
        confidence = sig.get("confidence", 0.5)

        if parsed["action"] == "buy" and parsed["etf"]:
            pos = self.positions.add(
                etf_name=parsed["etf"],
                code=parsed["code"],
                direction=parsed["direction"],
                qty=parsed["quantity"] or 1,
                price=parsed["price"] or 0,
                current_signal=sig_dir,
            )
            strategy = generate_strategy(pos, sig_dir, confidence)
            embed = build_position_embed(pos, strategy)
            send_discord(embed=embed)
            logger.info(f"Position added: {pos.etf_name}")

        elif parsed["action"] == "sell" and parsed["etf"]:
            removed = self.positions.remove_by_name(parsed["etf"])
            if removed:
                send_discord(f"✅ {removed.etf_name} 포지션 해제")
            else:
                send_discord(f"해당 포지션을 찾을 수 없습니다")

        elif parsed["action"] == "sell_all":
            n = len(self.positions.positions)
            self.positions.clear()
            send_discord(f"✅ 전체 포지션 해제 ({n}건)")

    def _handle_news_command(self):
        try:
            from kospi_corr.collectors.news import NewsCollector
            nc = NewsCollector()
            ns = nc.collect_signal()
            lines = [f"**뉴스 모니터링** (Urgency: {ns.urgency_level})"]
            hits = {k: v for k, v in ns.keyword_hits.items() if v > 0}
            if hits:
                lines.append(f"Keywords: {', '.join(f'{k}={v}' for k, v in hits.items())}")
            for art in ns.top_articles[:7]:
                kw = ", ".join(art.matched_keywords[:3])
                lines.append(f"• [{art.source}] {art.title} ({kw})")
            send_discord("\n".join(lines))
        except Exception as e:
            send_discord(f"뉴스 체크 실패: {e}")

    def _handle_status_command(self):
        now = datetime.now(KST)
        sig = self.last_signal or {}
        lines = [
            f"**시스템 상태** ({now.strftime('%H:%M KST')})",
            f"시그널 간격: {self.signal_interval//60}분",
            f"리포트 간격: {self.report_interval//60}분",
            f"현재 방향: {sig.get('direction', 'N/A')}",
            f"포지션: {len(self.positions.positions)}건",
            f"패턴분석: {'ON' if self.pattern_matcher else 'OFF'}",
            f"루프: 실행 중",
        ]
        send_discord("\n".join(lines))

    def _refresh_investor_data(self):
        try:
            from kospi_corr.data.providers.naver_scraper import fetch_investor_trend
            data = fetch_investor_trend()
            if data:
                self.investor_data = data
        except Exception as e:
            logger.debug(f"Investor data refresh failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KOSPI Unified Trading Loop")
    parser.add_argument("--signal-interval", type=int, default=5,
                        help="Signal interval in minutes (default: 5)")
    parser.add_argument("--report-interval", type=int, default=30,
                        help="Report interval in minutes (default: 30)")
    parser.add_argument("--poll-interval", type=int, default=15,
                        help="Discord poll interval in seconds (default: 15)")
    parser.add_argument("--no-pattern", action="store_true",
                        help="Disable pattern analysis (faster startup)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    loop = TradingLoop(
        signal_interval=args.signal_interval,
        report_interval=args.report_interval,
        poll_interval=args.poll_interval,
        enable_patterns=not args.no_pattern,
    )
    loop.run()


if __name__ == "__main__":
    main()
