#!/usr/bin/env python3
"""Phase-aware KOSPI ETF trading loop — signals, reports, position tracking, Discord.

Replaces the previous monolithic loop with a clean pipeline architecture:
  - SignalPipeline handles ALL data fetching, Bayesian computation, and considerations
  - MarketPhase detection drives source selection and weight profiles
  - Trading loop focuses on scheduling, Discord I/O, and position management

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/trading_loop.py
    PYTHONPATH=src python scripts/trading_loop.py --signal-interval 3
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from kospi_corr.engine.signal_pipeline import SignalPipeline, SignalResult
from kospi_corr.engine.market_phase import MarketPhase, detect_phase, is_weekday
from kospi_corr.engine.predictive import PredictiveEngine, PredictiveResult
from kospi_corr.engine.decision import DecisionEngine, TradingDecision

logger = logging.getLogger("trading_loop")
KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1481273084916011008"
ADMIN_USER_ID = "1466390278771576894"
API_BASE = "https://discord.com/api/v10"

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

    is_buy = bool(re.search(r"매수|진입|샀|삼|롱진입|숏진입", text_lower))
    is_sell = bool(re.search(r"매도|팔|익절|손절|청산|탈출|매도완료|전량매도", text_lower))

    if not is_buy and not is_sell:
        return None

    action = "buy" if is_buy else "sell"

    etf_info = None
    for alias, info in ETF_ALIASES.items():
        if alias in text_lower:
            etf_info = info
            break

    if not etf_info and action == "sell":
        return {"action": "sell_all", "etf": None, "code": None,
                "direction": None, "quantity": 0, "price": 0}

    if not etf_info:
        return None

    price_match = re.search(r"(\d[\d,]*)원", text)
    price = float(price_match.group(1).replace(",", "")) if price_match else 0

    qty = 0
    qty_match = re.search(r"(\d[\d,]*)\s*주", text)
    if qty_match:
        qty = int(qty_match.group(1).replace(",", ""))
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
        lines.append("포지션: 숏 (KOSPI 하락 베팅)")
        if signal_dir == "SHORT":
            lines.append(f"현재 시그널: SHORT {confidence:.0%} → **보유 유리**")
            target = pos.price * 1.02
            stop = pos.price * 0.988
        else:
            lines.append("현재 시그널: LONG → **매도 고려!**")
            target = pos.price * 1.005
            stop = pos.price * 0.995
    else:
        lines.append("포지션: 롱 (상승 베팅)")
        if signal_dir == "LONG":
            lines.append(f"현재 시그널: LONG {confidence:.0%} → **보유 유리**")
            target = pos.price * 1.025
            stop = pos.price * 0.985
        else:
            lines.append("현재 시그널: SHORT → **매도 고려!**")
            target = pos.price * 1.005
            stop = pos.price * 0.995

    lines.append("\n**매매 전략:**")
    lines.append(f"🎯 목표가: {target:,.0f}원 ({(target / pos.price - 1) * 100:+.1f}%)")
    lines.append(f"🛑 손절가: {stop:,.0f}원 ({(stop / pos.price - 1) * 100:+.1f}%)")
    lines.append("\n**매도 조건:**")

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
# Discord helpers
# ---------------------------------------------------------------------------

_DISCORD_HEADERS = {
    "Authorization": f"Bot {BOT_TOKEN}",
    "Content-Type": "application/json",
}


def send_discord(content: str = "", embed: dict | None = None) -> bool:
    if not content and not embed:
        return False
    url = f"{API_BASE}/channels/{CHANNEL_ID}/messages"
    payload: dict = {}
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

_COLOR_MAP = {"LONG": 0x00FF00, "SHORT": 0xFF0000, "NEUTRAL": 0xFFFF00}
_EMOJI_MAP = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "🟡"}
_PHASE_LABEL = {
    "premarket": "장전", "opening": "장 시작",
    "intraday": "장중", "closing": "장 마감",
    "postmarket": "장후",
}


def build_signal_embed(result: SignalResult, positions: list[Position]) -> dict:
    """Build Discord embed from a SignalResult."""
    d = result.direction
    fields: list[dict] = []

    # Key signal contributors (top 4)
    if result.key_signals:
        top = result.key_signals[:4]
        contrib_text = "\n".join(
            f"{'↑' if s['contribution'] > 0 else '↓'} {s['name']}: "
            f"P(L)={s['probability']:.0%}, 기여={s['contribution']:+.3f}"
            for s in top
        )
        fields.append({"name": "주요 기여", "value": contrib_text, "inline": False})

    # Signal result
    fields.append({
        "name": f"{_EMOJI_MAP.get(d, '🟡')} LONG / SHORT",
        "value": (
            f"**{result.long_probability:.1%}** / {result.short_probability:.1%}\n"
            f"신뢰도: {result.confidence:.0%}"
        ),
        "inline": False,
    })

    # Considerations
    if result.considerations:
        cons_text = "\n".join(f"• {c}" for c in result.considerations[:5])
        fields.append({"name": "💡 시장 판단", "value": cons_text, "inline": False})

    # Gate status
    if not result.gate_status.get("should_trade", True):
        reasons = result.gate_status.get("reasons", [])
        fields.append({
            "name": "⚠️ 거래 제한",
            "value": "\n".join(reasons) if reasons else "거래 제한 활성",
            "inline": False,
        })

    # Trading interpretation
    if d == "SHORT":
        interp = "→ 인버스2X(252670) 매수 유리"
    elif d == "LONG":
        interp = "→ 레버리지 ETF 매수 유리"
    else:
        interp = "→ 관망 (시그널 약함)"
    fields.append({"name": "매매 해석", "value": interp, "inline": False})

    # Active positions summary
    if positions:
        pos_lines = []
        for p in positions:
            tag = "숏" if p.direction == "short" else "롱"
            opposing = p.is_opposing_signal(d)
            warn = " ⚠️ 반대시그널!" if opposing else ""
            pos_lines.append(f"• {p.etf_name} ({tag}){warn}")
        fields.append({"name": "📍 포지션", "value": "\n".join(pos_lines), "inline": False})

    phase_str = _PHASE_LABEL.get(result.phase, result.phase)
    return {
        "title": f"📊 {phase_str} | {d} | {result.timestamp.strftime('%H:%M')} KST",
        "color": _COLOR_MAP.get(d, 0xFFFF00),
        "fields": fields,
        "footer": {"text": f"Phase: {result.phase} | Regime: {result.regime}"},
    }


def build_report_embed(result: SignalResult, positions: list[Position]) -> dict:
    """Build comprehensive 30-min report embed from SignalResult."""
    d = result.direction
    fields: list[dict] = []

    # Key signal contributors
    if result.key_signals:
        top = result.key_signals[:6]
        contrib_text = "\n".join(
            f"{'↑' if s['contribution'] > 0 else '↓'} {s['name']}: "
            f"P(L)={s['probability']:.0%}, 기여={s['contribution']:+.3f}"
            for s in top
        )
        fields.append({"name": "📊 시장 지표", "value": contrib_text, "inline": False})

    # Signal summary
    fields.append({
        "name": f"시그널: {_EMOJI_MAP.get(d, '🟡')} {d}",
        "value": (
            f"LONG {result.long_probability:.1%} / SHORT {result.short_probability:.1%} "
            f"(신뢰도 {result.confidence:.0%})"
        ),
        "inline": False,
    })

    # Data freshness indicator
    if result.data_ages:
        age_items = []
        for key, age in result.data_ages.items():
            if age < 60:
                age_items.append(f"✅ {key}: {age:.0f}s")
            elif age < 300:
                age_items.append(f"⚡ {key}: {age / 60:.1f}m")
            else:
                age_items.append(f"⚠️ {key}: {age / 60:.0f}m (stale)")
        if age_items:
            fields.append({
                "name": "📡 데이터 신선도",
                "value": "\n".join(age_items),
                "inline": False,
            })

    # Considerations
    if result.considerations:
        cons_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(result.considerations[:5]))
        fields.append({"name": "💡 고려사항", "value": cons_text, "inline": False})

    # Gate status
    if not result.gate_status.get("should_trade", True):
        reasons = result.gate_status.get("reasons", [])
        fields.append({
            "name": "⚠️ 거래 제한",
            "value": "\n".join(reasons) if reasons else "거래 제한 활성",
            "inline": False,
        })

    # Trading interpretation
    if d == "SHORT":
        interp = "→ 인버스2X(252670) 매수 유리 (KOSPI 하락 예상)"
    elif d == "LONG":
        interp = "→ 레버리지 ETF 매수 유리 (KOSPI 상승 예상)"
    else:
        interp = "→ 관망 (시그널 약함)"
    fields.append({"name": "💡 매매 해석", "value": interp, "inline": False})

    # Positions
    if positions:
        pos_lines = []
        for p in positions:
            tag = "숏" if p.direction == "short" else "롱"
            pos_lines.append(
                f"• {p.etf_name} ({tag}) {p.quantity:,}주 @{p.price:,.0f}원 "
                f"[진입: {p.entry_time.strftime('%H:%M')}]"
            )
        fields.append({"name": "📍 포지션", "value": "\n".join(pos_lines), "inline": False})

    now = datetime.now(KST)
    return {
        "title": f"📰 종합 리포트 | {now.strftime('%H:%M')} KST",
        "color": _COLOR_MAP.get(d, 0xFFFF00),
        "fields": fields,
        "footer": {"text": f"Phase: {result.phase} | Regime: {result.regime} | 30분 자동"},
    }


def build_predictive_embed(pred: PredictiveResult) -> dict:
    """Build Discord embed for predictive/forward-looking analysis."""
    if not pred.scenarios:
        return {
            "title": "🔮 예측 분석",
            "description": "데이터 수집 중 (warmup 미완료)",
            "color": 0x808080,
        }

    top = pred.scenarios[0]
    color = _COLOR_MAP.get(top.direction, 0xFFFF00)
    emoji = _EMOJI_MAP.get(top.direction, "🟡")
    fields: list[dict] = []

    # Scenarios
    for i, sc in enumerate(pred.scenarios[:3]):
        sc_emoji = _EMOJI_MAP.get(sc.direction, "🟡")
        conds = "\n".join(f"  • {c}" for c in sc.conditions_kr)
        risks = ", ".join(sc.risk_factors_kr) if sc.risk_factors_kr else "없음"
        etf_str = f" → {sc.target_etf_name}" if sc.target_etf else ""
        fields.append({
            "name": f"{sc_emoji} {sc.label_kr} ({sc.probability:.0%}){etf_str}",
            "value": (
                f"**조건:**\n{conds}\n"
                f"**진입:** {sc.entry_trigger_kr}\n"
                f"**리스크:** {risks}"
            ),
            "inline": False,
        })

    # Leading signals
    if pred.leading_signals:
        lead_lines = []
        for ls in pred.leading_signals[:5]:
            imp_emoji = {"LONG": "🟢", "SHORT": "🔴", "CAUTION": "⚠️"}.get(
                ls.implication, "❓"
            )
            lead_lines.append(
                f"{imp_emoji} {ls.description_kr} (강도 {ls.strength:.0%})"
            )
        fields.append({
            "name": "📡 선행 신호",
            "value": "\n".join(lead_lines),
            "inline": False,
        })

    # Entry triggers
    if pred.triggers:
        trig_lines = []
        for tr in pred.triggers[:5]:
            tr_emoji = "🟢" if tr.direction == "LONG" else "🔴"
            dist = f" (거리 {tr.distance_pct:.1f}%)" if tr.distance_pct > 0 else " (진행 중)"
            trig_lines.append(f"{tr_emoji} {tr.condition_kr}{dist}")
        fields.append({
            "name": "🎯 진입 트리거",
            "value": "\n".join(trig_lines),
            "inline": False,
        })

    # Momentum summary
    key_momenta = ["외국인 순매수", "프로그램 순매수", "VKOSPI"]
    mom_lines = []
    for name in key_momenta:
        for k, m in pred.momentum.items():
            if m.name == name:
                state = ""
                if m.is_accelerating:
                    state = " 가속↗"
                elif m.is_decelerating:
                    state = " 감속↘"
                if m.inflection_detected:
                    state += " 전환!"
                mom_lines.append(
                    f"• {m.name}: {m.velocity:+,.0f}/시간{state} "
                    f"({m.trend_duration_min:.0f}분 지속)"
                )
                break
    if mom_lines:
        fields.append({
            "name": "📈 모멘텀",
            "value": "\n".join(mom_lines),
            "inline": False,
        })

    warmup_str = "" if pred.warmup_complete else " (워밍업 중)"
    now = datetime.now(KST)
    return {
        "title": f"🔮 예측 분석{warmup_str} | {now.strftime('%H:%M')} KST",
        "description": (
            f"**최고확률: {emoji} {top.label_kr} ({top.probability:.0%})**\n"
            f"데이터 깊이: {pred.buffer_depth}건"
        ),
        "color": color,
        "fields": fields,
        "footer": {"text": "선행 시그널 기반 예측 — 참고용 (투자 판단은 본인 책임)"},
    }


def build_decision_embed(decision: TradingDecision) -> dict:
    """Build Discord embed for direct trading decision."""
    dir_color = {
        "SHORT": 0xFF0000, "LONG": 0x00FF00, "NEUTRAL": 0x808080,
    }
    dir_emoji = {
        "SHORT": "🔴", "LONG": "🟢", "NEUTRAL": "⚪",
    }
    urgency_emoji = {
        "즉시": "🚨", "조건부": "⏳", "대기": "👀", "관망": "💤",
    }

    d = decision.direction
    emoji = dir_emoji.get(d, "⚪")
    u_emoji = urgency_emoji.get(decision.urgency, "")
    fields: list[dict] = []

    # Core action (biggest, most visible)
    fields.append({
        "name": f"{u_emoji} 판단: {decision.action_kr}",
        "value": (
            f"**{decision.reason_kr}**\n"
            f"신뢰도: {decision.confidence:.0%} | 긴급도: {decision.urgency}"
        ),
        "inline": False,
    })

    # Entry/Exit/Stop
    if decision.action != "WAIT":
        fields.append({
            "name": "📍 진입/목표/손절",
            "value": (
                f"진입: {decision.entry_hint_kr}\n"
                f"목표: {decision.target_kr}\n"
                f"손절: {decision.stop_loss_kr}\n"
                f"규모: {decision.position_hint_kr}"
            ),
            "inline": False,
        })

    # Quant scenarios
    if decision.scenarios:
        sc_lines = []
        for sc in decision.scenarios[:4]:
            sc_emoji = "🟢" if sc.expected_move_pct > 0 else (
                "🔴" if sc.expected_move_pct < 0 else "⚪"
            )
            sc_lines.append(
                f"{sc_emoji} **{sc.name_kr}** ({sc.probability:.0%})\n"
                f"  {sc.trigger_kr}\n"
                f"  → {sc.action_kr} ({sc.time_horizon_kr})"
            )
        fields.append({
            "name": "📊 퀀트 시나리오",
            "value": "\n".join(sc_lines),
            "inline": False,
        })

    # Market context
    if decision.market_context_kr:
        ctx = "\n".join(f"• {c}" for c in decision.market_context_kr[:5])
        fields.append({
            "name": "📈 시장 상황",
            "value": ctx,
            "inline": False,
        })

    # Risks
    if decision.risk_factors_kr:
        risks = "\n".join(f"⚠️ {r}" for r in decision.risk_factors_kr[:4])
        fields.append({
            "name": "🛡️ 리스크",
            "value": risks,
            "inline": False,
        })

    now = datetime.now(KST)
    return {
        "title": (
            f"{emoji} 매매 판단 | {decision.urgency} | "
            f"{now.strftime('%H:%M')} KST"
        ),
        "description": (
            f"**{decision.etf_name}** ({decision.etf_code})\n"
            f"{decision.urgency_reason_kr}"
        ) if decision.action != "WAIT" else decision.urgency_reason_kr,
        "color": dir_color.get(d, 0x808080),
        "fields": fields,
        "footer": {
            "text": (
                "단타 매매 판단 — 리스크 감수형 | "
                "참고용 (투자 판단은 본인 책임)"
            ),
        },
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
# Discord command handler text
# ---------------------------------------------------------------------------

def handle_help() -> str:
    return (
        "**사용 가능한 명령어:**\n"
        "`!판단` — **직접 매매 판단** (뭘 사야 하나?)\n"
        "`!시그널` — 현재 시그널 + 예측 + 판단 (풀 분석)\n"
        "`!예측` — 선행 시나리오 분석 (LONG/SHORT 조건+트리거)\n"
        "`!장전체크` — 장전 시그널 (장전 소스 포함)\n"
        "`!뉴스` — 뉴스 키워드 모니터링\n"
        "`!포지션` — 보유 포지션 확인\n"
        "`!청산` — 모든 포지션 초기화\n"
        "`!상태` — 시스템 상태 (현재 Phase 포함)\n"
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
    """Phase-aware trading loop: pipeline signals + Discord + positions."""

    def __init__(
        self,
        signal_interval: int = 5,
        report_interval: int = 30,
        poll_interval: int = 15,
        enable_news: bool = True,
        enable_gate: bool = True,
    ):
        self.signal_interval = signal_interval * 60  # seconds
        self.report_interval = report_interval * 60
        self.poll_interval = poll_interval

        self.pipeline = SignalPipeline(
            enable_news=enable_news,
            enable_gate=enable_gate,
        )
        self.predictive = PredictiveEngine()
        self.decision_engine = DecisionEngine()
        self.positions = PositionTracker()

        self.last_signal_time: float = 0
        self.last_report_time: float = 0
        self.last_direction: str | None = None
        self.last_result: SignalResult | None = None
        self.last_predictive: PredictiveResult | None = None
        self.last_decision: TradingDecision | None = None
        self.last_message_id: str | None = None

    def run(self):
        """Main event loop."""
        logger.info(
            f"Trading loop started: signal={self.signal_interval // 60}min, "
            f"report={self.report_interval // 60}min, poll={self.poll_interval}s"
        )
        send_discord(
            f"🤖 Trading Bot 시작 (Phase-aware)\n"
            f"시그널: {self.signal_interval // 60}분 | 리포트: {self.report_interval // 60}분\n"
            f"`!도움말`로 명령어 확인"
        )

        # Initialize last_message_id
        try:
            msgs = get_discord_messages(limit=1)
            if msgs:
                self.last_message_id = msgs[0]["id"]
        except Exception:
            pass

        while True:
            try:
                now_ts = _time.time()
                now = datetime.now(KST)

                # Weekend — sleep 30 min
                if not is_weekday(now):
                    logger.debug("Weekend — sleeping 30min")
                    _time.sleep(1800)
                    continue

                phase = detect_phase(now)

                # Postmarket — sleep 5 min, only poll Discord
                if phase == MarketPhase.POSTMARKET:
                    self._process_discord()
                    _time.sleep(300)
                    continue

                # Poll Discord commands every iteration
                self._process_discord()

                # Adaptive signal interval:
                #   OPENING (09:00~09:10): 60s (gap absorption critical)
                #   CLOSING (15:20~15:30): 90s (program trading peak)
                #   INTRADAY: user-configured (default 5min)
                effective_interval = self.signal_interval
                if phase == MarketPhase.OPENING:
                    effective_interval = 60
                elif phase == MarketPhase.CLOSING:
                    effective_interval = 90

                if now_ts - self.last_signal_time >= effective_interval:
                    self._send_signal()
                    self.last_signal_time = now_ts

                # Comprehensive report (every report_interval, only market phases)
                if phase in (MarketPhase.OPENING, MarketPhase.INTRADAY, MarketPhase.CLOSING):
                    if now_ts - self.last_report_time >= self.report_interval:
                        self._send_report()
                        self.last_report_time = now_ts

            except KeyboardInterrupt:
                logger.info("Trading loop stopped by user")
                send_discord("🛑 Trading Bot 종료")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)

            _time.sleep(self.poll_interval)

    # ----- Signal -----

    def _send_signal(self):
        """Generate signal via pipeline and send to Discord."""
        logger.info("Generating signal...")
        try:
            result = self.pipeline.generate()
        except Exception as e:
            logger.error(f"Pipeline generate failed: {e}", exc_info=True)
            return

        self.last_result = result
        d = result.direction

        # Update predictive engine with new data
        domestic = getattr(result, "domestic_snapshot", None)
        try:
            pred = self.predictive.update(domestic, result)
            self.last_predictive = pred
        except Exception as e:
            logger.warning(f"Predictive update failed: {e}")
            pred = None

        # Build and send signal embed
        embed = build_signal_embed(result, self.positions.positions)
        if send_discord(embed=embed):
            logger.info(f"Signal sent: {d} L={result.long_probability:.1%} phase={result.phase}")
        else:
            logger.warning("Signal send failed")

        # Send predictive embed if warmup is complete
        if pred and pred.warmup_complete and pred.scenarios:
            pred_embed = build_predictive_embed(pred)
            send_discord(embed=pred_embed)

        # Generate and send decision (the core output)
        try:
            decision = self.decision_engine.decide(result, pred)
            self.last_decision = decision
            decision_embed = build_decision_embed(decision)
            send_discord(embed=decision_embed)
            logger.info(
                "Decision: %s %s (%s, %.0f%%)",
                decision.action, decision.etf_name,
                decision.urgency, decision.confidence * 100,
            )
        except Exception as e:
            logger.warning(f"Decision engine failed: {e}")

        # Check positions for sell alerts
        if d != "NEUTRAL":
            sell_alerts = self.positions.check_sell_signals(d)
            for pos, reason in sell_alerts:
                strategy = generate_strategy(pos, d, result.confidence)
                alert_embed = {
                    "title": f"⚠️ 매도 시그널 — {pos.etf_name}",
                    "description": f"**{reason}**\n\n{strategy}",
                    "color": 0xFF6600,
                    "footer": {"text": "포지션 매도 알림"},
                }
                send_discord(embed=alert_embed)

        # Track direction change
        if self.last_direction and self.last_direction != d and d != "NEUTRAL":
            send_discord(f"⚠️ **방향 전환**: {self.last_direction} → {d}")
        self.last_direction = d

    # ----- Report -----

    def _send_report(self):
        """Send comprehensive report, reusing recent signal if fresh."""
        logger.info("Building comprehensive report...")
        # Reuse last_result if generated within 5 minutes
        if self.last_result and (_time.time() - self.last_signal_time) < 300:
            result = self.last_result
        else:
            try:
                result = self.pipeline.generate()
            except Exception as e:
                logger.error(f"Pipeline generate for report failed: {e}", exc_info=True)
                return
            self.last_result = result

        embed = build_report_embed(result, self.positions.positions)
        if send_discord(embed=embed):
            logger.info(f"Report sent: {result.direction} phase={result.phase}")

    # ----- Discord processing -----

    def _process_discord(self):
        """Poll and handle Discord messages."""
        try:
            msgs = get_discord_messages(limit=10, after=self.last_message_id)
        except Exception as e:
            logger.debug(f"Discord poll failed: {e}")
            return

        if not msgs:
            return

        msgs.reverse()  # chronological order
        for msg in msgs:
            msg_id = msg["id"]
            author = msg.get("author", {})
            is_bot = author.get("bot", False)
            content = msg.get("content", "").strip()

            if is_bot or not content:
                self.last_message_id = msg_id
                continue

            self._handle_command(content)
            self.last_message_id = msg_id

    def _handle_command(self, content: str):
        """Route a Discord message to the appropriate handler."""
        cmd = content.split()[0].lower()

        if cmd in ("!시그널", "!signal"):
            self._send_signal()

        elif cmd in ("!판단", "!decide", "!매매"):
            self._handle_decision_command()

        elif cmd in ("!예측", "!predict"):
            self._handle_predict_command()

        elif cmd in ("!장전체크", "!premarket"):
            self._handle_premarket_command()

        elif cmd in ("!뉴스", "!news"):
            self._handle_news_command()

        elif cmd in ("!포지션", "!position"):
            send_discord(self.positions.format_positions())

        elif cmd in ("!청산", "!clear"):
            self.positions.clear()
            send_discord("포지션 초기화 완료")

        elif cmd in ("!상태", "!status"):
            self._handle_status_command()

        elif cmd in ("!도움말", "!help"):
            send_discord(handle_help())

        else:
            parsed = parse_user_message(content)
            if parsed:
                self._handle_trade_message(parsed)

    def _handle_premarket_command(self):
        """Force pre-market phase signal generation."""
        logger.info("Premarket check requested")
        try:
            result = self.pipeline.generate(phase_override=MarketPhase.PREMARKET)
            self.last_result = result
            embed = build_signal_embed(result, self.positions.positions)
            send_discord(embed=embed)
        except Exception as e:
            logger.error(f"Premarket signal failed: {e}", exc_info=True)
            send_discord(f"장전 체크 실패: {e}")

    def _handle_trade_message(self, parsed: dict):
        """Handle buy/sell messages from user."""
        result = self.last_result
        sig_dir = result.direction if result else "NEUTRAL"
        confidence = result.confidence if result else 0.5

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
                send_discord("해당 포지션을 찾을 수 없습니다")

        elif parsed["action"] == "sell_all":
            n = len(self.positions.positions)
            self.positions.clear()
            send_discord(f"✅ 전체 포지션 해제 ({n}건)")

    def _handle_decision_command(self):
        """Generate and send direct trading decision."""
        # Reuse recent result if fresh (< 2 min), otherwise regenerate
        if self.last_decision and (_time.time() - self.last_signal_time) < 120:
            embed = build_decision_embed(self.last_decision)
            send_discord(embed=embed)
        else:
            self._send_signal()  # generates decision as side effect
            if not self.last_decision:
                send_discord("판단 데이터 수집 중 — 시그널 몇 회 수집 후 재시도하세요.")

    def _handle_predict_command(self):
        """Send predictive analysis to Discord."""
        # Reuse if fresh (< 2 min), otherwise regenerate
        if (self.last_predictive and self.last_predictive.warmup_complete
                and (_time.time() - self.last_signal_time) < 120):
            embed = build_predictive_embed(self.last_predictive)
            send_discord(embed=embed)
        else:
            self._send_signal()
            if self.last_predictive:
                embed = build_predictive_embed(self.last_predictive)
                send_discord(embed=embed)
            else:
                send_discord("예측 데이터 아직 부족합니다. 몇 번 더 시그널을 수집 후 시도하세요.")

    def _handle_news_command(self):
        """Fetch and display news."""
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
        """Show system status including current phase."""
        now = datetime.now(KST)
        phase = detect_phase(now)
        result = self.last_result
        lines = [
            f"**시스템 상태** ({now.strftime('%H:%M KST')})",
            f"현재 Phase: {_PHASE_LABEL.get(phase.value, phase.value)}",
            f"시그널 간격: {self.signal_interval // 60}분",
            f"리포트 간격: {self.report_interval // 60}분",
            f"현재 방향: {result.direction if result else 'N/A'}",
            f"신뢰도: {result.confidence:.0%}" if result else "신뢰도: N/A",
            f"Regime: {result.regime}" if result else "Regime: N/A",
            f"포지션: {len(self.positions.positions)}건",
            f"뉴스: {'ON' if self.pipeline._enable_news else 'OFF'}",
            f"Gate: {'ON' if self.pipeline._enable_gate else 'OFF'}",
            f"루프: 실행 중",
        ]
        send_discord("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KOSPI Phase-Aware Trading Loop")
    parser.add_argument("--signal-interval", type=int, default=5,
                        help="Signal interval in minutes (default: 5)")
    parser.add_argument("--report-interval", type=int, default=30,
                        help="Report interval in minutes (default: 30)")
    parser.add_argument("--poll-interval", type=int, default=15,
                        help="Discord poll interval in seconds (default: 15)")
    parser.add_argument("--no-news", action="store_true",
                        help="Disable news collection")
    parser.add_argument("--no-gate", action="store_true",
                        help="Disable regime gate")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M",
    )

    loop = TradingLoop(
        signal_interval=args.signal_interval,
        report_interval=args.report_interval,
        poll_interval=args.poll_interval,
        enable_news=not args.no_news,
        enable_gate=not args.no_gate,
    )
    loop.run()


if __name__ == "__main__":
    main()
