"""Market phase detection and phase-specific signal configuration.

Defines the core distinction between pre-market and intraday modes:
  - PREMARKET (06:00-08:59 KST): Yesterday's close + news → gap direction
  - INTRADAY  (09:00-15:30 KST): ONLY real-time data, no stale overnight
  - POSTMARKET (15:30+ KST): Analysis mode, no live signals

Key principle: Once KOSPI opens at 09:00, yesterday's S&P500/NASDAQ close
is ALREADY PRICED IN. Only live-updating sources should influence signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Market Phase
# ---------------------------------------------------------------------------

class MarketPhase(StrEnum):
    PREMARKET = "premarket"       # 06:00-08:59 KST
    OPENING = "opening"           # 09:00-09:10 KST (gap absorption)
    INTRADAY = "intraday"         # 09:10-15:20 KST
    CLOSING = "closing"           # 15:20-15:30 KST
    POSTMARKET = "postmarket"     # 15:30-05:59 KST


def detect_phase(now: datetime | None = None) -> MarketPhase:
    """Detect current market phase from KST time."""
    if now is None:
        now = datetime.now(KST)
    h, m = now.hour, now.minute
    t = h * 60 + m

    if t < 360:       # 00:00-05:59
        return MarketPhase.POSTMARKET
    if t < 540:        # 06:00-08:59
        return MarketPhase.PREMARKET
    if t < 550:        # 09:00-09:09
        return MarketPhase.OPENING
    if t < 920:        # 09:10-15:19
        return MarketPhase.INTRADAY
    if t <= 930:       # 15:20-15:30
        return MarketPhase.CLOSING
    return MarketPhase.POSTMARKET


def is_weekday(now: datetime | None = None) -> bool:
    if now is None:
        now = datetime.now(KST)
    return now.weekday() < 5


# ---------------------------------------------------------------------------
# Signal source definitions — phase-aware
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalSourceConfig:
    """Configuration for one real-time signal source."""
    key: str
    yf_symbol: str
    label: str
    direction: int           # +1 = bullish_up, -1 = bullish_down
    base_weight: float
    category: str            # "direction_driver", "sentiment", "domestic_flow", "domestic_index"
    is_24h: bool = True      # True if source trades during KST daytime
    invert_quote: bool = False


# Sources that are LIVE 24 hours (or during KST daytime)
REALTIME_24H_SOURCES: dict[str, SignalSourceConfig] = {
    "usd_krw": SignalSourceConfig(
        key="usd_krw", yf_symbol="KRW=X", label="USD/KRW",
        direction=-1, base_weight=3.0, category="direction_driver",
        is_24h=True, invert_quote=True,
    ),
    "sp500_fut": SignalSourceConfig(
        key="sp500_fut", yf_symbol="ES=F", label="S&P500 선물",
        direction=+1, base_weight=1.5, category="direction_driver",
        is_24h=True,
    ),
    "nasdaq_fut": SignalSourceConfig(
        key="nasdaq_fut", yf_symbol="NQ=F", label="NASDAQ 선물",
        direction=+1, base_weight=1.2, category="direction_driver",
        is_24h=True,
    ),
    "wti_crude": SignalSourceConfig(
        key="wti_crude", yf_symbol="CL=F", label="WTI",
        direction=+1, base_weight=0.8, category="direction_driver",
        is_24h=True,
    ),
    "dxy": SignalSourceConfig(
        key="dxy", yf_symbol="DX-Y.NYB", label="DXY",
        direction=-1, base_weight=0.8, category="direction_driver",
        is_24h=True,
    ),
}

# Sources ONLY valid pre-market (stale during KOSPI hours)
PREMARKET_ONLY_SOURCES: dict[str, SignalSourceConfig] = {
    "sp500": SignalSourceConfig(
        key="sp500", yf_symbol="^GSPC", label="S&P500",
        direction=+1, base_weight=2.0, category="direction_driver",
        is_24h=False,
    ),
    "nasdaq": SignalSourceConfig(
        key="nasdaq", yf_symbol="^IXIC", label="NASDAQ",
        direction=+1, base_weight=1.5, category="direction_driver",
        is_24h=False,
    ),
    "vix": SignalSourceConfig(
        key="vix", yf_symbol="^VIX", label="VIX",
        direction=-1, base_weight=1.0, category="sentiment",
        is_24h=False,
    ),
}

# Domestic sources (scraped, only valid during KRX hours)
DOMESTIC_SOURCES: dict[str, SignalSourceConfig] = {
    "kospi200": SignalSourceConfig(
        key="kospi200", yf_symbol="^KS200", label="KOSPI200",
        direction=+1, base_weight=2.0, category="domestic_index",
        is_24h=False,
    ),
    "vkospi": SignalSourceConfig(
        key="vkospi", yf_symbol="", label="VKOSPI",
        direction=-1, base_weight=1.5, category="sentiment",
        is_24h=False,
    ),
    "foreign_flow": SignalSourceConfig(
        key="foreign_flow", yf_symbol="", label="외국인 순매수",
        direction=+1, base_weight=1.8, category="domestic_flow",
        is_24h=False,
    ),
    "program_trading": SignalSourceConfig(
        key="program_trading", yf_symbol="", label="프로그램 매매",
        direction=+1, base_weight=1.5, category="domestic_flow",
        is_24h=False,
    ),
}


def get_phase_sources(phase: MarketPhase) -> dict[str, SignalSourceConfig]:
    """Return signal sources appropriate for the given market phase."""
    if phase == MarketPhase.PREMARKET:
        # Pre-market: overnight data (S&P, NASDAQ close) + 24h sources
        sources = {}
        sources.update(REALTIME_24H_SOURCES)
        sources.update(PREMARKET_ONLY_SOURCES)
        return sources

    if phase in (MarketPhase.OPENING, MarketPhase.INTRADAY, MarketPhase.CLOSING):
        # Intraday: ONLY real-time 24h sources + domestic live data
        # NO stale overnight data (S&P close, NASDAQ close, VIX)
        sources = {}
        sources.update(REALTIME_24H_SOURCES)
        sources.update(DOMESTIC_SOURCES)
        return sources

    # POSTMARKET: minimal
    return dict(REALTIME_24H_SOURCES)


# ---------------------------------------------------------------------------
# Time-adaptive weight multipliers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseWeightProfile:
    """Weight multipliers that change throughout the trading day."""
    usd_krw: float = 1.0
    sp500_fut: float = 1.0
    nasdaq_fut: float = 1.0
    wti_crude: float = 1.0
    dxy: float = 1.0
    kospi200: float = 1.0
    vkospi: float = 1.0
    foreign_flow: float = 1.0
    program_trading: float = 1.0
    # Pre-market only
    sp500: float = 1.0
    nasdaq: float = 1.0
    vix: float = 1.0


# Weight profiles by phase — these multiply the base_weight
PHASE_WEIGHT_PROFILES: dict[MarketPhase, PhaseWeightProfile] = {
    MarketPhase.PREMARKET: PhaseWeightProfile(
        usd_krw=1.0,
        sp500_fut=1.2,       # Overnight futures important
        nasdaq_fut=1.1,
        wti_crude=0.8,
        dxy=0.8,
        sp500=1.3,           # Yesterday's close sets the gap
        nasdaq=1.2,
        vix=1.0,
    ),
    MarketPhase.OPENING: PhaseWeightProfile(
        usd_krw=1.2,         # FX dominates gap
        sp500_fut=1.0,       # Still matters during gap
        nasdaq_fut=0.9,
        wti_crude=0.5,
        dxy=0.6,
        kospi200=1.5,        # Gap movement itself
        vkospi=1.0,
        foreign_flow=1.3,    # Opening auction flow
        program_trading=0.8, # Program trading starts slow
    ),
    MarketPhase.INTRADAY: PhaseWeightProfile(
        usd_krw=1.3,         # FX is king during session
        sp500_fut=0.6,       # Reduced — KOSPI has own momentum now
        nasdaq_fut=0.5,
        wti_crude=0.4,
        dxy=0.5,
        kospi200=1.5,        # Direct market signal
        vkospi=1.2,          # Volatility regime
        foreign_flow=1.5,    # Institutional flow is key
        program_trading=1.3, # Program trading meaningful intraday
    ),
    MarketPhase.CLOSING: PhaseWeightProfile(
        usd_krw=1.0,
        sp500_fut=0.8,       # Pre-US anticipation
        nasdaq_fut=0.7,
        wti_crude=0.4,
        dxy=0.5,
        kospi200=1.2,
        vkospi=1.0,
        foreign_flow=1.8,    # Institutional flow dominant at close
        program_trading=1.8, # Program trading heaviest at close
    ),
    MarketPhase.POSTMARKET: PhaseWeightProfile(
        usd_krw=0.8,
        sp500_fut=1.0,
        nasdaq_fut=0.8,
        wti_crude=0.5,
        dxy=0.6,
    ),
}


def get_phase_weight(phase: MarketPhase, source_key: str) -> float:
    """Get the time-adaptive weight multiplier for a source in a phase."""
    profile = PHASE_WEIGHT_PROFILES.get(phase, PHASE_WEIGHT_PROFILES[MarketPhase.INTRADAY])
    return getattr(profile, source_key, 1.0)


# ---------------------------------------------------------------------------
# Phase-dependent tau (time-decay half-life)
# ---------------------------------------------------------------------------

PHASE_TAU: dict[MarketPhase, float] = {
    MarketPhase.PREMARKET: 14400.0,   # 4 hours — overnight data is stable
    MarketPhase.OPENING: 900.0,       # 15 min — rapid gap absorption
    MarketPhase.INTRADAY: 1800.0,     # 30 min — fast decay for stale signals
    MarketPhase.CLOSING: 600.0,       # 10 min — very fast near close
    MarketPhase.POSTMARKET: 7200.0,   # 2 hours
}


def get_phase_tau(phase: MarketPhase) -> float:
    """Get the time-decay tau_seconds for a given phase."""
    return PHASE_TAU.get(phase, 1800.0)
