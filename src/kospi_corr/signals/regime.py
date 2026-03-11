"""Regime gate — special market condition detector.

Checks for conditions where normal trading signals should be overridden
or disabled: options expiry (만기일), circuit breakers, sidecar triggers,
elevated VKOSPI, and geopolitical event spikes from news monitoring.
"""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from kospi_corr.collectors.news import NewsCollector, NewsSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RegimeType(StrEnum):
    """Market regime classifications."""
    NORMAL = "NORMAL"
    EXPIRY_DAY = "EXPIRY_DAY"           # 만기일 (3rd Thursday)
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"  # CB triggered
    SIDECAR = "SIDECAR"                 # Sidecar triggered
    HIGH_VOLATILITY = "HIGH_VOLATILITY" # VKOSPI elevated
    EVENT_DRIVEN = "EVENT_DRIVEN"       # Major geopolitical news spike
    COMBINED = "COMBINED"               # Multiple special conditions


class RiskLevel(StrEnum):
    """Trading risk levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class RegimeResult:
    """Output of the regime gate check."""
    should_trade: bool
    regime_type: RegimeType
    risk_level: RiskLevel
    reasons: list[str] = field(default_factory=list)
    vkospi_level: float | None = None
    news_signal: NewsSignal | None = None
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict."""
        result = {
            "should_trade": self.should_trade,
            "regime_type": self.regime_type.value,
            "risk_level": self.risk_level.value,
            "reasons": self.reasons,
            "vkospi_level": self.vkospi_level,
            "checked_at": self.checked_at.isoformat(),
        }
        if self.news_signal is not None:
            result["news_sentiment"] = self.news_signal.sentiment_score
            result["news_urgency"] = self.news_signal.urgency_level
        return result


# ---------------------------------------------------------------------------
# VKOSPI thresholds
# ---------------------------------------------------------------------------

# Based on historical VKOSPI ranges:
#   < 15: very calm (low vol)
#   15-20: normal
#   20-25: elevated
#   25-30: high vol — signals less reliable
#   > 30: extreme — often crisis
_VKOSPI_ELEVATED = 25.0
_VKOSPI_HIGH = 30.0
_VKOSPI_EXTREME = 40.0

# News sentiment thresholds for EVENT_DRIVEN regime
_NEWS_EVENT_THRESHOLD = 0.50     # sentiment ≥ 0.50 → flag event
_NEWS_CRITICAL_THRESHOLD = 0.75  # sentiment ≥ 0.75 → halt trading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_third_thursday(year: int, month: int) -> date:
    """Return the 3rd Thursday of the given month.

    Korean derivatives expiry (만기일) falls on the 2nd Thursday of each
    month for stock options and the 3rd Thursday for KOSPI200 futures/options
    quarterly expiry.  This function returns the 3rd Thursday, which is the
    major 만기일 for quarterly months (Mar, Jun, Sep, Dec).
    """
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    thursdays = [
        d
        for d in cal.itermonthdays2(year, month)
        if d[0] != 0 and d[1] == calendar.THURSDAY
    ]
    # thursdays is list of (day, weekday) — 3rd Thursday is index 2
    if len(thursdays) >= 3:
        return date(year, month, thursdays[2][0])
    # Fallback: shouldn't happen but return last Thursday
    return date(year, month, thursdays[-1][0])


def _is_expiry_day(check_date: date) -> bool:
    """Check if the given date is a KOSPI200 derivatives expiry day (만기일).

    Monthly expiry is the 2nd Thursday; quarterly expiry (major 만기일)
    is the 2nd Thursday of Mar/Jun/Sep/Dec.  For safety we flag all
    2nd Thursdays.
    """
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    thursdays = [
        d[0]
        for d in cal.itermonthdays2(check_date.year, check_date.month)
        if d[0] != 0 and d[1] == calendar.THURSDAY
    ]
    # 2nd Thursday (standard KRX monthly options expiry)
    if len(thursdays) >= 2 and check_date.day == thursdays[1]:
        return True
    # 3rd Thursday — quarterly futures expiry (큰 만기일)
    if len(thursdays) >= 3 and check_date.day == thursdays[2]:
        quarterly_months = {3, 6, 9, 12}
        if check_date.month in quarterly_months:
            return True
    return False


def _assess_vkospi(level: float | None) -> tuple[bool, RiskLevel, str | None]:
    """Assess VKOSPI level.

    Returns (is_elevated, risk_level, reason_or_none).
    """
    if level is None:
        return False, RiskLevel.MEDIUM, None

    if level >= _VKOSPI_EXTREME:
        return True, RiskLevel.EXTREME, (
            f"VKOSPI {level:.1f} ≥ {_VKOSPI_EXTREME} (extreme volatility — "
            f"crisis level, all signals unreliable)"
        )
    elif level >= _VKOSPI_HIGH:
        return True, RiskLevel.HIGH, (
            f"VKOSPI {level:.1f} ≥ {_VKOSPI_HIGH} (high volatility — "
            f"signal reliability degraded)"
        )
    elif level >= _VKOSPI_ELEVATED:
        return True, RiskLevel.MEDIUM, (
            f"VKOSPI {level:.1f} ≥ {_VKOSPI_ELEVATED} (elevated volatility)"
        )
    else:
        return False, RiskLevel.LOW, None


# ---------------------------------------------------------------------------
# RegimeGate
# ---------------------------------------------------------------------------

class RegimeGate:
    """Checks for special market conditions that override trading signals.

    Detects:
    - **만기일 (Expiry day)**: 2nd Thursday monthly options, 3rd Thursday
      quarterly futures in Mar/Jun/Sep/Dec.
    - **Circuit breaker / Sidecar**: When explicitly flagged via parameters
      (real-time detection requires KRX WebSocket).
    - **VKOSPI regime**: Elevated, high, or extreme volatility levels.
    - **Event-driven regime**: Major geopolitical keyword spike from
      :class:`NewsCollector`.

    Parameters
    ----------
    news_collector : NewsCollector | None
        If provided, used to check for event-driven regime.
        If None, a default NewsCollector is created.
    vkospi_extreme_halt : bool
        If True, set ``should_trade = False`` when VKOSPI ≥ 40.
    expiry_day_halt : bool
        If True, set ``should_trade = False`` on 만기일.
        Defaults to False (expiry days are flagged but trading allowed).
    """

    def __init__(
        self,
        news_collector: NewsCollector | None = None,
        vkospi_extreme_halt: bool = True,
        expiry_day_halt: bool = False,
    ) -> None:
        self._news_collector = news_collector
        self._vkospi_extreme_halt = vkospi_extreme_halt
        self._expiry_day_halt = expiry_day_halt

    def check(
        self,
        check_date: date | None = None,
        vkospi: float | None = None,
        circuit_breaker_active: bool = False,
        sidecar_active: bool = False,
        skip_news: bool = False,
    ) -> RegimeResult:
        """Run all regime checks and return a unified result.

        Parameters
        ----------
        check_date : date | None
            Date to check for expiry; defaults to today.
        vkospi : float | None
            Current VKOSPI level.  If None, VKOSPI check is skipped.
        circuit_breaker_active : bool
            True if a circuit breaker is currently triggered.
        sidecar_active : bool
            True if a sidecar (program trading halt) is active.
        skip_news : bool
            If True, skip RSS news collection (useful for backtesting
            or when news data is unavailable).

        Returns
        -------
        RegimeResult
            Unified regime assessment.
        """
        if check_date is None:
            check_date = date.today()

        should_trade = True
        detected_regimes: list[RegimeType] = []
        reasons: list[str] = []
        risk_level = RiskLevel.LOW
        news_signal: NewsSignal | None = None

        # --- Check 1: Circuit Breaker ---
        if circuit_breaker_active:
            detected_regimes.append(RegimeType.CIRCUIT_BREAKER)
            reasons.append("Circuit breaker triggered — trading halted by KRX")
            should_trade = False
            risk_level = RiskLevel.EXTREME

        # --- Check 2: Sidecar ---
        if sidecar_active:
            detected_regimes.append(RegimeType.SIDECAR)
            reasons.append(
                "Sidecar triggered — program trading halted, "
                "high directional momentum"
            )
            should_trade = False
            risk_level = max(risk_level, RiskLevel.HIGH, key=_risk_ord)

        # --- Check 3: Expiry Day (만기일) ---
        if _is_expiry_day(check_date):
            detected_regimes.append(RegimeType.EXPIRY_DAY)
            reasons.append(
                f"{check_date.isoformat()} is a derivatives expiry day (만기일) — "
                f"expect elevated volume and volatility"
            )
            if self._expiry_day_halt:
                should_trade = False
            risk_level = max(risk_level, RiskLevel.MEDIUM, key=_risk_ord)

        # --- Check 4: VKOSPI ---
        vkospi_elevated, vkospi_risk, vkospi_reason = _assess_vkospi(vkospi)
        if vkospi_elevated:
            detected_regimes.append(RegimeType.HIGH_VOLATILITY)
            if vkospi_reason:
                reasons.append(vkospi_reason)
            risk_level = max(risk_level, vkospi_risk, key=_risk_ord)
            if self._vkospi_extreme_halt and vkospi is not None and vkospi >= _VKOSPI_EXTREME:
                should_trade = False

        # --- Check 5: News Event Filter ---
        if not skip_news:
            news_signal = self._check_news()
            if news_signal is not None:
                if news_signal.sentiment_score >= _NEWS_CRITICAL_THRESHOLD:
                    detected_regimes.append(RegimeType.EVENT_DRIVEN)
                    reasons.append(
                        f"Critical news event detected (sentiment={news_signal.sentiment_score:.2f}, "
                        f"urgency={news_signal.urgency_level}) — "
                        f"keyword spike across categories: "
                        f"{', '.join(k for k, v in news_signal.keyword_hits.items() if v > 0)}"
                    )
                    should_trade = False
                    risk_level = max(risk_level, RiskLevel.EXTREME, key=_risk_ord)
                elif news_signal.sentiment_score >= _NEWS_EVENT_THRESHOLD:
                    detected_regimes.append(RegimeType.EVENT_DRIVEN)
                    reasons.append(
                        f"Elevated news activity (sentiment={news_signal.sentiment_score:.2f}, "
                        f"urgency={news_signal.urgency_level}) — "
                        f"event-driven regime, signals may be less reliable"
                    )
                    risk_level = max(risk_level, RiskLevel.HIGH, key=_risk_ord)

        # --- Determine final regime type ---
        if len(detected_regimes) == 0:
            regime_type = RegimeType.NORMAL
        elif len(detected_regimes) == 1:
            regime_type = detected_regimes[0]
        else:
            regime_type = RegimeType.COMBINED

        if not reasons:
            reasons.append("All checks passed — normal market conditions")

        return RegimeResult(
            should_trade=should_trade,
            regime_type=regime_type,
            risk_level=risk_level,
            reasons=reasons,
            vkospi_level=vkospi,
            news_signal=news_signal,
        )

    def _check_news(self) -> NewsSignal | None:
        """Collect news signal, returning None on failure."""
        collector = self._news_collector
        if collector is None:
            collector = NewsCollector()

        try:
            return collector.collect_signal()
        except Exception as exc:
            logger.warning("News collection failed in regime gate: %s", exc)
            return None


def _risk_ord(level: RiskLevel) -> int:
    """Ordinal for risk level comparison via max()."""
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EXTREME": 3}.get(level.value, 0)
