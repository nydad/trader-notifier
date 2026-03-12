"""Unified signal pipeline -- the single entry point for trading signal generation.

Orchestrates the full signal generation process with market phase awareness:
  1. Phase detection (premarket / opening / intraday / closing / postmarket)
  2. Source selection (only real-time sources during intraday)
  3. Data fetching (yfinance for global, domestic.py for Korean market)
  4. Signal construction with age_seconds tracking and phase-adaptive weights
  5. Bayesian log-odds fusion
  6. RegimeGate check (expiry, circuit breaker, VKOSPI extreme)
  7. Confidence estimation via ConfidenceCalculator
  8. Human-readable consideration generation (Korean)

Usage::

    pipeline = SignalPipeline()
    result = pipeline.generate()
    print(result.direction, result.long_probability, result.confidence)
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from kospi_corr.domain.types import SignalDirection
from kospi_corr.engine.bayesian import (
    BayesianEngine,
    BayesianResult,
    BayesianSignal,
    RegimeDetector,
)
from kospi_corr.engine.confidence import ConfidenceCalculator
from kospi_corr.engine.market_phase import (
    KST,
    MarketPhase,
    SignalSourceConfig,
    detect_phase,
    get_phase_sources,
    get_phase_tau,
    get_phase_weight,
)
from kospi_corr.signals.regime import RegimeGate, RegimeResult

# Domestic collector may be created in parallel -- graceful fallback.
try:
    from kospi_corr.collectors.domestic import (
        DomesticSnapshot,
        fetch_domestic_snapshot,
    )
    _HAS_DOMESTIC = True
except ImportError:
    _HAS_DOMESTIC = False
    DomesticSnapshot = None  # type: ignore[assignment,misc]

    def fetch_domestic_snapshot() -> None:  # type: ignore[misc]
        return None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sigmoid helper (z-score -> probability)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically-safe sigmoid clipped to (0.01, 0.99)."""
    return float(np.clip(1.0 / (1.0 + np.exp(-x)), 0.01, 0.99))


# ---------------------------------------------------------------------------
# SignalResult -- complete pipeline output
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """Complete signal output with all metadata."""

    phase: MarketPhase
    direction: str               # "LONG", "SHORT", "NEUTRAL"
    long_probability: float
    short_probability: float
    confidence: float
    key_signals: list[dict[str, Any]]  # sorted by |contribution|
    regime: str                  # regime name (e.g. "trending_up")
    vkospi: float
    gate_status: dict[str, Any]  # {"should_trade", "risk_level", "reasons"}
    considerations: list[str]    # Human-readable considerations (Korean)
    timestamp: datetime
    data_ages: dict[str, float]  # source_key -> age_seconds
    domestic_snapshot: Any = None  # DomesticSnapshot reference for predictive engine

    @property
    def direction_kr(self) -> str:
        """Korean label for the direction."""
        return {
            "LONG": "매수(LONG)",
            "SHORT": "매도(SHORT)",
            "NEUTRAL": "관망(NEUTRAL)",
        }.get(self.direction, self.direction)


# ---------------------------------------------------------------------------
# SignalPipeline
# ---------------------------------------------------------------------------

class SignalPipeline:
    """Phase-aware signal generation pipeline.

    This is the SINGLE entry point for generating trading signals.
    It handles:
      1. Phase detection
      2. Source selection (only real-time sources during intraday)
      3. Data fetching (yfinance for global, domestic.py for Korean)
      4. Signal construction with proper age_seconds tracking
      5. Bayesian fusion with phase-appropriate weights
      6. RegimeGate check (expiry, CB, VKOSPI extreme)
      7. Confidence estimation
      8. Human-readable consideration generation
    """

    def __init__(
        self,
        enable_news: bool = True,
        enable_gate: bool = True,
    ) -> None:
        self._enable_news = enable_news
        self._enable_gate = enable_gate

        self._bayesian = BayesianEngine(prior=0.5)
        self._confidence = ConfidenceCalculator()
        self._gate = RegimeGate() if enable_gate else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        phase_override: MarketPhase | None = None,
    ) -> SignalResult:
        """Generate a complete signal for the current market phase.

        Parameters
        ----------
        phase_override : MarketPhase | None
            Force a specific phase instead of auto-detecting from KST clock.

        Returns
        -------
        SignalResult
        """
        now = datetime.now(KST)
        phase = phase_override if phase_override is not None else detect_phase(now)
        logger.info(
            "Signal pipeline started -- phase=%s, time=%s",
            phase, now.strftime("%H:%M KST"),
        )

        # 1. Select sources for this phase
        sources = get_phase_sources(phase)

        # 2. Fetch yfinance data
        fetch_start = datetime.now(KST)
        yf_data = self._fetch_yfinance_data(sources)
        fetch_elapsed = (datetime.now(KST) - fetch_start).total_seconds()
        logger.info(
            "yfinance fetch complete -- %d sources in %.1fs",
            len(yf_data), fetch_elapsed,
        )

        # 3. Fetch domestic data (only during KRX trading hours)
        domestic = self._fetch_domestic(phase)

        # 4. Determine VKOSPI for regime detection
        vkospi = self._extract_vkospi(yf_data, domestic)

        # 5. Build Bayesian signals with phase weights + age tracking
        signals = self._build_signals(yf_data, domestic, phase)
        if not signals:
            logger.warning("No valid signals built -- returning NEUTRAL")
            return self._empty_result(phase, vkospi, now)

        # 6. KOSPI200 trend slope (for regime detector)
        trend_slope = self._extract_trend_slope(yf_data, domestic)

        # 7. Bayesian fusion
        bayesian_result = self._bayesian.compute(
            signals=signals,
            vkospi=vkospi,
            trend_slope=trend_slope,
        )

        # 8. Confidence (thorough calculation)
        conf_breakdown = self._confidence.calculate(bayesian_result)
        confidence = conf_breakdown.overall

        # 9. RegimeGate check
        gate_result = self._check_gate(vkospi)

        # 10. Final direction (gate can override to NEUTRAL)
        direction = bayesian_result.direction.name
        if gate_result is not None and not gate_result.should_trade:
            direction = "NEUTRAL"

        # 11. Data ages map
        data_ages = self._compute_data_ages(yf_data, domestic, fetch_start)

        # 12. Human-readable considerations
        considerations = self._generate_considerations(
            bayesian_result, yf_data, domestic, phase, gate_result,
        )

        gate_dict = self._gate_to_dict(gate_result)

        return SignalResult(
            phase=phase,
            direction=direction,
            long_probability=bayesian_result.long_probability,
            short_probability=bayesian_result.short_probability,
            confidence=confidence,
            key_signals=bayesian_result.key_signals,
            regime=bayesian_result.regime.regime.value,
            vkospi=vkospi,
            gate_status=gate_dict,
            considerations=considerations,
            timestamp=now,
            data_ages=data_ages,
            domestic_snapshot=domestic,
        )

    # ------------------------------------------------------------------
    # Data fetching -- yfinance
    # ------------------------------------------------------------------

    def _fetch_yfinance_data(
        self,
        sources: dict[str, SignalSourceConfig],
    ) -> dict[str, dict[str, Any]]:
        """Fetch yfinance data for all sources that have a yf_symbol.

        Returns a dict keyed by source key.  Each value contains:
        ``current``, ``prev_close``, ``change_pct``, ``mean_ret``,
        ``std_ret``, ``z_score``, ``prob``, ``source``.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed -- cannot fetch market data")
            return {}

        result: dict[str, dict[str, Any]] = {}

        for key, source in sources.items():
            if not source.yf_symbol:
                continue  # domestic-only sources (vkospi, foreign_flow)

            try:
                _time.sleep(0.1)  # rate-limit (minimal for yfinance)
                data = self._fetch_single_ticker(yf, source)
                if data is not None:
                    result[key] = data
                    logger.info(
                        "%s: %.2f (%+.2f%%)",
                        source.label,
                        data["current"],
                        data["change_pct"] * 100,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch %s (%s): %s", key, source.yf_symbol, exc,
                )

        return result

    @staticmethod
    def _fetch_single_ticker(
        yf: Any,
        source: SignalSourceConfig,
    ) -> dict[str, Any] | None:
        """Fetch price + 30-day stats for one yfinance ticker."""
        t = yf.Ticker(source.yf_symbol)
        current: float | None = None
        prev_close: float | None = None

        # Primary: fast_info
        try:
            fi = t.fast_info
            current = fi.get("lastPrice") or fi.get("last_price")
            prev_close = fi.get("previousClose") or fi.get("previous_close")
        except Exception:
            pass

        # Fallback: 5-day history
        if current is None or prev_close is None:
            try:
                hist = t.history(period="5d")
                if hist is not None and not hist.empty:
                    close_col = "Close" if "Close" in hist.columns else "close"
                    current = current or float(hist[close_col].iloc[-1])
                    if len(hist) >= 2:
                        prev_close = prev_close or float(hist[close_col].iloc[-2])
            except Exception:
                return None

        if current is None or prev_close is None or prev_close == 0:
            return None

        # change_pct is always raw (positive = price went up).
        # Directional interpretation is handled by source.direction in z-score.
        # invert_quote is a legacy flag — NOT used for sign flipping.
        change_pct = (current - prev_close) / prev_close

        # 30-day stats for z-score
        mean_ret = 0.0
        std_ret = 0.01  # safe default
        try:
            hist30 = t.history(period="1mo")
            if hist30 is not None and len(hist30) >= 5:
                close_col = "Close" if "Close" in hist30.columns else "close"
                rets = hist30[close_col].pct_change().dropna()
                if len(rets) >= 3:
                    mean_ret = float(rets.mean())
                    std_ret = max(float(rets.std()), 1e-6)
        except Exception:
            pass

        # z-score -> probability
        z = (change_pct - mean_ret) / std_ret
        z_dir = z * source.direction
        prob = _sigmoid(z_dir)

        return {
            "current": float(current),
            "prev_close": float(prev_close),
            "change_pct": float(change_pct),
            "mean_ret": mean_ret,
            "std_ret": std_ret,
            "z_score": float(z),
            "prob": prob,
            "source": source,
        }

    # ------------------------------------------------------------------
    # Data fetching -- domestic
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_domestic(phase: MarketPhase) -> Any:
        """Fetch domestic snapshot during KRX trading phases."""
        domestic_phases = {
            MarketPhase.OPENING,
            MarketPhase.INTRADAY,
            MarketPhase.CLOSING,
        }
        if phase not in domestic_phases:
            return None
        if not _HAS_DOMESTIC:
            logger.debug("Domestic collector not available -- skipping")
            return None

        try:
            return fetch_domestic_snapshot()
        except Exception as exc:
            logger.warning("Domestic snapshot fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        yf_data: dict[str, dict[str, Any]],
        domestic: Any,
        phase: MarketPhase,
    ) -> list[BayesianSignal]:
        """Build BayesianSignal list with phase weights and age tracking."""
        tau = get_phase_tau(phase)
        signals: list[BayesianSignal] = []

        # Global sources from yfinance
        for key, data in yf_data.items():
            source: SignalSourceConfig = data["source"]
            phase_mult = get_phase_weight(phase, source.key)
            effective_weight = source.base_weight * phase_mult

            signals.append(BayesianSignal(
                name=source.key,
                probability=data["prob"],
                weight=effective_weight,
                category=source.category,
                age_seconds=0.0,  # just fetched
                tau_seconds=tau,
            ))

        # Domestic signals (if available)
        if domestic is not None and _HAS_DOMESTIC:
            signals.extend(self._domestic_to_signals(domestic, phase, tau))

        return signals

    @staticmethod
    def _domestic_to_signals(
        domestic: Any,
        phase: MarketPhase,
        tau: float,
    ) -> list[BayesianSignal]:
        """Convert DomesticSnapshot fields to BayesianSignals."""
        signals: list[BayesianSignal] = []

        # KOSPI200 change -> signal
        try:
            kospi200_change = getattr(domestic, "kospi200_change_pct", None)
            if kospi200_change is not None:
                z = kospi200_change / 0.01  # ~1% move = 1 sigma
                prob = _sigmoid(z)
                phase_mult = get_phase_weight(phase, "kospi200")
                signals.append(BayesianSignal(
                    name="kospi200",
                    probability=prob,
                    weight=2.0 * phase_mult,
                    category="domestic_index",
                    age_seconds=0.0,
                    tau_seconds=tau,
                ))
        except Exception as exc:
            logger.debug("KOSPI200 signal construction failed: %s", exc)

        # Foreign flow -> signal (P(LONG) directly if available)
        try:
            foreign_prob = getattr(domestic, "foreign_flow_p_long", None)
            if foreign_prob is not None and 0 < foreign_prob < 1:
                phase_mult = get_phase_weight(phase, "foreign_flow")
                signals.append(BayesianSignal(
                    name="foreign_flow",
                    probability=foreign_prob,
                    weight=1.8 * phase_mult,
                    category="domestic_flow",
                    age_seconds=0.0,
                    tau_seconds=tau,
                ))
        except Exception as exc:
            logger.debug("Foreign flow signal construction failed: %s", exc)

        # VKOSPI level -> signal (high VKOSPI = bearish)
        try:
            vkospi_level = getattr(domestic, "vkospi", None)
            if vkospi_level is not None and vkospi_level > 0:
                # 15=bullish(0.7), 25=neutral(0.5), 35+=bearish(0.3)
                z_vkospi = -(vkospi_level - 20.0) / 5.0
                prob = _sigmoid(z_vkospi)
                phase_mult = get_phase_weight(phase, "vkospi")
                signals.append(BayesianSignal(
                    name="vkospi",
                    probability=prob,
                    weight=1.5 * phase_mult,
                    category="sentiment",
                    age_seconds=0.0,
                    tau_seconds=tau,
                ))
        except Exception as exc:
            logger.debug("VKOSPI signal construction failed: %s", exc)

        # Program trading -> signal (net buying = bullish)
        try:
            prog_p_long = getattr(domestic, "program_p_long", None)
            prog_total = getattr(domestic, "program_total_net", 0.0)
            if prog_p_long is not None and prog_total != 0:
                phase_mult = get_phase_weight(phase, "program_trading")
                signals.append(BayesianSignal(
                    name="program_trading",
                    probability=prog_p_long,
                    weight=1.5 * phase_mult,
                    category="domestic_flow",
                    age_seconds=0.0,
                    tau_seconds=tau,
                ))
        except Exception as exc:
            logger.debug("Program trading signal construction failed: %s", exc)

        return signals

    # ------------------------------------------------------------------
    # VKOSPI and trend extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_vkospi(yf_data: dict[str, dict], domestic: Any) -> float:
        """Extract VKOSPI level, preferring domestic real-time data."""
        # Prefer domestic snapshot
        if domestic is not None:
            vk = getattr(domestic, "vkospi", None)
            if vk is not None and vk > 0:
                return float(vk)

        # Fallback: VIX from yfinance (rough proxy)
        vix_data = yf_data.get("vix")
        if vix_data is not None:
            return float(vix_data["current"])

        return 18.0  # neutral default

    @staticmethod
    def _extract_trend_slope(yf_data: dict[str, dict], domestic: Any) -> float:
        """Extract KOSPI200 trend slope for regime detection."""
        if domestic is not None:
            slope = getattr(domestic, "trend_slope", None)
            if slope is not None:
                return float(slope)
            chg = getattr(domestic, "kospi200_change_pct", None)
            if chg is not None:
                return float(chg)

        # Approximate from S&P500 futures change
        sp_data = yf_data.get("sp500_fut") or yf_data.get("sp500")
        if sp_data is not None:
            return sp_data.get("change_pct", 0.0)

        return 0.0

    # ------------------------------------------------------------------
    # RegimeGate
    # ------------------------------------------------------------------

    def _check_gate(self, vkospi: float) -> RegimeResult | None:
        """Run regime gate checks.  Returns None if gate is disabled."""
        if self._gate is None:
            return None

        try:
            return self._gate.check(
                vkospi=vkospi,
                skip_news=not self._enable_news,
            )
        except Exception as exc:
            logger.warning("RegimeGate check failed: %s", exc)
            return None

    @staticmethod
    def _gate_to_dict(gate_result: RegimeResult | None) -> dict[str, Any]:
        """Convert RegimeResult to a plain dict for SignalResult."""
        if gate_result is None:
            return {
                "should_trade": True,
                "risk_level": "LOW",
                "reasons": ["RegimeGate disabled"],
            }
        return {
            "should_trade": gate_result.should_trade,
            "risk_level": gate_result.risk_level.value,
            "reasons": gate_result.reasons,
        }

    # ------------------------------------------------------------------
    # Data ages
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_data_ages(
        yf_data: dict[str, dict],
        domestic: Any,
        fetch_start: datetime,
    ) -> dict[str, float]:
        """Compute age_seconds for each fetched source."""
        now = datetime.now(KST)
        base_age = (now - fetch_start).total_seconds()
        ages: dict[str, float] = {}
        for key in yf_data:
            ages[key] = base_age
        if domestic is not None:
            for name in ("kospi200", "foreign_flow", "vkospi", "program_trading"):
                if name == "program_trading":
                    if getattr(domestic, "program_total_net", 0.0) != 0:
                        ages[name] = base_age
                elif getattr(domestic, name, None) is not None or \
                     getattr(domestic, f"{name}_change_pct", None) is not None:
                    ages[name] = base_age
        return ages

    # ------------------------------------------------------------------
    # Consideration generation (Korean)
    # ------------------------------------------------------------------

    def _generate_considerations(
        self,
        result: BayesianResult,
        yf_data: dict[str, dict],
        domestic: Any,
        phase: MarketPhase,
        gate_result: RegimeResult | None,
    ) -> list[str]:
        """Generate actionable Korean text considerations for the trader."""
        items: list[str] = []

        # Phase context
        _phase_labels = {
            MarketPhase.PREMARKET: "장전 모드 -- 해외 종가 + 선물 기반 갭 예측",
            MarketPhase.OPENING: "개장 모드 -- 갭 흡수 중, 변동성 높음",
            MarketPhase.INTRADAY: "장중 모드 -- 해외 종가 제외, 실시간 데이터만 반영",
            MarketPhase.CLOSING: "장마감 임박 -- 프로그램 매매 주의",
            MarketPhase.POSTMARKET: "장후 분석 모드 -- 실시간 시그널 제한",
        }
        items.append(_phase_labels.get(phase, f"현재 페이즈: {phase}"))

        # FX trend
        fx_data = yf_data.get("usd_krw")
        if fx_data is not None:
            chg = fx_data["change_pct"]
            dir_kr = "약세" if chg > 0 else "강세"
            impact = "인버스 포지션 유리" if chg > 0 else "롱 포지션 유리"
            items.append(f"원화 {dir_kr} ({chg:+.2%}) -> {impact}")

        # S&P500 futures
        sp_data = yf_data.get("sp500_fut")
        if sp_data is not None:
            items.append(f"S&P500 선물 {sp_data['change_pct']:+.2%}")

        # NASDAQ futures
        nq_data = yf_data.get("nasdaq_fut")
        if nq_data is not None:
            items.append(f"NASDAQ 선물 {nq_data['change_pct']:+.2%}")

        # Pre-market overnight closes
        if phase == MarketPhase.PREMARKET:
            for sym_key, label in [("sp500", "S&P500"), ("nasdaq", "NASDAQ")]:
                close_data = yf_data.get(sym_key)
                if close_data is not None:
                    items.append(
                        f"{label} 전일 종가 {close_data['change_pct']:+.2%}"
                    )

        # WTI (only if significant)
        wti_data = yf_data.get("wti_crude")
        if wti_data is not None:
            chg = wti_data["change_pct"]
            if abs(chg) >= 0.02:
                items.append(f"WTI {chg:+.2%} -- 유가 급변, 환율 2차 영향 주의")

        # KOSPI200 momentum (domestic)
        if domestic is not None:
            k200_chg = getattr(domestic, "kospi200_change_pct", None)
            if k200_chg is not None:
                dir_kr = "상승" if k200_chg > 0 else "하락"
                momentum = "롱" if k200_chg > 0 else "숏"
                items.append(
                    f"KOSPI200 {k200_chg:+.1%} {dir_kr} 중 -> {momentum} 모멘텀"
                )

            foreign_net = getattr(domestic, "foreign_net", None)
            if foreign_net is not None:
                dir_kr = "순매수" if foreign_net > 0 else "순매도"
                impact = "상승 지지" if foreign_net > 0 else "하락 압력"
                items.append(
                    f"외국인 {dir_kr} {abs(foreign_net):,.0f}억 -> {impact}"
                )

            # Program trading
            prog_total = getattr(domestic, "program_total_net", 0.0)
            if prog_total != 0:
                prog_arb = getattr(domestic, "program_arb_net", 0.0)
                prog_nonarb = getattr(domestic, "program_nonarb_net", 0.0)
                prog_dir = "순매수" if prog_total > 0 else "순매도"
                prog_impact = "상승 지지" if prog_total > 0 else "하락 압력"
                items.append(
                    f"프로그램 {prog_dir} {abs(prog_total):,.0f}억 "
                    f"(차익 {prog_arb:+,.0f}, 비차익 {prog_nonarb:+,.0f}) "
                    f"-> {prog_impact}"
                )

        # Signal confidence
        conf_pct = result.confidence * 100
        if conf_pct >= 75:
            items.append(f"시그널 신뢰도 {conf_pct:.0f}% -> 적극 대응")
        elif conf_pct >= 55:
            items.append(f"시그널 신뢰도 {conf_pct:.0f}% -> 일반 대응")
        else:
            items.append(f"시그널 신뢰도 {conf_pct:.0f}% -> 보수적 대응 권고")

        # Regime info
        items.append(f"레짐: {result.regime.regime.value}")

        # Gate warnings
        if gate_result is not None:
            if not gate_result.should_trade:
                items.append("RegimeGate: 매매 중단 권고")
                for reason in gate_result.reasons:
                    items.append(f"  - {reason}")
            elif gate_result.risk_level.value in ("HIGH", "EXTREME"):
                items.append(f"주의: 리스크 레벨 {gate_result.risk_level.value}")
                for reason in gate_result.reasons:
                    items.append(f"  - {reason}")

        # Direction-specific recommendation
        if result.direction == SignalDirection.LONG:
            prob_pct = result.long_probability * 100
            items.append(
                f"-> LONG 시그널 (P={prob_pct:.1f}%) -- 레버리지 ETF 매수 기회"
            )
        elif result.direction == SignalDirection.SHORT:
            prob_pct = result.short_probability * 100
            items.append(
                f"-> SHORT 시그널 (P={prob_pct:.1f}%) -- 인버스2X(252670) 매수 기회"
            )
        else:
            items.append("-> NEUTRAL -- 관망 또는 스캘핑 위주 권고")

        # Disclaimer
        items.append(
            "* 본 시그널은 참고용이며, "
            "투자 판단의 최종 책임은 투자자에게 있습니다."
        )

        return items

    # ------------------------------------------------------------------
    # Empty result fallback
    # ------------------------------------------------------------------

    def _empty_result(
        self,
        phase: MarketPhase,
        vkospi: float,
        now: datetime,
    ) -> SignalResult:
        """Return a safe NEUTRAL result when no signals are available."""
        gate_result = self._check_gate(vkospi)
        return SignalResult(
            phase=phase,
            direction="NEUTRAL",
            long_probability=0.5,
            short_probability=0.5,
            confidence=0.0,
            key_signals=[],
            regime="range_bound",
            vkospi=vkospi,
            gate_status=self._gate_to_dict(gate_result),
            considerations=[
                f"{phase.value} 페이즈 -- 유효한 시그널 없음",
                "데이터 수신 실패 또는 시장 휴장 가능성",
                "* 본 시그널은 참고용이며, "
                "투자 판단의 최종 책임은 투자자에게 있습니다.",
            ],
            timestamp=now,
            data_ages={},
        )
