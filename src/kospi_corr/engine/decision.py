"""Decision engine — direct actionable trading recommendations.

Transforms signals, predictions, and market context into a single
DIRECT trading decision: "인버스2X 10만주 매수" not "P(LONG)=42%".

Designed for aggressive daytrader profile:
  - KODEX 200선물인버스2X (252670) 주력 숏 — 10만주+ 대량
  - 레버리지 ETF 롱 — 섹터별 분산
  - 리스크 감수, 수익률 극대화 지향
  - 1분 내 판단 필요

Usage::

    engine = DecisionEngine()
    decision = engine.decide(signal_result, predictive_result)
    # decision.action_kr = "인버스2X 매수 (10만주)"
    # decision.scenarios = [QuantScenario(...), ...]
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuantScenario:
    """One quantitative scenario with conditional probability."""
    name_kr: str
    probability: float
    trigger_kr: str
    expected_move_pct: float
    etf_code: str
    etf_name: str
    time_horizon_kr: str
    action_kr: str


@dataclass
class TradingDecision:
    """Complete actionable trading recommendation."""

    # Core decision
    action: str                     # BUY_SHORT, BUY_LONG, HOLD, EXIT, WAIT
    action_kr: str                  # "인버스2X 매수", "레버리지 매수", "관망"
    etf_code: str                   # 252670, 122630, etc.
    etf_name: str
    direction: str                  # LONG, SHORT, NEUTRAL

    # Confidence & urgency
    confidence: float               # 0-1
    urgency: str                    # 즉시, 조건부, 대기, 관망
    urgency_reason_kr: str

    # Entry/Exit levels
    entry_hint_kr: str              # "현재가 부근 진입"
    target_kr: str                  # "+5~7% (268원 목표)"
    stop_loss_kr: str               # "-2% (245원 손절)"
    position_hint_kr: str           # "10만주 기준 2,500만원"

    # Analysis
    reason_kr: str                  # 핵심 판단 근거 1줄
    market_context_kr: list[str]    # 시장 상황 요약 (3-5줄)
    risk_factors_kr: list[str]      # 리스크 요인

    # Quant scenarios
    scenarios: list[QuantScenario]

    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(KST))


# ---------------------------------------------------------------------------
# ETF catalog
# ---------------------------------------------------------------------------

_ETF_CATALOG = {
    "SHORT": {
        "default": ("252670", "KODEX 인버스2X"),
    },
    "LONG": {
        "default": ("122630", "KODEX 레버리지"),
        "semiconductor": ("091170", "KODEX 반도체레버리지"),
        "defense": ("472170", "KODEX 방산레버리지"),
        "kosdaq": ("233740", "KODEX 코스닥150레버리지"),
    },
}

# Inverse 2X typical price range for position sizing
_INVERSE2X_PRICE_RANGE = (200, 400)  # won
_DEFAULT_POSITION_SIZE = 100000  # 10만주


# ---------------------------------------------------------------------------
# Sector signal keywords
# ---------------------------------------------------------------------------

_SECTOR_KEYWORDS = {
    "semiconductor": ["반도체", "semiconductor", "hynix", "삼성전자", "nvidia",
                       "ai칩", "gpu", "메모리"],
    "defense": ["방산", "defense", "한화", "이란", "전쟁", "미사일", "military"],
    "kosdaq": ["코스닥", "kosdaq", "바이오", "2차전지"],
}


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Produces direct actionable trading decisions.

    Combines:
      - SignalResult (Bayesian direction + confidence)
      - PredictiveResult (momentum + leading signals + triggers)
      - Market context (considerations from pipeline)

    Output: TradingDecision with specific ETF, position size, entry/exit.
    """

    # Thresholds for decision making
    STRONG_SIGNAL = 0.62        # P(direction) > this = strong
    MODERATE_SIGNAL = 0.55      # P(direction) > this = moderate
    HIGH_CONFIDENCE = 0.65      # confidence > this = high
    MOMENTUM_CONFIRM = 0.5      # leading signal strength sum

    def decide(
        self,
        signal: Any,
        predictive: Any | None = None,
    ) -> TradingDecision:
        """Generate a trading decision from signal + predictive results.

        Args:
            signal: SignalResult from SignalPipeline
            predictive: PredictiveResult from PredictiveEngine (optional)
        """
        p_long = getattr(signal, "long_probability", 0.5)
        p_short = 1.0 - p_long
        confidence = getattr(signal, "confidence", 0.5)
        direction = getattr(signal, "direction", "NEUTRAL")
        considerations = getattr(signal, "considerations", [])
        phase = getattr(signal, "phase", "intraday")
        domestic = getattr(signal, "domestic_snapshot", None)

        # Extract predictive context
        leading = []
        momentum = {}
        triggers = []
        scenarios_pred = []
        if predictive is not None:
            leading = getattr(predictive, "leading_signals", [])
            momentum = getattr(predictive, "momentum", {})
            triggers = getattr(predictive, "triggers", [])
            scenarios_pred = getattr(predictive, "scenarios", [])

        # --- 1. Determine core direction & strength ---
        strength = self._classify_strength(p_long, p_short, direction)

        # --- 2. Check momentum confirmation ---
        momentum_confirms = self._check_momentum(
            direction, leading, momentum,
        )

        # --- 3. Pick ETF ---
        etf_code, etf_name = self._pick_etf(
            direction, considerations, leading,
        )

        # --- 4. Determine urgency ---
        urgency, urgency_reason = self._determine_urgency(
            strength, confidence, momentum_confirms, leading, phase,
        )

        # --- 5. Calculate entry/exit ---
        entry_hint, target_hint, stop_hint = self._calc_levels(
            direction, etf_code, domestic, strength,
        )

        # --- 6. Position sizing hint ---
        position_hint = self._position_hint(etf_code, direction)

        # --- 7. Build action string ---
        action, action_kr = self._build_action(
            direction, strength, urgency, etf_name,
        )

        # --- 8. Core reason ---
        reason_kr = self._build_reason(
            direction, p_long, p_short, confidence,
            leading, momentum, considerations,
        )

        # --- 9. Market context ---
        context_kr = self._build_context(
            considerations, domestic, momentum,
        )

        # --- 10. Risk factors ---
        risk_kr = self._build_risks(
            direction, leading, momentum, domestic,
        )

        # --- 11. Quant scenarios ---
        quant_scenarios = self._build_quant_scenarios(
            direction, p_long, p_short, confidence,
            leading, momentum, domestic, scenarios_pred,
        )

        return TradingDecision(
            action=action,
            action_kr=action_kr,
            etf_code=etf_code,
            etf_name=etf_name,
            direction=direction,
            confidence=confidence,
            urgency=urgency,
            urgency_reason_kr=urgency_reason,
            entry_hint_kr=entry_hint,
            target_kr=target_hint,
            stop_loss_kr=stop_hint,
            position_hint_kr=position_hint,
            reason_kr=reason_kr,
            market_context_kr=context_kr,
            risk_factors_kr=risk_kr,
            scenarios=quant_scenarios,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_strength(
        self, p_long: float, p_short: float, direction: str,
    ) -> str:
        """Classify signal strength: strong, moderate, weak."""
        p = p_long if direction == "LONG" else p_short
        if p >= self.STRONG_SIGNAL:
            return "strong"
        if p >= self.MODERATE_SIGNAL:
            return "moderate"
        return "weak"

    def _check_momentum(
        self,
        direction: str,
        leading: list,
        momentum: dict,
    ) -> bool:
        """Check if momentum/leading signals confirm the direction."""
        confirm_score = 0.0
        for sig in leading:
            impl = getattr(sig, "implication", "")
            strength = getattr(sig, "strength", 0)
            if impl == direction:
                confirm_score += strength
            elif impl in ("LONG", "SHORT") and impl != direction:
                confirm_score -= strength * 0.5
        return confirm_score >= self.MOMENTUM_CONFIRM

    def _pick_etf(
        self,
        direction: str,
        considerations: list[str],
        leading: list,
    ) -> tuple[str, str]:
        """Pick the best ETF for the direction."""
        if direction == "SHORT":
            return _ETF_CATALOG["SHORT"]["default"]

        if direction == "NEUTRAL":
            return "", "관망"

        # LONG — check sector signals
        all_text = " ".join(considerations)
        for sig in leading:
            all_text += " " + getattr(sig, "description_kr", "")

        for sector, keywords in _SECTOR_KEYWORDS.items():
            if any(kw in all_text for kw in keywords):
                if sector in _ETF_CATALOG["LONG"]:
                    return _ETF_CATALOG["LONG"][sector]

        return _ETF_CATALOG["LONG"]["default"]

    def _determine_urgency(
        self,
        strength: str,
        confidence: float,
        momentum_confirms: bool,
        leading: list,
        phase: Any,
    ) -> tuple[str, str]:
        """Determine urgency level."""
        # Immediate: strong signal + high confidence + momentum confirms
        if strength == "strong" and confidence >= self.HIGH_CONFIDENCE:
            if momentum_confirms:
                return "즉시", "강한 시그널 + 모멘텀 확인 → 즉시 진입"
            return "즉시", "강한 시그널 → 진입 타이밍"

        # Conditional: moderate signal or waiting for confirmation
        if strength == "moderate":
            if momentum_confirms:
                return "조건부", "중간 시그널 + 모멘텀 확인 → 트리거 대기"
            return "조건부", "중간 시그널 → 추가 확인 필요"

        # Inflection detected — watch closely
        has_inflection = any(
            getattr(s, "signal_type", "") == "reversal"
            for s in leading
        )
        if has_inflection:
            return "대기", "전환 신호 감지 — 방향 확정 대기"

        return "관망", "약한 시그널 — 방향 불확실"

    def _calc_levels(
        self,
        direction: str,
        etf_code: str,
        domestic: Any,
        strength: str,
    ) -> tuple[str, str, str]:
        """Calculate entry, target, stop-loss hints."""
        if direction == "NEUTRAL":
            return "관망", "해당 없음", "해당 없음"

        # Inverse 2X specific
        if etf_code == "252670":
            return self._inverse2x_levels(strength)

        # Generic leveraged ETF
        kospi = None
        if domestic:
            kospi = getattr(domestic, "kospi200_current", None)

        if direction == "SHORT":
            target_pct = 3.0 if strength == "strong" else 2.0
            stop_pct = 1.5
        else:
            target_pct = 4.0 if strength == "strong" else 2.5
            stop_pct = 2.0

        entry = "현재가 부근 진입"
        target = f"+{target_pct:.1f}% 목표"
        stop = f"-{stop_pct:.1f}% 손절"

        if kospi:
            if direction == "LONG":
                target += f" (KOSPI200 {kospi * (1 + target_pct/200):.1f})"
                stop += f" (KOSPI200 {kospi * (1 - stop_pct/200):.1f})"
            else:
                target += f" (KOSPI200 {kospi * (1 - target_pct/200):.1f})"
                stop += f" (KOSPI200 {kospi * (1 + stop_pct/200):.1f})"

        return entry, target, stop

    def _inverse2x_levels(self, strength: str) -> tuple[str, str, str]:
        """Specific levels for KODEX Inverse 2X (252670).

        Based on user's actual trading pattern:
        - Buy range: 249-268원
        - Typical profit: +5~7% (15-20원)
        - Stop: -2~3% (5-8원)
        """
        if strength == "strong":
            entry = "현재가 즉시 진입"
            target = "+5~7% 목표 (15~20원 상승분)"
            stop = "-2% 손절 (5원 하락 시)"
        elif strength == "moderate":
            entry = "현재가 -1% 매수 대기 or 즉시 절반 진입"
            target = "+3~5% 목표 (10~15원 상승분)"
            stop = "-2% 손절 (5원 하락 시)"
        else:
            entry = "관망 후 트리거 확인 시 진입"
            target = "+2~3% 보수적 목표"
            stop = "-1.5% 타이트 손절"
        return entry, target, stop

    def _position_hint(self, etf_code: str, direction: str) -> str:
        """Position size hint based on user pattern."""
        if etf_code == "252670":
            return "10만주 기준 (가격 250원대 = ~2,500만원)"
        if direction == "NEUTRAL":
            return "관망"
        return "레버리지 ETF 적정 수량 (총 자산 5~10%)"

    def _build_action(
        self,
        direction: str,
        strength: str,
        urgency: str,
        etf_name: str,
    ) -> tuple[str, str]:
        """Build action code and Korean description."""
        if direction == "NEUTRAL" or urgency == "관망":
            return "WAIT", "관망 — 방향 불확실"

        if direction == "SHORT":
            if urgency == "즉시":
                return "BUY_SHORT", f"{etf_name} 매수 (즉시)"
            if urgency == "조건부":
                return "BUY_SHORT", f"{etf_name} 매수 대기 (트리거 확인)"
            return "WATCH_SHORT", f"{etf_name} 주시 (대기)"

        # LONG
        if urgency == "즉시":
            return "BUY_LONG", f"{etf_name} 매수 (즉시)"
        if urgency == "조건부":
            return "BUY_LONG", f"{etf_name} 매수 대기 (트리거 확인)"
        return "WATCH_LONG", f"{etf_name} 주시 (대기)"

    def _build_reason(
        self,
        direction: str,
        p_long: float,
        p_short: float,
        confidence: float,
        leading: list,
        momentum: dict,
        considerations: list[str],
    ) -> str:
        """Build 1-line core reason."""
        p = p_long if direction == "LONG" else p_short
        parts = []

        # Direction probability
        dir_kr = "LONG" if direction == "LONG" else "SHORT"
        parts.append(f"P({dir_kr})={p:.0%}")

        # Top leading signal
        for sig in leading[:1]:
            desc = getattr(sig, "description_kr", "")
            if desc:
                parts.append(desc[:30])

        # Top consideration
        if considerations:
            parts.append(considerations[0][:30])

        return " | ".join(parts)

    def _build_context(
        self,
        considerations: list[str],
        domestic: Any,
        momentum: dict,
    ) -> list[str]:
        """Build market context summary."""
        ctx = []

        # From considerations (top 3)
        for c in considerations[:3]:
            ctx.append(c)

        # Domestic data points
        if domestic:
            kospi = getattr(domestic, "kospi200_current", None)
            kospi_chg = getattr(domestic, "kospi200_change_pct", None)
            foreign = getattr(domestic, "foreign_net", 0)
            program = getattr(domestic, "program_total_net", 0)
            vkospi = getattr(domestic, "vkospi", 20)

            if kospi and kospi_chg is not None:
                ctx.append(f"KOSPI200: {kospi:.1f} ({kospi_chg:+.2f}%)")
            if foreign:
                ctx.append(f"외국인 순매수: {foreign:+,.0f}억원")
            if program:
                ctx.append(f"프로그램 순매수: {program:+,.0f}억원")
            if vkospi and vkospi != 20.0:
                ctx.append(f"VKOSPI: {vkospi:.1f}")

        return ctx[:6]

    def _build_risks(
        self,
        direction: str,
        leading: list,
        momentum: dict,
        domestic: Any,
    ) -> list[str]:
        """Build risk factor list."""
        risks = []

        # Opposing signals
        for sig in leading:
            impl = getattr(sig, "implication", "")
            if impl == "CAUTION":
                risks.append(getattr(sig, "description_kr", "주의 신호"))
            elif direction == "SHORT" and impl == "LONG":
                risks.append(
                    f"반대 신호: {getattr(sig, 'description_kr', '롱 신호')}"
                )
            elif direction == "LONG" and impl == "SHORT":
                risks.append(
                    f"반대 신호: {getattr(sig, 'description_kr', '숏 신호')}"
                )

        # VKOSPI risk
        if domestic:
            vkospi = getattr(domestic, "vkospi", 20)
            if vkospi > 25:
                risks.append(f"VKOSPI {vkospi:.1f} — 변동성 확대 구간")

        # Program trading opposing
        prog_m = momentum.get("program_total_net")
        if prog_m:
            vel = getattr(prog_m, "velocity", 0)
            if direction == "SHORT" and vel > 1000:
                risks.append("프로그램 매수 진행 중 — 숏 반대 압력")
            elif direction == "LONG" and vel < -1000:
                risks.append("프로그램 매도 진행 중 — 롱 반대 압력")

        if not risks:
            risks.append("특이 리스크 없음")

        return risks[:5]

    def _build_quant_scenarios(
        self,
        direction: str,
        p_long: float,
        p_short: float,
        confidence: float,
        leading: list,
        momentum: dict,
        domestic: Any,
        pred_scenarios: list,
    ) -> list[QuantScenario]:
        """Build multiple quant scenarios with probabilities."""
        scenarios = []

        # Use predictive scenarios as base if available
        for ps in pred_scenarios[:2]:
            ps_dir = getattr(ps, "direction", "")
            ps_prob = getattr(ps, "probability", 0)
            ps_label = getattr(ps, "label_kr", "")
            ps_etf = getattr(ps, "target_etf", "")
            ps_etf_name = getattr(ps, "target_etf_name", "")
            ps_trigger = getattr(ps, "entry_trigger_kr", "")
            ps_conditions = getattr(ps, "conditions_kr", [])

            if ps_dir == "LONG":
                move = 2.0 + ps_prob * 3.0
            elif ps_dir == "SHORT":
                move = -(2.0 + ps_prob * 3.0)
            else:
                move = 0.0

            action = ""
            if ps_dir == "SHORT":
                action = f"인버스2X 매수 → +{abs(move):.1f}% 목표"
            elif ps_dir == "LONG":
                action = f"{ps_etf_name} 매수 → +{abs(move):.1f}% 목표"
            else:
                action = "관망"

            cond_str = ", ".join(ps_conditions[:2]) if ps_conditions else ps_trigger

            scenarios.append(QuantScenario(
                name_kr=ps_label,
                probability=ps_prob,
                trigger_kr=cond_str,
                expected_move_pct=move,
                etf_code=ps_etf,
                etf_name=ps_etf_name,
                time_horizon_kr="30분~2시간",
                action_kr=action,
            ))

        # Add shock scenario (tail risk)
        scenarios.append(QuantScenario(
            name_kr="급변 시나리오",
            probability=0.08,
            trigger_kr="돌발 뉴스, 환율 급변, 프로그램 폭탄 매도/매수",
            expected_move_pct=-5.0 if direction == "LONG" else 5.0,
            etf_code="252670" if direction == "LONG" else "122630",
            etf_name="KODEX 인버스2X" if direction == "LONG" else "KODEX 레버리지",
            time_horizon_kr="10분 이내",
            action_kr="즉시 손절 후 반대 포지션 고려",
        ))

        # Add range-bound scenario
        scenarios.append(QuantScenario(
            name_kr="횡보 시나리오",
            probability=max(0.05, 1.0 - sum(s.probability for s in scenarios)),
            trigger_kr="외국인/프로그램 방향 혼재, 환율 보합",
            expected_move_pct=0.0,
            etf_code="",
            etf_name="관망",
            time_horizon_kr="장중 전체",
            action_kr="스캘핑 or 관망",
        ))

        scenarios.sort(key=lambda s: s.probability, reverse=True)
        return scenarios
