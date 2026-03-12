"""Predictive signal engine — forward-looking scenario analysis.

Transforms reactive "current state" signals into predictive scenarios
by tracking indicator momentum, detecting leading signals, and generating
conditional entry triggers.

Key insight: A day-trader needs to know what's LIKELY to happen next,
not what already happened. This module adds temporal context.

Usage::

    engine = PredictiveEngine()
    # Call update() every 5 minutes during trading hours
    result = engine.update(domestic_snapshot, signal_result)
    # result.scenarios[0] = most probable scenario with entry trigger
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimedSnapshot:
    """A point-in-time snapshot with extracted key values."""
    timestamp: datetime
    kospi200: float | None = None
    kospi200_change_pct: float | None = None
    kosdaq_current: float | None = None
    kosdaq_change_pct: float | None = None
    foreign_net: float = 0.0
    institution_net: float = 0.0
    program_total_net: float = 0.0
    program_arb_net: float = 0.0
    program_nonarb_net: float = 0.0
    vkospi: float = 20.0
    long_probability: float = 0.5


@dataclass
class MomentumState:
    """Momentum for one indicator."""
    name: str
    current: float
    velocity: float             # units per hour
    acceleration: float         # change in velocity
    trend_duration_min: float   # minutes in current direction
    is_accelerating: bool = False
    is_decelerating: bool = False
    inflection_detected: bool = False


@dataclass
class LeadingSignal:
    """A detected leading indicator pattern."""
    indicator: str
    signal_type: str        # reversal, continuation, divergence, threshold
    description_kr: str
    implication: str        # LONG, SHORT, CAUTION
    strength: float         # 0-1


@dataclass
class Scenario:
    """A forward-looking trading scenario."""
    direction: str              # LONG, SHORT, NEUTRAL
    probability: float
    label_kr: str
    conditions_kr: list[str]
    entry_trigger_kr: str
    target_etf: str
    target_etf_name: str
    risk_factors_kr: list[str]


@dataclass
class EntryTrigger:
    """A specific actionable entry condition."""
    direction: str
    condition_kr: str
    indicator: str
    current_value: float
    threshold_value: float
    distance_pct: float
    target_etf: str


@dataclass
class PredictiveResult:
    """Complete predictive analysis output."""
    scenarios: list[Scenario]
    momentum: dict[str, MomentumState]
    leading_signals: list[LeadingSignal]
    triggers: list[EntryTrigger]
    buffer_depth: int
    warmup_complete: bool
    generated_at: datetime = field(default_factory=lambda: datetime.now(KST))


# ---------------------------------------------------------------------------
# SnapshotBuffer — temporal memory
# ---------------------------------------------------------------------------

class SnapshotBuffer:
    """Ring buffer storing recent snapshots for trend analysis."""

    def __init__(self, maxlen: int = 60) -> None:
        self._buf: deque[TimedSnapshot] = deque(maxlen=maxlen)

    def push(self, snap: TimedSnapshot) -> None:
        self._buf.append(snap)

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> TimedSnapshot | None:
        return self._buf[-1] if self._buf else None

    def window(self, minutes: int) -> list[TimedSnapshot]:
        if not self._buf:
            return []
        cutoff = self._buf[-1].timestamp - timedelta(minutes=minutes)
        return [s for s in self._buf if s.timestamp >= cutoff]

    def series(self, field_name: str) -> list[tuple[datetime, float]]:
        result = []
        for s in self._buf:
            val = getattr(s, field_name, None)
            if val is not None:
                result.append((s.timestamp, float(val)))
        return result

    def min_max(self, field_name: str, minutes: int = 60,
                ) -> tuple[float, float] | None:
        pts = self.window(minutes)
        vals = [getattr(s, field_name, None) for s in pts]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return min(vals), max(vals)


# ---------------------------------------------------------------------------
# MomentumTracker
# ---------------------------------------------------------------------------

class MomentumTracker:
    """Compute velocity and acceleration from time-series data."""

    @staticmethod
    def compute(
        buffer: SnapshotBuffer,
        field_name: str,
        display_name: str,
        window_min: int = 30,
    ) -> MomentumState:
        series = buffer.series(field_name)
        if len(series) < 2:
            current = series[-1][1] if series else 0.0
            return MomentumState(
                name=display_name, current=current,
                velocity=0.0, acceleration=0.0, trend_duration_min=0.0,
            )

        current = series[-1][1]

        # Velocity: slope over window (units/hour)
        velocity = MomentumTracker._slope(series, window_min)

        # Acceleration: compare recent vs older velocity
        half = window_min // 2 or 5
        vel_recent = MomentumTracker._slope(series, half)
        vel_older = MomentumTracker._slope_offset(series, half, half)
        acceleration = vel_recent - vel_older

        # Trend duration: how long sign has been consistent
        trend_dur = MomentumTracker._trend_duration(series)

        is_accel = (velocity > 0 and acceleration > 0) or \
                   (velocity < 0 and acceleration < 0)
        is_decel = (velocity > 0 and acceleration < 0) or \
                   (velocity < 0 and acceleration > 0)
        # Inflection: velocity sign changed in last 2 points
        inflection = False
        if len(series) >= 3:
            v_prev = series[-2][1] - series[-3][1]
            v_curr = series[-1][1] - series[-2][1]
            if v_prev * v_curr < 0:
                inflection = True

        return MomentumState(
            name=display_name,
            current=current,
            velocity=round(velocity, 1),
            acceleration=round(acceleration, 1),
            trend_duration_min=round(trend_dur, 0),
            is_accelerating=is_accel,
            is_decelerating=is_decel,
            inflection_detected=inflection,
        )

    @staticmethod
    def _slope(series: list[tuple[datetime, float]], window_min: int) -> float:
        if len(series) < 2:
            return 0.0
        cutoff = series[-1][0] - timedelta(minutes=window_min)
        pts = [(t, v) for t, v in series if t >= cutoff]
        if len(pts) < 2:
            return 0.0
        t0 = pts[0][0]
        xs = [(p[0] - t0).total_seconds() / 3600.0 for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sxy - sx * sy) / denom

    @staticmethod
    def _slope_offset(
        series: list[tuple[datetime, float]],
        window_min: int,
        offset_min: int,
    ) -> float:
        if len(series) < 2:
            return 0.0
        end = series[-1][0] - timedelta(minutes=offset_min)
        start = end - timedelta(minutes=window_min)
        pts = [(t, v) for t, v in series if start <= t <= end]
        if len(pts) < 2:
            return 0.0
        t0 = pts[0][0]
        xs = [(p[0] - t0).total_seconds() / 3600.0 for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sxy - sx * sy) / denom

    @staticmethod
    def _trend_duration(series: list[tuple[datetime, float]]) -> float:
        if len(series) < 2:
            return 0.0
        current_sign = 1 if series[-1][1] >= series[-2][1] else -1
        for i in range(len(series) - 2, 0, -1):
            sign = 1 if series[i][1] >= series[i - 1][1] else -1
            if sign != current_sign:
                return (series[-1][0] - series[i][0]).total_seconds() / 60.0
        return (series[-1][0] - series[0][0]).total_seconds() / 60.0
# ---------------------------------------------------------------------------
# LeadingIndicatorLogic
# ---------------------------------------------------------------------------

class LeadingIndicatorLogic:
    """Detect leading indicator patterns from momentum data."""

    # Thresholds (tunable)
    PROGRAM_EXTREME = 10000.0     # 억원 — extreme program net
    FOREIGN_VELOCITY_SLOW = 1000.0  # 억원/hour — "slowing" threshold
    VKOSPI_SPIKE_RATE = 2.0       # points/hour — rapid expansion

    @staticmethod
    def evaluate(
        momentum: dict[str, MomentumState],
        buffer: SnapshotBuffer,
    ) -> list[LeadingSignal]:
        signals: list[LeadingSignal] = []

        foreign = momentum.get("foreign_net")
        program = momentum.get("program_total_net")
        program_na = momentum.get("program_nonarb_net")
        vkospi_m = momentum.get("vkospi")

        # 1. Program trading direction change (leads KOSPI ~15min)
        if program and program.inflection_detected:
            if program.velocity > 0:
                signals.append(LeadingSignal(
                    indicator="program_trading",
                    signal_type="reversal",
                    description_kr="프로그램 매수 전환 감지 → KOSPI 반등 선행 (10-30분)",
                    implication="LONG",
                    strength=0.7,
                ))
            else:
                signals.append(LeadingSignal(
                    indicator="program_trading",
                    signal_type="reversal",
                    description_kr="프로그램 매도 전환 감지 → KOSPI 하락 선행 (10-30분)",
                    implication="SHORT",
                    strength=0.7,
                ))

        # 2. Foreign flow deceleration (reversal precursor)
        if foreign and foreign.velocity < 0 and foreign.is_decelerating:
            strength = min(abs(foreign.acceleration) / 2000.0, 1.0)
            signals.append(LeadingSignal(
                indicator="foreign_flow",
                signal_type="reversal",
                description_kr=(
                    f"외국인 매도 속도 감소 중 "
                    f"({foreign.velocity:+,.0f}억/시간, 감속) → 매도 피로"
                ),
                implication="LONG",
                strength=min(0.3 + strength, 0.8),
            ))
        elif foreign and foreign.velocity > 0 and foreign.is_accelerating:
            signals.append(LeadingSignal(
                indicator="foreign_flow",
                signal_type="continuation",
                description_kr=(
                    f"외국인 매수 가속 중 "
                    f"({foreign.velocity:+,.0f}억/시간) → 상승 지속"
                ),
                implication="LONG",
                strength=0.6,
            ))
        elif foreign and foreign.velocity < 0 and foreign.is_accelerating:
            signals.append(LeadingSignal(
                indicator="foreign_flow",
                signal_type="continuation",
                description_kr=(
                    f"외국인 매도 가속 중 "
                    f"({foreign.velocity:+,.0f}억/시간) → 하락 지속"
                ),
                implication="SHORT",
                strength=0.6,
            ))

        # 3. Program non-arb extreme threshold
        if program_na and abs(program_na.current) > LeadingIndicatorLogic.PROGRAM_EXTREME:
            if program_na.current < 0:
                signals.append(LeadingSignal(
                    indicator="program_nonarb",
                    signal_type="threshold",
                    description_kr=(
                        f"비차익 매도 {abs(program_na.current):,.0f}억 "
                        f"(극단적 수준) → 추가 하락 또는 기술적 반등"
                    ),
                    implication="CAUTION",
                    strength=0.5,
                ))
            else:
                signals.append(LeadingSignal(
                    indicator="program_nonarb",
                    signal_type="threshold",
                    description_kr=(
                        f"비차익 매수 {program_na.current:,.0f}억 "
                        f"(극단적 수준) → 모멘텀 지속 가능"
                    ),
                    implication="LONG",
                    strength=0.5,
                ))

        # 4. KOSPI vs KOSDAQ divergence
        kospi_pts = buffer.series("kospi200_change_pct")
        kosdaq_pts = buffer.series("kosdaq_change_pct")
        if len(kospi_pts) >= 3 and len(kosdaq_pts) >= 3:
            k_dir = kospi_pts[-1][1] > kospi_pts[-3][1]
            q_dir = kosdaq_pts[-1][1] > kosdaq_pts[-3][1]
            if k_dir != q_dir:
                signals.append(LeadingSignal(
                    indicator="kospi_kosdaq",
                    signal_type="divergence",
                    description_kr="KOSPI↔KOSDAQ 방향 괴리 → 섹터 로테이션 진행 중",
                    implication="CAUTION",
                    strength=0.4,
                ))

        # 5. VKOSPI rapid expansion
        if vkospi_m and vkospi_m.velocity > LeadingIndicatorLogic.VKOSPI_SPIKE_RATE:
            signals.append(LeadingSignal(
                indicator="vkospi",
                signal_type="threshold",
                description_kr=(
                    f"VKOSPI 급등 중 ({vkospi_m.velocity:+.1f}pt/시간) "
                    f"→ 변동성 확대, 방향 전환 주의"
                ),
                implication="CAUTION",
                strength=0.6,
            ))

        return signals
# ---------------------------------------------------------------------------
# ScenarioGenerator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Generate forward-looking scenarios with conditional probabilities."""

    @staticmethod
    def generate(
        base_long_prob: float,
        momentum: dict[str, MomentumState],
        leading: list[LeadingSignal],
        buffer: SnapshotBuffer,
    ) -> list[Scenario]:
        # Adjust base probability with momentum & leading signals
        adj = 0.0
        for sig in leading:
            if sig.implication == "LONG":
                adj += sig.strength * 0.08
            elif sig.implication == "SHORT":
                adj -= sig.strength * 0.08

        # Momentum-based adjustment
        foreign = momentum.get("foreign_net")
        if foreign:
            if foreign.is_decelerating and foreign.velocity < 0:
                adj += 0.05  # selling fatigue → slight long bias
            elif foreign.is_accelerating and foreign.velocity < 0:
                adj -= 0.05  # selling pressure increasing

        adj_long = max(0.05, min(0.95, base_long_prob + adj))
        adj_short = 1.0 - adj_long

        scenarios = []
        latest = buffer.latest()

        # --- LONG scenario ---
        long_conditions = []
        long_risks = []
        long_entry = "KOSPI200 추가 상승 확인 시 롱 진입"

        if foreign and foreign.is_decelerating and foreign.velocity < 0:
            long_conditions.append(
                f"외국인 매도 감속 중 ({foreign.velocity:+,.0f}억/시간)"
            )
        if foreign and foreign.velocity > 0:
            long_conditions.append(
                f"외국인 매수 전환 ({foreign.velocity:+,.0f}억/시간)"
            )

        program = momentum.get("program_total_net")
        if program and program.inflection_detected and program.velocity > 0:
            long_conditions.append("프로그램 매수 전환 감지")
        elif program and program.velocity > 0:
            long_conditions.append(
                f"프로그램 매수 추세 ({program.velocity:+,.0f}억/시간)"
            )

        if not long_conditions:
            long_conditions.append("현재 숏 압력 완화 또는 반등 모멘텀 발생 시")

        # Entry trigger from support/resistance
        if latest and latest.kospi200:
            trigger_up = latest.kospi200 * 1.003
            long_entry = f"KOSPI200 {trigger_up:.1f} 돌파 시 롱 진입"

        long_risks.append("환율 재상승 시 롱 무효")
        if program and program.current < -5000:
            long_risks.append("프로그램 매도 규모 여전히 큼 — 반등 제한")

        # ETF recommendation
        long_etf, long_name = "122630", "KODEX 레버리지"
        for sig in leading:
            if "나스닥" in sig.description_kr or "반도체" in sig.description_kr:
                long_etf, long_name = "091170", "KODEX 반도체레버리지"
                break

        scenarios.append(Scenario(
            direction="LONG",
            probability=round(adj_long, 2),
            label_kr="반등 시나리오" if adj_long < 0.4 else "상승 시나리오",
            conditions_kr=long_conditions,
            entry_trigger_kr=long_entry,
            target_etf=long_etf,
            target_etf_name=long_name,
            risk_factors_kr=long_risks,
        ))

        # --- SHORT scenario ---
        short_conditions = []
        short_risks = []
        short_entry = "KOSPI200 추가 하락 확인 시 숏 진입"

        if foreign and foreign.velocity < 0 and foreign.is_accelerating:
            short_conditions.append(
                f"외국인 매도 가속 ({foreign.velocity:+,.0f}억/시간)"
            )
        elif foreign and foreign.velocity < 0:
            short_conditions.append(
                f"외국인 매도 지속 ({foreign.velocity:+,.0f}억/시간)"
            )

        program_na = momentum.get("program_nonarb_net")
        if program_na and program_na.velocity < 0:
            short_conditions.append(
                f"비차익 매도 확대 중 ({program_na.velocity:+,.0f}억/시간)"
            )

        if not short_conditions:
            short_conditions.append("추가 악재 또는 외국인 매도 재가속 시")

        if latest and latest.kospi200:
            trigger_dn = latest.kospi200 * 0.997
            short_entry = f"KOSPI200 {trigger_dn:.1f} 이탈 시 숏 진입"

        short_risks.append("프로그램 매수 전환 시 급반등 가능")
        if foreign and foreign.is_decelerating:
            short_risks.append("외국인 매도 감속 — 숏 타이밍 늦을 수 있음")

        scenarios.append(Scenario(
            direction="SHORT",
            probability=round(adj_short, 2),
            label_kr="하락 시나리오" if adj_short > 0.6 else "조정 시나리오",
            conditions_kr=short_conditions,
            entry_trigger_kr=short_entry,
            target_etf="252670",
            target_etf_name="KODEX 인버스2X",
            risk_factors_kr=short_risks,
        ))

        # --- NEUTRAL scenario (if close to 50/50) ---
        if abs(adj_long - 0.5) < 0.15:
            scenarios.append(Scenario(
                direction="NEUTRAL",
                probability=round(1.0 - adj_long - adj_short, 2)
                if adj_long + adj_short < 1.0 else 0.0,
                label_kr="횡보/관망",
                conditions_kr=["방향성 불확실, 시그널 혼조"],
                entry_trigger_kr="명확한 방향 확인 전까지 대기",
                target_etf="",
                target_etf_name="관망",
                risk_factors_kr=["갑작스런 방향 전환 가능"],
            ))

        scenarios.sort(key=lambda s: s.probability, reverse=True)
        return scenarios
# ---------------------------------------------------------------------------
# TriggerEngine
# ---------------------------------------------------------------------------

class TriggerEngine:
    """Compute specific actionable entry triggers from current state.

    Triggers are concrete price/flow levels that, if reached, confirm
    a directional scenario. They give the trader exact "buy when X" rules.
    """

    @staticmethod
    def compute(
        buffer: SnapshotBuffer,
        momentum: dict[str, MomentumState],
        leading: list[LeadingSignal],
    ) -> list[EntryTrigger]:
        triggers: list[EntryTrigger] = []
        latest = buffer.latest()
        if not latest:
            return triggers

        # 1. KOSPI200 support/resistance breakout triggers
        minmax = buffer.min_max("kospi200", minutes=60)
        if minmax and latest.kospi200:
            lo, hi = minmax
            current = latest.kospi200
            range_size = hi - lo if hi > lo else current * 0.003

            # Breakout above intraday high → LONG trigger
            if hi > lo:
                resistance = hi + range_size * 0.1
                dist = (resistance - current) / current * 100 if current else 0
                triggers.append(EntryTrigger(
                    direction="LONG",
                    condition_kr=f"KOSPI200 {resistance:.1f} 돌파 시 (1시간 고점+α)",
                    indicator="kospi200_resistance",
                    current_value=round(current, 1),
                    threshold_value=round(resistance, 1),
                    distance_pct=round(dist, 2),
                    target_etf="122630",
                ))

            # Breakdown below intraday low → SHORT trigger
            if hi > lo:
                support = lo - range_size * 0.1
                dist = (current - support) / current * 100 if current else 0
                triggers.append(EntryTrigger(
                    direction="SHORT",
                    condition_kr=f"KOSPI200 {support:.1f} 이탈 시 (1시간 저점-α)",
                    indicator="kospi200_support",
                    current_value=round(current, 1),
                    threshold_value=round(support, 1),
                    distance_pct=round(dist, 2),
                    target_etf="252670",
                ))

        # 2. Foreign flow velocity trigger
        foreign = momentum.get("foreign_net")
        if foreign:
            # Foreign buying acceleration → LONG
            if foreign.velocity > 0 and foreign.is_accelerating:
                triggers.append(EntryTrigger(
                    direction="LONG",
                    condition_kr=(
                        f"외국인 매수 가속 확인 "
                        f"(현재 {foreign.velocity:+,.0f}억/시간, 가속 중)"
                    ),
                    indicator="foreign_velocity",
                    current_value=round(foreign.velocity, 0),
                    threshold_value=0.0,
                    distance_pct=0.0,
                    target_etf="122630",
                ))
            # Foreign selling acceleration → SHORT
            elif foreign.velocity < 0 and foreign.is_accelerating:
                triggers.append(EntryTrigger(
                    direction="SHORT",
                    condition_kr=(
                        f"외국인 매도 가속 확인 "
                        f"(현재 {foreign.velocity:+,.0f}억/시간, 가속 중)"
                    ),
                    indicator="foreign_velocity",
                    current_value=round(foreign.velocity, 0),
                    threshold_value=0.0,
                    distance_pct=0.0,
                    target_etf="252670",
                ))
            # Foreign selling deceleration → reversal watch
            elif foreign.velocity < 0 and foreign.is_decelerating:
                triggers.append(EntryTrigger(
                    direction="LONG",
                    condition_kr=(
                        f"외국인 매도 감속 → 순매수 전환 시 "
                        f"(현재 {foreign.velocity:+,.0f}억/시간)"
                    ),
                    indicator="foreign_reversal",
                    current_value=round(foreign.velocity, 0),
                    threshold_value=0.0,
                    distance_pct=abs(foreign.velocity) / max(
                        abs(foreign.velocity) + 500, 1
                    ) * 100,
                    target_etf="122630",
                ))

        # 3. Program trading sign change trigger
        program = momentum.get("program_total_net")
        if program:
            if program.inflection_detected:
                direction = "LONG" if program.velocity > 0 else "SHORT"
                etf = "122630" if direction == "LONG" else "252670"
                label = "매수 전환" if direction == "LONG" else "매도 전환"
                triggers.append(EntryTrigger(
                    direction=direction,
                    condition_kr=(
                        f"프로그램 {label} 감지 — "
                        f"이미 진행 중 (속도 {program.velocity:+,.0f}억/시간)"
                    ),
                    indicator="program_inflection",
                    current_value=round(program.current, 0),
                    threshold_value=0.0,
                    distance_pct=0.0,
                    target_etf=etf,
                ))

        # 4. VKOSPI spike → hedge or caution
        vkospi_m = momentum.get("vkospi")
        if vkospi_m and vkospi_m.velocity > 1.5:
            triggers.append(EntryTrigger(
                direction="SHORT",
                condition_kr=(
                    f"VKOSPI 급등 중 ({vkospi_m.velocity:+.1f}pt/시간) "
                    f"— 변동성 확대, 인버스 헤지"
                ),
                indicator="vkospi_spike",
                current_value=round(vkospi_m.current, 1),
                threshold_value=round(vkospi_m.current + 2.0, 1),
                distance_pct=round(
                    2.0 / max(vkospi_m.current, 1) * 100, 1
                ),
                target_etf="252670",
            ))

        # Sort: 0 distance (already triggered) first, then by distance
        triggers.sort(key=lambda t: t.distance_pct)
        return triggers


# ---------------------------------------------------------------------------
# PredictiveEngine — top-level orchestrator
# ---------------------------------------------------------------------------

class PredictiveEngine:
    """Top-level predictive engine that ties all components together.

    Usage::

        engine = PredictiveEngine()
        result = engine.update(domestic_snapshot, signal_result)
        # result.scenarios[0] = highest probability scenario
        # result.triggers = concrete entry conditions
        # result.leading_signals = detected leading patterns
    """

    # Minimum snapshots before producing meaningful predictions
    MIN_WARMUP = 3

    # Which DomesticSnapshot fields to track for momentum
    _MOMENTUM_FIELDS = [
        ("foreign_net", "외국인 순매수"),
        ("program_total_net", "프로그램 순매수"),
        ("program_arb_net", "차익 순매수"),
        ("program_nonarb_net", "비차익 순매수"),
        ("vkospi", "VKOSPI"),
        ("kospi200_change_pct", "KOSPI200 등락률"),
    ]

    def __init__(self, buffer_size: int = 60) -> None:
        self.buffer = SnapshotBuffer(maxlen=buffer_size)
        self._last_update: datetime | None = None

    def update(
        self,
        domestic: Any,
        signal: Any,
    ) -> PredictiveResult:
        """Push new data and generate predictive result.

        Args:
            domestic: DomesticSnapshot from collectors.domestic
            signal: SignalResult from engine.signal_pipeline
        """
        # --- 1. Convert to TimedSnapshot & push ---
        snap = self._to_snapshot(domestic, signal)
        self.buffer.push(snap)
        self._last_update = snap.timestamp

        warmup = len(self.buffer) >= self.MIN_WARMUP

        # --- 2. Compute momentum for each tracked indicator ---
        momentum: dict[str, MomentumState] = {}
        for field_name, display in self._MOMENTUM_FIELDS:
            momentum[field_name] = MomentumTracker.compute(
                self.buffer, field_name, display,
            )

        # --- 3. Detect leading signals ---
        leading = LeadingIndicatorLogic.evaluate(momentum, self.buffer)

        # --- 4. Generate scenarios ---
        base_p = snap.long_probability
        scenarios = ScenarioGenerator.generate(
            base_p, momentum, leading, self.buffer,
        )

        # --- 5. Compute entry triggers ---
        triggers = TriggerEngine.compute(self.buffer, momentum, leading)

        # --- 6. Log summary ---
        if warmup:
            top = scenarios[0] if scenarios else None
            n_triggers = len(triggers)
            n_leading = len(leading)
            logger.info(
                "Predictive: top=%s(%.0f%%), %d triggers, %d leading, "
                "depth=%d",
                top.direction if top else "?",
                (top.probability * 100) if top else 0,
                n_triggers,
                n_leading,
                len(self.buffer),
            )

        return PredictiveResult(
            scenarios=scenarios,
            momentum=momentum,
            leading_signals=leading,
            triggers=triggers,
            buffer_depth=len(self.buffer),
            warmup_complete=warmup,
        )

    def _to_snapshot(self, domestic: Any, signal: Any) -> TimedSnapshot:
        """Extract a TimedSnapshot from domestic + signal objects."""
        now = datetime.now(KST)

        # Defaults
        snap = TimedSnapshot(timestamp=now)

        # From domestic snapshot
        if domestic is not None:
            snap.kospi200 = getattr(domestic, "kospi200_current", None)
            snap.kospi200_change_pct = getattr(
                domestic, "kospi200_change_pct", None,
            )
            snap.kosdaq_current = getattr(domestic, "kosdaq_current", None)
            snap.kosdaq_change_pct = getattr(
                domestic, "kosdaq_change_pct", None,
            )
            snap.foreign_net = getattr(domestic, "foreign_net", 0.0)
            snap.institution_net = getattr(domestic, "institution_net", 0.0)
            snap.program_total_net = getattr(
                domestic, "program_total_net", 0.0,
            )
            snap.program_arb_net = getattr(domestic, "program_arb_net", 0.0)
            snap.program_nonarb_net = getattr(
                domestic, "program_nonarb_net", 0.0,
            )
            snap.vkospi = getattr(domestic, "vkospi", 20.0)

        # From signal result
        if signal is not None:
            snap.long_probability = getattr(signal, "long_probability", 0.5)

        return snap
