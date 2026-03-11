"""Historical pattern matching for conditional probability analysis.

Given current market conditions (indicator changes from previous close),
finds similar historical days and computes conditional probabilities
for next-day KOSPI direction.

Example output:
  "환율↑ + S&P↓" → KOSPI 하락 73% (47건, avg -0.82%)
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """A single market condition to match against history."""
    name: str           # internal key
    indicator: str      # column name in data
    check: callable     # fn(return_pct) -> bool
    label_kr: str       # Korean description

    def __hash__(self):
        return hash(self.name)


# Standard conditions for KOSPI prediction
STANDARD_CONDITIONS = [
    Condition("fx_up_strong",  "usd_krw", lambda x: x > 0.005, "환율 강한 상승(+0.5%↑)"),
    Condition("fx_up",         "usd_krw", lambda x: 0.002 < x <= 0.005, "환율 상승(+0.2~0.5%)"),
    Condition("fx_down",       "usd_krw", lambda x: x < -0.002, "환율 하락(-0.2%↓)"),
    Condition("sp_up_strong",  "sp500",   lambda x: x > 0.01, "S&P500 강한 상승(+1%↑)"),
    Condition("sp_up",         "sp500",   lambda x: 0.003 < x <= 0.01, "S&P500 상승(+0.3~1%)"),
    Condition("sp_down",       "sp500",   lambda x: x < -0.003, "S&P500 하락(-0.3%↓)"),
    Condition("sp_down_strong","sp500",   lambda x: x < -0.01, "S&P500 강한 하락(-1%↓)"),
    Condition("nq_up",         "nasdaq",  lambda x: x > 0.005, "NASDAQ 상승(+0.5%↑)"),
    Condition("nq_down",       "nasdaq",  lambda x: x < -0.005, "NASDAQ 하락(-0.5%↓)"),
    Condition("vix_spike",     "vix",     lambda x: x > 0.08, "VIX 급등(+8%↑)"),
    Condition("vix_high",      "vix",     lambda x: x > 0.03, "VIX 상승(+3%↑)"),
    Condition("vix_drop",      "vix",     lambda x: x < -0.05, "VIX 하락(-5%↓)"),
    Condition("oil_up",        "wti",     lambda x: x > 0.02, "유가 상승(+2%↑)"),
    Condition("oil_down",      "wti",     lambda x: x < -0.02, "유가 하락(-2%↓)"),
    Condition("dxy_up",        "dxy",     lambda x: x > 0.003, "달러지수 상승(+0.3%↑)"),
    Condition("dxy_down",      "dxy",     lambda x: x < -0.003, "달러지수 하락(-0.3%↓)"),
]


@dataclass
class PatternMatch:
    """Result of a pattern match against history."""
    conditions: str           # e.g. "환율↑ + S&P↓"
    sample_size: int
    up_probability: float     # P(KOSPI up next day)
    down_probability: float
    avg_return: float         # average next-day KOSPI return
    median_return: float
    max_drawdown: float       # worst case in sample
    best_gain: float          # best case in sample
    direction: str            # "상승" / "하락" / "보합"
    confidence: str           # "높음" / "보통" / "낮음" based on sample size


class PatternMatcher:
    """Historical pattern analyzer using yfinance daily data.

    Downloads and caches 2 years of daily returns for key indicators.
    When queried with current conditions, finds matching historical days
    and computes conditional probabilities for next-day KOSPI return.
    """

    TICKERS = {
        "kospi":   "^KS11",
        "usd_krw": "KRW=X",
        "sp500":   "^GSPC",
        "nasdaq":  "^IXIC",
        "vix":     "^VIX",
        "wti":     "CL=F",
        "dxy":     "DX-Y.NYB",
    }

    def __init__(self, lookback_years: int = 2, cache_hours: int = 6):
        self._lookback_years = lookback_years
        self._cache_hours = cache_hours
        self._data: pd.DataFrame | None = None
        self._last_fetch: float = 0

    def _ensure_data(self) -> bool:
        """Download and cache historical data. Returns True if data available."""
        now = _time.time()
        if self._data is not None and (now - self._last_fetch) < self._cache_hours * 3600:
            return True

        try:
            import yfinance as yf

            period = f"{self._lookback_years}y"
            dfs = {}

            for name, sym in self.TICKERS.items():
                try:
                    data = yf.download(sym, period=period, progress=False, timeout=15)
                    if data is not None and not data.empty:
                        close_col = "Close" if "Close" in data.columns else "close"
                        # Handle MultiIndex columns from yf.download
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                            close_col = "Close" if "Close" in data.columns else "close"
                        dfs[name] = data[close_col].pct_change()
                except Exception as e:
                    logger.debug(f"PatternMatcher: {sym} download failed: {e}")

            if "kospi" not in dfs:
                logger.warning("PatternMatcher: KOSPI data not available")
                return False

            self._data = pd.DataFrame(dfs).dropna()
            self._last_fetch = now
            logger.info(f"PatternMatcher: loaded {len(self._data)} days of history")
            return len(self._data) > 30

        except Exception as e:
            logger.error(f"PatternMatcher data download failed: {e}")
            return False

    def find_patterns(self, current_conditions: dict[str, float]) -> list[PatternMatch]:
        """Find historical patterns matching current market conditions.

        Args:
            current_conditions: {indicator: return_pct}
                e.g. {"usd_krw": 0.005, "sp500": -0.008, "vix": 0.03}

        Returns:
            List of PatternMatch, sorted by confidence (sample_size).
        """
        if not self._ensure_data():
            return []

        df = self._data
        # Next-day KOSPI return (what we want to predict)
        kospi_next = df["kospi"].shift(-1)

        # Find which standard conditions are active NOW
        active: list[Condition] = []
        for cond in STANDARD_CONDITIONS:
            if cond.indicator in current_conditions:
                val = current_conditions[cond.indicator]
                try:
                    if cond.check(val):
                        active.append(cond)
                except Exception:
                    pass

        if not active:
            return []

        results: list[PatternMatch] = []

        # 1. Combined pattern (all active conditions)
        if len(active) >= 2:
            combined_mask = pd.Series(True, index=df.index)
            labels = []
            for cond in active:
                if cond.indicator in df.columns:
                    col_mask = df[cond.indicator].apply(
                        lambda x, c=cond: c.check(x) if pd.notna(x) else False
                    )
                    combined_mask &= col_mask
                    labels.append(cond.label_kr)

            matching_rets = kospi_next[combined_mask].dropna()
            if len(matching_rets) >= 5:
                results.append(self._build_match(
                    " + ".join(labels), matching_rets
                ))

        # 2. Individual conditions
        for cond in active:
            if cond.indicator not in df.columns:
                continue
            single_mask = df[cond.indicator].apply(
                lambda x, c=cond: c.check(x) if pd.notna(x) else False
            )
            matching_rets = kospi_next[single_mask].dropna()
            if len(matching_rets) >= 10:
                results.append(self._build_match(cond.label_kr, matching_rets))

        # 3. Special composite patterns
        self._add_composite_patterns(df, kospi_next, current_conditions, results)

        # Sort by sample size (most reliable first), then by how extreme the probability is
        results.sort(key=lambda p: (
            -abs(p.up_probability - 0.5) * min(p.sample_size, 100),
        ))

        return results[:8]  # top 8 patterns

    def _add_composite_patterns(
        self, df: pd.DataFrame, kospi_next: pd.Series,
        conditions: dict, results: list[PatternMatch]
    ):
        """Add predefined composite patterns that are known to be predictive."""
        composites = [
            {
                "name": "외국인 매도 패턴 (환율↑ + S&P↓)",
                "conditions": [
                    ("usd_krw", lambda x: x > 0.002),
                    ("sp500", lambda x: x < -0.003),
                ],
                "active_check": lambda c: c.get("usd_krw", 0) > 0.002 and c.get("sp500", 0) < -0.003,
            },
            {
                "name": "리스크오프 (VIX↑ + S&P↓ + 환율↑)",
                "conditions": [
                    ("vix", lambda x: x > 0.03),
                    ("sp500", lambda x: x < -0.003),
                    ("usd_krw", lambda x: x > 0.001),
                ],
                "active_check": lambda c: (
                    c.get("vix", 0) > 0.03 and c.get("sp500", 0) < -0.003
                    and c.get("usd_krw", 0) > 0.001
                ),
            },
            {
                "name": "리스크온 (VIX↓ + S&P↑ + 환율↓)",
                "conditions": [
                    ("vix", lambda x: x < -0.03),
                    ("sp500", lambda x: x > 0.003),
                    ("usd_krw", lambda x: x < -0.001),
                ],
                "active_check": lambda c: (
                    c.get("vix", 0) < -0.03 and c.get("sp500", 0) > 0.003
                    and c.get("usd_krw", 0) < -0.001
                ),
            },
            {
                "name": "유가 충격 (유가±3% + 환율 변동)",
                "conditions": [
                    ("wti", lambda x: abs(x) > 0.03),
                    ("usd_krw", lambda x: abs(x) > 0.002),
                ],
                "active_check": lambda c: abs(c.get("wti", 0)) > 0.03 and abs(c.get("usd_krw", 0)) > 0.002,
            },
        ]

        for comp in composites:
            if not comp["active_check"](conditions):
                continue

            mask = pd.Series(True, index=df.index)
            for col, fn in comp["conditions"]:
                if col in df.columns:
                    mask &= df[col].apply(lambda x, f=fn: f(x) if pd.notna(x) else False)

            rets = kospi_next[mask].dropna()
            if len(rets) >= 5:
                results.append(self._build_match(comp["name"], rets))

    @staticmethod
    def _build_match(label: str, rets: pd.Series) -> PatternMatch:
        n = len(rets)
        up_pct = float((rets > 0).mean())
        down_pct = float((rets < 0).mean())
        avg = float(rets.mean())
        med = float(rets.median())

        if up_pct > 0.58:
            direction = "상승"
        elif down_pct > 0.58:
            direction = "하락"
        else:
            direction = "보합"

        if n >= 40:
            conf = "높음"
        elif n >= 20:
            conf = "보통"
        else:
            conf = "낮음"

        return PatternMatch(
            conditions=label,
            sample_size=n,
            up_probability=up_pct,
            down_probability=down_pct,
            avg_return=avg,
            median_return=med,
            max_drawdown=float(rets.min()),
            best_gain=float(rets.max()),
            direction=direction,
            confidence=conf,
        )

    def format_patterns(self, patterns: list[PatternMatch]) -> str:
        """Format pattern matches as Korean text for Discord."""
        if not patterns:
            return "매칭되는 역사적 패턴 없음"

        lines = []
        for i, p in enumerate(patterns[:5]):
            emoji = {"상승": "📈", "하락": "📉", "보합": "➡️"}[p.direction]
            prob = max(p.up_probability, p.down_probability)
            lines.append(
                f"{emoji} {p.conditions}\n"
                f"  → KOSPI {p.direction} {prob:.0%} "
                f"({p.sample_size}건, 평균 {p.avg_return:+.2%}, "
                f"최대 {p.best_gain:+.2%} / 최소 {p.max_drawdown:+.2%}) "
                f"[{p.confidence}]"
            )

        return "\n".join(lines)
