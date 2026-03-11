"""Bayesian Log-Odds Pooling engine for KOSPI ETF leverage trading.

Implements the signal fusion formula agreed upon by all 5 Spark Panel agents:

    P(LONG) = sigmoid( prior_lo + sum( w_i * f_i * lo_i ) )

Where:
    lo_i = log(p_i / (1 - p_i))          -- log-odds for each signal
    w_i  = regime-dependent weight         -- adjusted by RegimeDetector
    f_i  = exp(-age_i / tau_i)            -- time-decay factor

This module also contains ``RegimeDetector`` which classifies the current
market state into one of four regimes using VKOSPI level and KOSPI200 trend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Sequence

import numpy as np
from scipy.special import expit as scipy_sigmoid  # numerically stable sigmoid

from kospi_corr.domain.types import SignalDirection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = 1e-8
_CLIP_LO = 1e-6   # clamp probabilities away from 0 and 1 for log-odds
_CLIP_HI = 1 - _EPS


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

class Regime(StrEnum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"


# Default regime weight multipliers.
# Keys are indicator *categories* (matching IndicatorCategory in types.py)
# plus special keys for signal groups.
# Values are multipliers applied ON TOP of the base weight.
_DEFAULT_REGIME_WEIGHTS: dict[Regime, dict[str, float]] = {
    Regime.TRENDING_UP: {
        "direction_driver": 1.2,     # overnight futures, S&P, WTI, FX
        "flow": 1.4,                 # foreign futures net weight UP
        "sector": 1.0,
        "regime_gate": 0.8,
        "sentiment": 0.6,
    },
    Regime.TRENDING_DOWN: {
        "direction_driver": 1.2,
        "flow": 1.6,                 # foreign futures even more important
        "sector": 1.0,
        "regime_gate": 1.0,
        "sentiment": 1.2,            # fear signals matter more
    },
    Regime.RANGE_BOUND: {
        "direction_driver": 0.8,
        "flow": 1.0,
        "sector": 1.2,
        "regime_gate": 1.0,
        "sentiment": 1.0,
    },
    Regime.VOLATILE: {
        "direction_driver": 0.7,
        "flow": 0.9,
        "sector": 0.8,
        "regime_gate": 1.4,          # VKOSPI / vol signals dominate
        "sentiment": 1.3,
    },
}


@dataclass(frozen=True)
class RegimeSnapshot:
    """Result of regime classification."""
    regime: Regime
    vkospi_level: float
    trend_slope: float   # positive = up, negative = down
    confidence: float    # [0, 1] how clear the classification is


class RegimeDetector:
    """Classify market regime from VKOSPI and KOSPI200 trend direction.

    Thresholds (all tuneable):
        vkospi_high:  VKOSPI above this -> VOLATILE regardless of trend.
        vkospi_low:   VKOSPI below this -> trend regime is trusted.
        trend_thresh: Absolute slope below this -> RANGE_BOUND.

    The ``trend_slope`` is typically computed externally as the OLS slope of
    log(KOSPI200) over a trailing window (e.g. 20 trading days), or simply
    the 20-day return divided by window length.

    Parameters
    ----------
    vkospi_high : float
        VKOSPI level above which regime is forced to VOLATILE.
    vkospi_low : float
        VKOSPI level below which we trust the trend classification.
    trend_threshold : float
        Absolute trend slope below which we classify RANGE_BOUND.
    regime_weights : dict, optional
        Override default regime->category weight multipliers.
    """

    def __init__(
        self,
        vkospi_high: float = 25.0,
        vkospi_low: float = 15.0,
        trend_threshold: float = 0.001,
        regime_weights: dict[Regime, dict[str, float]] | None = None,
    ) -> None:
        self.vkospi_high = vkospi_high
        self.vkospi_low = vkospi_low
        self.trend_threshold = trend_threshold
        self.regime_weights = regime_weights or dict(_DEFAULT_REGIME_WEIGHTS)

    def detect(
        self,
        vkospi: float,
        trend_slope: float,
    ) -> RegimeSnapshot:
        """Classify the current regime.

        Parameters
        ----------
        vkospi : float
            Current VKOSPI level (e.g. 18.5).
        trend_slope : float
            Slope of KOSPI200 trend (positive = up).

        Returns
        -------
        RegimeSnapshot
        """
        # Step 1: volatility override
        if vkospi >= self.vkospi_high:
            confidence = min(1.0, (vkospi - self.vkospi_high) / 10.0 + 0.6)
            return RegimeSnapshot(
                regime=Regime.VOLATILE,
                vkospi_level=vkospi,
                trend_slope=trend_slope,
                confidence=confidence,
            )

        # Step 2: trend vs range
        abs_slope = abs(trend_slope)
        if abs_slope < self.trend_threshold:
            # Confidence increases when slope is very flat
            confidence = min(1.0, 1.0 - abs_slope / self.trend_threshold)
            return RegimeSnapshot(
                regime=Regime.RANGE_BOUND,
                vkospi_level=vkospi,
                trend_slope=trend_slope,
                confidence=max(0.3, confidence),
            )

        # Step 3: trending up or down
        regime = Regime.TRENDING_UP if trend_slope > 0 else Regime.TRENDING_DOWN

        # Confidence: higher slope + lower VKOSPI -> more confident trend
        slope_conf = min(1.0, abs_slope / (self.trend_threshold * 5))
        vol_penalty = max(0.0, (vkospi - self.vkospi_low) / (self.vkospi_high - self.vkospi_low))
        confidence = slope_conf * (1.0 - 0.3 * vol_penalty)

        return RegimeSnapshot(
            regime=regime,
            vkospi_level=vkospi,
            trend_slope=trend_slope,
            confidence=max(0.2, min(1.0, confidence)),
        )

    def get_weight_multiplier(self, regime: Regime, signal_category: str) -> float:
        """Return the weight multiplier for a signal category under a regime."""
        regime_map = self.regime_weights.get(regime, {})
        return regime_map.get(signal_category, 1.0)


# ---------------------------------------------------------------------------
# Bayesian signal input
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BayesianSignal:
    """One signal observation for the Bayesian engine.

    Attributes:
        name:          Signal identifier (e.g. ``sp500_overnight``).
        probability:   The signal's individual P(LONG) estimate, in (0, 1).
        weight:        Base weight (before regime adjustment).
        category:      Signal category for regime weight lookup
                       (e.g. ``direction_driver``, ``flow``, ``sentiment``).
        age_seconds:   How old this observation is (0 = fresh).
        tau_seconds:   Half-life for time-decay.  Default 6 hours (21600s).
                       After tau seconds the signal contribution is ~37% of
                       its original strength.
    """
    name: str
    probability: float
    weight: float = 1.0
    category: str = "direction_driver"
    age_seconds: float = 0.0
    tau_seconds: float = 21600.0  # 6 hours default


@dataclass
class BayesianResult:
    """Output of BayesianEngine.compute()."""
    long_probability: float
    short_probability: float
    confidence: float
    regime: RegimeSnapshot
    direction: SignalDirection
    prior_log_odds: float
    posterior_log_odds: float
    key_signals: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bayesian Engine
# ---------------------------------------------------------------------------

class BayesianEngine:
    """Bayesian Log-Odds Pooling for KOSPI directional signals.

    Implements::

        P(LONG) = sigmoid( prior_lo + sum_i( w_i * f_i * lo_i ) )

    Where ``lo_i = logit(p_i)`` and ``f_i = exp(-age_i / tau_i)``.

    Parameters
    ----------
    prior : float
        Prior P(LONG).  Default 0.5 (uninformative).
    regime_detector : RegimeDetector, optional
        If not provided, a default detector is instantiated.
    neutral_band : float
        Half-width of the NEUTRAL probability zone around 0.5.
    max_log_odds : float
        Clamp individual log-odds to [-max, +max] to prevent a single
        extreme signal from dominating.
    """

    def __init__(
        self,
        prior: float = 0.5,
        regime_detector: RegimeDetector | None = None,
        neutral_band: float = 0.05,
        max_log_odds: float = 5.0,
    ) -> None:
        self.prior = np.clip(prior, _CLIP_LO, _CLIP_HI)
        self.prior_log_odds = float(self._logit(self.prior))
        self.regime_detector = regime_detector or RegimeDetector()
        self.neutral_band = neutral_band
        self.max_log_odds = max_log_odds

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _logit(p: float | np.ndarray) -> float | np.ndarray:
        """Log-odds: log(p / (1-p)), numerically safe."""
        p = np.clip(p, _CLIP_LO, _CLIP_HI)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        """Sigmoid via scipy (handles overflow gracefully)."""
        return scipy_sigmoid(x)

    @staticmethod
    def _time_decay(age_seconds: float, tau_seconds: float) -> float:
        """Exponential time decay factor f = exp(-age / tau).

        Returns 1.0 for fresh data (age=0), decays toward 0.
        """
        if tau_seconds <= 0:
            return 1.0
        return float(np.exp(-age_seconds / tau_seconds))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        signals: Sequence[BayesianSignal],
        vkospi: float = 18.0,
        trend_slope: float = 0.0,
    ) -> BayesianResult:
        """Run Bayesian log-odds pooling.

        Parameters
        ----------
        signals : sequence of BayesianSignal
            Individual signal probability estimates.
        vkospi : float
            Current VKOSPI level for regime detection.
        trend_slope : float
            KOSPI200 trend slope for regime detection.

        Returns
        -------
        BayesianResult
        """
        # 1. Detect regime
        regime_snap = self.regime_detector.detect(vkospi, trend_slope)

        # 2. Accumulate log-odds
        accumulated_lo = 0.0
        key_signals: list[dict[str, Any]] = []
        valid_count = 0

        for sig in signals:
            # Skip invalid probabilities
            if np.isnan(sig.probability) or sig.probability <= 0 or sig.probability >= 1:
                logger.debug(
                    "Skipping invalid signal %s (p=%.4f)", sig.name, sig.probability
                )
                continue

            # a) log-odds for this signal
            lo_i = float(self._logit(sig.probability))
            lo_i = float(np.clip(lo_i, -self.max_log_odds, self.max_log_odds))

            # b) regime-dependent weight multiplier
            regime_mult = self.regime_detector.get_weight_multiplier(
                regime_snap.regime, sig.category
            )
            w_i = sig.weight * regime_mult

            # c) time-decay factor
            f_i = self._time_decay(sig.age_seconds, sig.tau_seconds)

            # d) contribution
            contribution = w_i * f_i * lo_i
            accumulated_lo += contribution
            valid_count += 1

            key_signals.append({
                "name": sig.name,
                "probability": round(sig.probability, 4),
                "log_odds": round(lo_i, 4),
                "weight": round(w_i, 4),
                "decay": round(f_i, 4),
                "contribution": round(contribution, 4),
                "category": sig.category,
            })

        # 3. Posterior log-odds
        posterior_lo = self.prior_log_odds + accumulated_lo

        # 4. Map to probability
        long_prob = float(self._sigmoid(posterior_lo))
        short_prob = 1.0 - long_prob

        # 5. Confidence estimation (simple heuristic here; ConfidenceCalculator
        #    provides a more thorough version).
        confidence = self._estimate_confidence(
            long_prob, valid_count, regime_snap, key_signals
        )

        # 6. Direction classification
        if long_prob > 0.5 + self.neutral_band:
            direction = SignalDirection.LONG
        elif long_prob < 0.5 - self.neutral_band:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Sort key_signals by absolute contribution (most influential first).
        key_signals.sort(key=lambda s: abs(s["contribution"]), reverse=True)

        return BayesianResult(
            long_probability=round(long_prob, 6),
            short_probability=round(short_prob, 6),
            confidence=round(confidence, 4),
            regime=regime_snap,
            direction=direction,
            prior_log_odds=round(self.prior_log_odds, 6),
            posterior_log_odds=round(posterior_lo, 6),
            key_signals=key_signals,
        )

    # ------------------------------------------------------------------
    # Internal confidence heuristic
    # ------------------------------------------------------------------

    def _estimate_confidence(
        self,
        long_prob: float,
        n_signals: int,
        regime: RegimeSnapshot,
        key_signals: list[dict[str, Any]],
    ) -> float:
        """Quick confidence estimate (more sophisticated version in confidence.py).

        Factors:
            - Distance from 0.5 (further = more confident)
            - Number of valid signals (more = better)
            - Signal agreement (all same sign = confident)
            - Regime clarity
        """
        # a) Probability distance from neutral
        prob_dist = abs(long_prob - 0.5) * 2.0  # [0, 1]

        # b) Signal count factor  (diminishing returns above ~5 signals)
        count_factor = min(1.0, n_signals / 5.0) if n_signals > 0 else 0.0

        # c) Agreement: fraction of signals pointing in the majority direction
        if key_signals:
            positive = sum(1 for s in key_signals if s["contribution"] > 0)
            negative = len(key_signals) - positive
            majority = max(positive, negative)
            agreement = majority / len(key_signals)
        else:
            agreement = 0.0

        # d) Regime confidence
        regime_conf = regime.confidence

        # Weighted combination
        confidence = (
            0.35 * prob_dist
            + 0.25 * count_factor
            + 0.25 * agreement
            + 0.15 * regime_conf
        )

        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_from_probabilities(
        self,
        signal_probs: dict[str, float],
        weights: dict[str, float] | None = None,
        categories: dict[str, str] | None = None,
        ages: dict[str, float] | None = None,
        taus: dict[str, float] | None = None,
        vkospi: float = 18.0,
        trend_slope: float = 0.0,
    ) -> BayesianResult:
        """Build BayesianSignals from plain dicts and run compute().

        Parameters
        ----------
        signal_probs : dict
            ``{signal_name: p_long}`` where 0 < p_long < 1.
        weights, categories, ages, taus : dict, optional
            Per-signal overrides; missing keys use defaults.
        vkospi, trend_slope : float
            For regime detection.
        """
        weights = weights or {}
        categories = categories or {}
        ages = ages or {}
        taus = taus or {}

        signals: list[BayesianSignal] = []
        for name, prob in signal_probs.items():
            signals.append(BayesianSignal(
                name=name,
                probability=prob,
                weight=weights.get(name, 1.0),
                category=categories.get(name, "direction_driver"),
                age_seconds=ages.get(name, 0.0),
                tau_seconds=taus.get(name, 21600.0),
            ))

        return self.compute(signals, vkospi=vkospi, trend_slope=trend_slope)
