"""Confidence estimation for the KOSPI signal engine.

Provides a thorough confidence score that goes beyond the quick heuristic
inside BayesianEngine.  The ``ConfidenceCalculator`` considers:

    1. **Signal agreement** -- how many signals agree on direction.
    2. **Signal freshness** -- average age / staleness of the data.
    3. **Regime stability** -- whether the regime has been stable recently.
    4. **Signal diversity** -- are agreeing signals from different categories?
    5. **Probability extremity** -- how far the posterior is from 0.5.

The output is a single float in [0, 1] plus a breakdown dict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from kospi_corr.domain.types import SignalDirection
from kospi_corr.engine.bayesian import BayesianResult, Regime, RegimeSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of the confidence score."""
    overall: float
    agreement_score: float
    freshness_score: float
    regime_stability_score: float
    diversity_score: float
    extremity_score: float
    n_signals: int
    n_agreeing: int
    n_categories: int
    details: dict[str, Any] = field(default_factory=dict)


class ConfidenceCalculator:
    """Estimate reliability of the Bayesian engine's output.

    Parameters
    ----------
    min_signals : int
        Minimum number of valid signals for a meaningful confidence.
        Below this, confidence is penalised heavily.
    freshness_tau : float
        Reference time constant (seconds) for freshness scoring.
        Signals older than this are considered stale.  Default = 1 hour.
    regime_history_len : int
        Number of recent regime snapshots to consider for stability.
    weights : dict, optional
        Custom weights for the five sub-scores.
        Keys: ``agreement``, ``freshness``, ``regime_stability``,
        ``diversity``, ``extremity``.  Default weights sum to 1.0.
    """

    def __init__(
        self,
        min_signals: int = 3,
        freshness_tau: float = 3600.0,
        regime_history_len: int = 5,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.min_signals = min_signals
        self.freshness_tau = freshness_tau
        self.regime_history_len = regime_history_len

        default_weights = {
            "agreement": 0.30,
            "freshness": 0.15,
            "regime_stability": 0.15,
            "diversity": 0.15,
            "extremity": 0.25,
        }
        self.weights = weights or default_weights

        # Internal regime history buffer (most recent last).
        self._regime_history: list[Regime] = []

    # ------------------------------------------------------------------
    # Regime history management
    # ------------------------------------------------------------------

    def record_regime(self, regime: Regime) -> None:
        """Append a regime observation to the internal history buffer."""
        self._regime_history.append(regime)
        # Keep only the last N entries.
        if len(self._regime_history) > self.regime_history_len * 2:
            self._regime_history = self._regime_history[-self.regime_history_len:]

    def clear_history(self) -> None:
        """Reset internal regime history."""
        self._regime_history.clear()

    # ------------------------------------------------------------------
    # Sub-score computations
    # ------------------------------------------------------------------

    def _agreement_score(
        self, key_signals: list[dict[str, Any]], direction_sign: float
    ) -> tuple[float, int]:
        """Fraction of signals that agree with the overall direction.

        ``direction_sign`` is +1 for LONG, -1 for SHORT, 0 for NEUTRAL.

        Returns (score, n_agreeing).
        """
        if not key_signals:
            return 0.0, 0

        if direction_sign == 0:
            # NEUTRAL -- agreement is meaningless; return moderate score.
            return 0.5, 0

        agreeing = sum(
            1
            for s in key_signals
            if (s.get("contribution", 0) > 0) == (direction_sign > 0)
        )
        score = agreeing / len(key_signals)
        return score, agreeing

    def _freshness_score(self, key_signals: list[dict[str, Any]]) -> float:
        """Average freshness across signals based on their decay factor.

        The ``decay`` field in each signal dict is ``exp(-age/tau)`` which is
        already in [0, 1] with 1 = perfectly fresh.
        """
        if not key_signals:
            return 0.0

        decays = [s.get("decay", 1.0) for s in key_signals]
        # Weight-average by absolute contribution so stale-but-important
        # signals drag freshness down more.
        abs_contribs = [abs(s.get("contribution", 1.0)) for s in key_signals]
        total_contrib = sum(abs_contribs)

        if total_contrib < 1e-12:
            return float(np.mean(decays))

        weighted_freshness = sum(d * c for d, c in zip(decays, abs_contribs))
        return weighted_freshness / total_contrib

    def _regime_stability_score(self, current_regime: Regime) -> float:
        """How stable the regime has been over recent observations.

        Returns 1.0 if all recent regimes match the current one,
        lower if there have been switches.
        """
        history = self._regime_history[-self.regime_history_len:]
        if not history:
            # No history -> moderate confidence (unknown stability)
            return 0.5

        matching = sum(1 for r in history if r == current_regime)
        return matching / len(history)

    def _diversity_score(self, key_signals: list[dict[str, Any]]) -> tuple[float, int]:
        """How many distinct signal categories contribute.

        More category diversity -> higher confidence (independent evidence).

        Returns (score, n_categories).
        """
        if not key_signals:
            return 0.0, 0

        categories = {s.get("category", "unknown") for s in key_signals}
        n_cats = len(categories)

        # Expect at most ~5 categories; diminishing returns above 3.
        score = min(1.0, n_cats / 3.0)
        return score, n_cats

    def _extremity_score(self, long_probability: float) -> float:
        """How far the posterior probability is from 0.5.

        Maps distance to [0, 1] with a soft saturation.
        At P=0.7 or P=0.3 -> ~0.8.
        At P=0.9 or P=0.1 -> ~1.0.
        """
        dist = abs(long_probability - 0.5) * 2.0  # [0, 1]
        # Apply sqrt for mild compression so moderate signals still get credit.
        return float(np.sqrt(dist))

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def calculate(self, result: BayesianResult) -> ConfidenceBreakdown:
        """Compute a comprehensive confidence score for a BayesianResult.

        Parameters
        ----------
        result : BayesianResult
            Output from ``BayesianEngine.compute()``.

        Returns
        -------
        ConfidenceBreakdown
            Contains ``.overall`` confidence and sub-score breakdown.
        """
        key_signals = result.key_signals
        n_signals = len(key_signals)

        # Determine direction sign
        if result.direction == SignalDirection.LONG:
            direction_sign = 1.0
        elif result.direction == SignalDirection.SHORT:
            direction_sign = -1.0
        else:
            direction_sign = 0.0

        # Sub-scores
        agreement, n_agreeing = self._agreement_score(key_signals, direction_sign)
        freshness = self._freshness_score(key_signals)
        regime_stab = self._regime_stability_score(result.regime.regime)
        diversity, n_categories = self._diversity_score(key_signals)
        extremity = self._extremity_score(result.long_probability)

        # Record current regime for future stability tracking
        self.record_regime(result.regime.regime)

        # Weighted combination
        raw_confidence = (
            self.weights.get("agreement", 0.30) * agreement
            + self.weights.get("freshness", 0.15) * freshness
            + self.weights.get("regime_stability", 0.15) * regime_stab
            + self.weights.get("diversity", 0.15) * diversity
            + self.weights.get("extremity", 0.25) * extremity
        )

        # Penalty for too few signals
        if n_signals < self.min_signals:
            penalty = n_signals / self.min_signals
            raw_confidence *= penalty

        overall = float(np.clip(raw_confidence, 0.0, 1.0))

        return ConfidenceBreakdown(
            overall=round(overall, 4),
            agreement_score=round(agreement, 4),
            freshness_score=round(freshness, 4),
            regime_stability_score=round(regime_stab, 4),
            diversity_score=round(diversity, 4),
            extremity_score=round(extremity, 4),
            n_signals=n_signals,
            n_agreeing=n_agreeing,
            n_categories=n_categories,
            details={
                "direction_sign": direction_sign,
                "min_signals_required": self.min_signals,
                "regime_history_depth": len(self._regime_history),
            },
        )

    def calculate_from_result(self, result: BayesianResult) -> float:
        """Convenience: return just the overall confidence float."""
        return self.calculate(result).overall
