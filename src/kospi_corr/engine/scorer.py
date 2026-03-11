"""Weighted signal scorer for KOSPI ETF leverage trading.

Takes raw indicator values and their configured weights, normalizes each
signal to [0, 1], and computes a weighted composite score that maps to
LONG / SHORT probabilities.

This is the Phase 2 scorer (simple weighted average).  Phase 3 replaces it
with BayesianEngine for proper log-odds fusion, but WeightedScorer remains
useful as a fast baseline and as input to the Bayesian pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import numpy as np

from kospi_corr.domain.types import SignalDirection

logger = logging.getLogger(__name__)

# Epsilon to avoid division by zero / log(0)
_EPS = 1e-8


class NormMethod(StrEnum):
    """Supported normalisation methods."""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    SIGMOID = "sigmoid"
    RANK = "rank"


@dataclass(frozen=True)
class SignalInput:
    """One indicator observation fed into the scorer.

    Attributes:
        name:       Indicator key (e.g. ``wti_crude``, ``sp500``).
        raw_value:  The raw indicator value (price change, return, etc.).
        weight:     Importance weight for this signal (>= 0).
        direction_hint:
            +1 means higher raw_value is bullish (e.g. S&P500 up -> KOSPI up).
            -1 means higher raw_value is bearish (e.g. USD/KRW up -> KOSPI down).
        norm_min:   Lower bound for min-max normalisation (optional).
        norm_max:   Upper bound for min-max normalisation (optional).
        norm_mean:  Mean for z-score normalisation (optional).
        norm_std:   Std-dev for z-score normalisation (optional).
    """
    name: str
    raw_value: float
    weight: float = 1.0
    direction_hint: int = 1
    norm_min: float | None = None
    norm_max: float | None = None
    norm_mean: float | None = None
    norm_std: float | None = None


@dataclass
class ScorerResult:
    """Output of WeightedScorer.score()."""
    long_probability: float
    short_probability: float
    raw_score: float  # weighted sum before sigmoid, in (-inf, +inf)
    direction: SignalDirection
    signal_contributions: dict[str, float] = field(default_factory=dict)


class WeightedScorer:
    """Normalise indicator values and produce a weighted directional score.

    The pipeline is:
        1. Normalise each signal to [0, 1].
        2. Shift to [-0.5, +0.5] so that 0 = neutral.
        3. Flip sign for inverse-correlated indicators (direction_hint == -1).
        4. Multiply by weight and sum.
        5. Map the composite score through a sigmoid to get P(LONG).

    Parameters
    ----------
    norm_method : NormMethod
        Default normalisation method.  Per-signal overrides are possible
        via ``norm_min/max`` or ``norm_mean/std`` fields on ``SignalInput``.
    sigmoid_scale : float
        Controls the steepness of the final sigmoid mapping.
        Larger values -> sharper transition around 0.5.
    neutral_band : float
        Half-width of the "NEUTRAL" zone around 0.5.
        If P(LONG) is within [0.5 - band, 0.5 + band], direction is NEUTRAL.
    """

    def __init__(
        self,
        norm_method: NormMethod = NormMethod.SIGMOID,
        sigmoid_scale: float = 4.0,
        neutral_band: float = 0.05,
    ) -> None:
        self.norm_method = norm_method
        self.sigmoid_scale = sigmoid_scale
        self.neutral_band = neutral_band

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    def normalise(self, signal: SignalInput) -> float:
        """Normalise a single signal value to [0, 1]."""
        v = signal.raw_value

        # If the caller supplies explicit min/max, use minmax regardless.
        if signal.norm_min is not None and signal.norm_max is not None:
            span = signal.norm_max - signal.norm_min
            if abs(span) < _EPS:
                return 0.5
            return float(np.clip((v - signal.norm_min) / span, 0.0, 1.0))

        # If the caller supplies mean/std, use z-score -> sigmoid.
        if signal.norm_mean is not None and signal.norm_std is not None:
            if signal.norm_std < _EPS:
                return 0.5
            z = (v - signal.norm_mean) / signal.norm_std
            return float(self._sigmoid(z))

        # Fall back to default method.
        if self.norm_method == NormMethod.SIGMOID:
            # Treat raw_value as already in a "reasonable" range; apply sigmoid.
            return float(self._sigmoid(v))
        elif self.norm_method == NormMethod.ZSCORE:
            # Without historical stats, we can only use sigmoid on raw value.
            return float(self._sigmoid(v))
        else:
            # MINMAX / RANK without context -> sigmoid fallback.
            return float(self._sigmoid(v))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, signals: Sequence[SignalInput]) -> ScorerResult:
        """Compute weighted score from a list of signals.

        Returns
        -------
        ScorerResult
            Contains ``long_probability``, ``short_probability``,
            ``raw_score``, ``direction``, and per-signal contributions.
        """
        if not signals:
            return ScorerResult(
                long_probability=0.5,
                short_probability=0.5,
                raw_score=0.0,
                direction=SignalDirection.NEUTRAL,
            )

        total_weight = 0.0
        weighted_sum = 0.0
        contributions: dict[str, float] = {}

        for sig in signals:
            if np.isnan(sig.raw_value):
                logger.debug("Skipping NaN signal: %s", sig.name)
                continue

            normed = self.normalise(sig)        # [0, 1]
            centered = normed - 0.5              # [-0.5, +0.5]
            directed = centered * sig.direction_hint  # flip if inverse
            contribution = directed * sig.weight

            weighted_sum += contribution
            total_weight += abs(sig.weight)
            contributions[sig.name] = contribution

        # Normalise by total weight so that adding more signals doesn't
        # inflate the score unboundedly.
        if total_weight > _EPS:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 0.0

        # Scale and push through sigmoid for final probability.
        long_prob = float(self._sigmoid(np.float64(raw_score * self.sigmoid_scale)))
        short_prob = 1.0 - long_prob

        # Direction classification.
        if long_prob > 0.5 + self.neutral_band:
            direction = SignalDirection.LONG
        elif long_prob < 0.5 - self.neutral_band:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        return ScorerResult(
            long_probability=round(long_prob, 6),
            short_probability=round(short_prob, 6),
            raw_score=round(raw_score, 6),
            direction=direction,
            signal_contributions=contributions,
        )

    def score_from_dict(
        self,
        values: dict[str, float],
        weights: dict[str, float] | None = None,
        directions: dict[str, int] | None = None,
        norm_params: dict[str, dict[str, float]] | None = None,
    ) -> ScorerResult:
        """Convenience wrapper: build SignalInputs from plain dicts.

        Parameters
        ----------
        values : dict
            ``{indicator_name: raw_value}``
        weights : dict, optional
            ``{indicator_name: weight}``.  Missing keys default to 1.0.
        directions : dict, optional
            ``{indicator_name: +1 or -1}``.  Missing keys default to +1.
        norm_params : dict, optional
            ``{indicator_name: {"min": ..., "max": ...}}`` or
            ``{indicator_name: {"mean": ..., "std": ...}}``.
        """
        weights = weights or {}
        directions = directions or {}
        norm_params = norm_params or {}

        inputs: list[SignalInput] = []
        for name, raw in values.items():
            np_kw: dict[str, Any] = {}
            if name in norm_params:
                p = norm_params[name]
                if "min" in p and "max" in p:
                    np_kw["norm_min"] = p["min"]
                    np_kw["norm_max"] = p["max"]
                elif "mean" in p and "std" in p:
                    np_kw["norm_mean"] = p["mean"]
                    np_kw["norm_std"] = p["std"]

            inputs.append(
                SignalInput(
                    name=name,
                    raw_value=raw,
                    weight=weights.get(name, 1.0),
                    direction_hint=directions.get(name, 1),
                    **np_kw,
                )
            )

        return self.score(inputs)
