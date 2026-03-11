"""Signal engine for KOSPI ETF leverage trading.

Public API:
    WeightedScorer   -- Phase 2 simple weighted scorer
    BayesianEngine   -- Phase 3 Bayesian Log-Odds Pooling
    RegimeDetector   -- Market regime classification (4 states)
    ConfidenceCalculator -- Signal reliability estimation
"""
from kospi_corr.engine.bayesian import (
    BayesianEngine,
    BayesianResult,
    BayesianSignal,
    Regime,
    RegimeDetector,
    RegimeSnapshot,
)
from kospi_corr.engine.confidence import ConfidenceBreakdown, ConfidenceCalculator
from kospi_corr.engine.scorer import (
    NormMethod,
    ScorerResult,
    SignalInput,
    WeightedScorer,
)

__all__ = [
    # scorer.py
    "WeightedScorer",
    "SignalInput",
    "ScorerResult",
    "NormMethod",
    # bayesian.py
    "BayesianEngine",
    "BayesianSignal",
    "BayesianResult",
    "RegimeDetector",
    "RegimeSnapshot",
    "Regime",
    # confidence.py
    "ConfidenceCalculator",
    "ConfidenceBreakdown",
]
