"""Signal engine for KOSPI ETF leverage trading.

Public API:
    SignalPipeline       -- Unified signal generation pipeline
    SignalResult         -- Pipeline output container
    WeightedScorer       -- Phase 2 simple weighted scorer
    BayesianEngine       -- Phase 3 Bayesian Log-Odds Pooling
    RegimeDetector       -- Market regime classification (4 states)
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
from kospi_corr.engine.decision import DecisionEngine, TradingDecision
from kospi_corr.engine.predictive import PredictiveEngine, PredictiveResult
from kospi_corr.engine.signal_pipeline import SignalPipeline, SignalResult

__all__ = [
    # signal_pipeline.py
    "SignalPipeline",
    "SignalResult",
    # predictive.py
    "PredictiveEngine",
    "PredictiveResult",
    # decision.py
    "DecisionEngine",
    "TradingDecision",
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
