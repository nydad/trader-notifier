"""Backtesting framework for KOSPI ETF leverage trading signals.

Public API:
  - BacktestEngine / BacktestResult / results_to_dataframe -- core simulation
  - BacktestRunner / RunnerOutput / CorrectionMethod -- orchestration & ranking
  - SignalEvaluator / SignalCombinationGenerator -- signal definitions
"""
from kospi_corr.backtest.engine import BacktestEngine, BacktestResult, results_to_dataframe
from kospi_corr.backtest.runner import BacktestRunner, CorrectionMethod, RankedResult, RunnerOutput
from kospi_corr.backtest.signals import (
    SignalCombinationGenerator,
    SignalEvaluator,
    build_default_conditions,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestRunner",
    "CorrectionMethod",
    "RankedResult",
    "RunnerOutput",
    "SignalCombinationGenerator",
    "SignalEvaluator",
    "build_default_conditions",
    "results_to_dataframe",
]
