"""Backtest runner: orchestrates full signal-combination backtesting.

Responsibilities:
  1. Generate all valid signal combinations via SignalCombinationGenerator.
  2. Run BacktestEngine for every (ETF x signal combination) pair.
  3. Rank results by win_rate, Sharpe ratio, and profit_factor.
  4. Apply multiple-testing correction (Bonferroni / Benjamini-Hochberg FDR).
  5. Return a ranked, filtered summary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from kospi_corr.backtest.engine import BacktestEngine, BacktestResult, results_to_dataframe
from kospi_corr.backtest.signals import (
    SignalCombinationGenerator,
    build_default_conditions,
)
from kospi_corr.domain.types import (
    BacktestMetrics,
    BacktestParams,
    SignalCombination,
    SignalCondition,
    SignalDirection,
)

logger = logging.getLogger(__name__)


class CorrectionMethod(StrEnum):
    """Multiple-testing p-value correction method."""

    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg


@dataclass
class RankedResult:
    """A single entry in the ranked output, enriched with statistical tests."""

    rank: int
    etf_symbol: str
    combo: SignalCombination
    metrics: BacktestMetrics
    raw_p_value: float
    adjusted_p_value: float
    is_significant: bool  # After correction, at the chosen alpha


@dataclass
class RunnerOutput:
    """Full output of a BacktestRunner execution."""

    ranked: list[RankedResult]
    summary_df: pd.DataFrame
    all_results: list[BacktestResult]
    n_combinations_tested: int
    correction_method: CorrectionMethod
    significance_level: float


class BacktestRunner:
    """Orchestrate generation, execution, and statistical ranking of backtests.

    Example usage::

        runner = BacktestRunner()
        output = runner.execute(
            indicator_returns=ind_returns_df,
            etf_prices_dict={"122630": etf_df},
            indicator_columns=["ind_wti", "ind_usdkrw"],
        )
        print(output.summary_df.head(20))
    """

    def __init__(
        self,
        params: BacktestParams | None = None,
        correction: CorrectionMethod = CorrectionMethod.FDR_BH,
        alpha: float = 0.05,
    ) -> None:
        self.params = params or BacktestParams()
        self.correction = correction
        self.alpha = alpha
        self._engine = BacktestEngine(self.params)
        self._generator = SignalCombinationGenerator()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        indicator_returns: pd.DataFrame,
        etf_prices_dict: dict[str, pd.DataFrame],
        indicator_columns: list[str] | None = None,
        conditions: list[SignalCondition] | None = None,
        max_signal_depth: int = 2,
        min_trades: int = 5,
        directions: tuple[SignalDirection, ...] = (SignalDirection.LONG,),
        sort_by: str = "sharpe_ratio",
    ) -> RunnerOutput:
        """Run the full backtest pipeline.

        Args:
            indicator_returns: DataFrame of daily indicator changes / returns.
            etf_prices_dict: ``{etf_code: DataFrame}`` with ``open`` and ``close``.
            indicator_columns: Which columns in *indicator_returns* to build
                conditions from.  Defaults to columns starting with ``ind_``.
            conditions: Pre-built conditions (overrides auto-generation).
            max_signal_depth: Maximum conditions per combination (1-3).
            min_trades: Minimum trade count to include a result.
            directions: Trade directions to generate.
            sort_by: Primary sort column for ranking.

        Returns:
            RunnerOutput with ranked results and summary DataFrame.
        """
        # Step 1 -- Generate signal combinations
        if conditions is None:
            ind_cols = indicator_columns or [
                c for c in indicator_returns.columns if c.startswith("ind_")
            ]
            conditions = build_default_conditions(ind_cols)

        combos = self._generator.generate(
            conditions, max_depth=max_signal_depth, directions=directions
        )
        logger.info("Generated %d signal combinations to test.", len(combos))

        # Step 2 -- Run engine for each ETF x combo
        all_results: list[BacktestResult] = []
        for etf_code, etf_prices in etf_prices_dict.items():
            logger.info("Running backtest for ETF %s ...", etf_code)
            results = self._engine.batch_run(
                combos,
                etf_code,
                indicator_returns,
                etf_prices,
                min_trades=min_trades,
            )
            all_results.extend(results)

        logger.info(
            "Total results with >= %d trades: %d", min_trades, len(all_results)
        )

        # Step 3 -- Statistical testing & ranking
        raw_p_values = self._compute_p_values(all_results)
        adjusted_p_values = self._correct_p_values(raw_p_values)

        ranked = self._build_ranking(
            all_results, raw_p_values, adjusted_p_values, sort_by
        )

        # Step 4 -- Summary DataFrame
        summary_df = self._build_summary_df(ranked)

        return RunnerOutput(
            ranked=ranked,
            summary_df=summary_df,
            all_results=all_results,
            n_combinations_tested=len(combos) * len(etf_prices_dict),
            correction_method=self.correction,
            significance_level=self.alpha,
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_p_values(results: list[BacktestResult]) -> np.ndarray:
        """Compute a one-sided t-test p-value for each result.

        H0: mean trade return <= 0  (strategy has no edge)
        H1: mean trade return > 0

        For combinations with 0 or 1 trades, p = 1.0.
        """
        p_values = np.ones(len(results))
        for i, r in enumerate(results):
            returns = np.array([t.return_pct for t in r.trades])
            if len(returns) < 2:
                continue
            t_stat, two_sided_p = sp_stats.ttest_1samp(returns, 0.0)
            # One-sided: only significant if mean > 0
            if t_stat > 0:
                p_values[i] = two_sided_p / 2.0
            else:
                p_values[i] = 1.0 - two_sided_p / 2.0
        return p_values

    def _correct_p_values(self, raw_p: np.ndarray) -> np.ndarray:
        """Apply multiple-testing correction."""
        n = len(raw_p)
        if n == 0:
            return raw_p.copy()

        if self.correction == CorrectionMethod.BONFERRONI:
            return np.minimum(raw_p * n, 1.0)

        elif self.correction == CorrectionMethod.FDR_BH:
            return self._benjamini_hochberg(raw_p)

        else:
            logger.warning("Unknown correction %s, using Bonferroni.", self.correction)
            return np.minimum(raw_p * n, 1.0)

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction.

        Returns adjusted p-values (q-values).
        """
        n = len(p_values)
        if n == 0:
            return p_values.copy()

        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH formula: q_i = p_i * n / rank_i, then enforce monotonicity
        ranks = np.arange(1, n + 1)
        adjusted = sorted_p * n / ranks

        # Enforce monotonicity from the bottom up
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.minimum(adjusted, 1.0)

        # Unsort
        result = np.empty(n)
        result[sorted_idx] = adjusted
        return result

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _build_ranking(
        self,
        results: list[BacktestResult],
        raw_p: np.ndarray,
        adj_p: np.ndarray,
        sort_by: str,
    ) -> list[RankedResult]:
        """Build ranked list sorted by the chosen metric."""
        items: list[RankedResult] = []
        for i, r in enumerate(results):
            items.append(
                RankedResult(
                    rank=0,  # assigned after sorting
                    etf_symbol=r.etf_symbol,
                    combo=r.combo,
                    metrics=r.metrics,
                    raw_p_value=float(raw_p[i]),
                    adjusted_p_value=float(adj_p[i]),
                    is_significant=float(adj_p[i]) < self.alpha,
                )
            )

        # Sort descending for positive-is-better metrics
        reverse = sort_by not in ("max_drawdown",)
        items.sort(
            key=lambda x: getattr(x.metrics, sort_by, 0.0), reverse=reverse
        )

        for rank, item in enumerate(items, start=1):
            item.rank = rank

        return items

    @staticmethod
    def _build_summary_df(ranked: list[RankedResult]) -> pd.DataFrame:
        """Convert ranked results into a flat DataFrame."""
        rows = []
        for r in ranked:
            m = r.metrics
            rows.append(
                {
                    "rank": r.rank,
                    "combo_key": m.combo_key,
                    "label": r.combo.label,
                    "etf": r.etf_symbol,
                    "direction": r.combo.direction.name,
                    "total_trades": m.total_trades,
                    "win_count": m.win_count,
                    "loss_count": m.loss_count,
                    "win_rate": m.win_rate,
                    "avg_return": m.avg_return,
                    "total_pnl": m.total_pnl,
                    "max_drawdown": m.max_drawdown,
                    "sharpe_ratio": m.sharpe_ratio,
                    "profit_factor": m.profit_factor,
                    "avg_holding_days": m.avg_holding_days,
                    "raw_p_value": r.raw_p_value,
                    "adjusted_p_value": r.adjusted_p_value,
                    "is_significant": r.is_significant,
                }
            )
        return pd.DataFrame(rows)
