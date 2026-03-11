"""Backtest pipeline -- signal testing orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from kospi_corr.domain.types import BacktestParams, SignalDirection
from kospi_corr.backtest.signals import SignalCombinationGenerator, build_default_conditions
from kospi_corr.backtest.simulator import BacktestEngine, results_to_dataframe
from kospi_corr.visualization.ranking_table import RankingTableRenderer

logger = logging.getLogger(__name__)


class BacktestPipeline:
    """End-to-end backtest: generate signals -> simulate -> rank -> report."""

    def __init__(self, output_dir: Path = Path("output"), params: BacktestParams | None = None):
        self.output_dir = output_dir
        self.params = params or BacktestParams()
        self._engine = BacktestEngine(self.params)
        self._ranking = RankingTableRenderer()

    def execute(
        self,
        indicator_returns: pd.DataFrame,
        etf_prices_dict: dict[str, pd.DataFrame],
        indicator_columns: list[str] | None = None,
        max_signal_depth: int = 2,
        min_trades: int = 5,
        directions: tuple[SignalDirection, ...] = (SignalDirection.LONG,),
    ) -> dict:
        ind_cols = indicator_columns or [c for c in indicator_returns.columns if c.startswith("ind_")]
        conditions = build_default_conditions(ind_cols)
        combos = SignalCombinationGenerator().generate(conditions, max_depth=max_signal_depth, directions=directions)

        all_results = []
        for code, prices in etf_prices_dict.items():
            results = self._engine.batch_backtest(combos, code, indicator_returns, prices, min_trades=min_trades)
            all_results.extend(results)

        summary_df = results_to_dataframe(all_results)
        artifacts = {}
        if not summary_df.empty:
            p = self.output_dir / "backtest_ranking.png"
            self._ranking.render_backtest_ranking(summary_df, p)
            artifacts["ranking"] = p
            csv_p = self.output_dir / "backtest_results.csv"
            csv_p.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(csv_p, index=False)
            artifacts["csv"] = csv_p
        return {"summary_df": summary_df, "all_results": all_results, "artifacts": artifacts}
