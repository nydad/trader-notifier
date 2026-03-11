"""Backtest engine: simulates trading based on signal combinations.

Vectorized day-trade backtest: enter at open, exit at close.
Models slippage, commission, position sizing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kospi_corr.domain.types import (
    BacktestMetrics,
    BacktestParams,
    SignalCombination,
    SignalDirection,
    Trade,
)
from kospi_corr.backtest.signals import SignalEvaluator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete result of backtesting one signal combination on one ETF."""
    combo: SignalCombination
    etf_symbol: str
    trades: list[Trade]
    metrics: BacktestMetrics
    equity_curve: pd.Series


class BacktestEngine:
    """Simulate trading strategy based on indicator signals."""

    def __init__(self, params: BacktestParams | None = None):
        self.params = params or BacktestParams()

    def backtest_combination(
        self,
        combo: SignalCombination,
        etf_symbol: str,
        indicator_returns: pd.DataFrame,
        etf_prices: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest for one signal combination on one ETF.

        Args:
            combo: Signal combination to test
            etf_symbol: ETF code to trade
            indicator_returns: DataFrame of indicator daily returns
            etf_prices: DataFrame with 'open' and 'close' for the ETF

        Returns:
            BacktestResult with trades, metrics, and equity curve.
        """
        evaluator = SignalEvaluator()
        trades: list[Trade] = []
        capital = self.params.initial_capital
        equity_values = []
        equity_dates = []

        position_size = capital * self.params.position_budget_ratio
        slippage_mult = 1 + (self.params.slippage_bps / 10000)
        commission_rate = self.params.commission_bps / 10000

        for date_idx in indicator_returns.index:
            row = indicator_returns.loc[date_idx]

            if not evaluator.evaluate_combination(row, combo):
                equity_values.append(capital)
                equity_dates.append(date_idx)
                continue

            if date_idx not in etf_prices.index:
                equity_values.append(capital)
                equity_dates.append(date_idx)
                continue

            etf_row = etf_prices.loc[date_idx]
            entry_price = etf_row.get("open", etf_row.get("close"))
            exit_price = etf_row.get("close")

            if pd.isna(entry_price) or pd.isna(exit_price):
                equity_values.append(capital)
                equity_dates.append(date_idx)
                continue

            if combo.direction == SignalDirection.LONG:
                adj_entry = entry_price * slippage_mult
                adj_exit = exit_price / slippage_mult
            else:
                adj_entry = entry_price / slippage_mult
                adj_exit = exit_price * slippage_mult

            qty = position_size / adj_entry
            if combo.direction == SignalDirection.LONG:
                pnl = (adj_exit - adj_entry) * qty
            else:
                pnl = (adj_entry - adj_exit) * qty

            commission = position_size * commission_rate * 2
            pnl -= commission
            return_pct = pnl / position_size

            trades.append(Trade(
                series_symbol=etf_symbol,
                direction=combo.direction,
                entry_date=date_idx.date() if hasattr(date_idx, "date") else date_idx,
                entry_price=float(entry_price),
                exit_date=date_idx.date() if hasattr(date_idx, "date") else date_idx,
                exit_price=float(exit_price),
                qty=float(qty),
                pnl=float(pnl),
                return_pct=float(return_pct),
                holding_days=1,
                exit_reason="day_close",
            ))

            capital += pnl
            equity_values.append(capital)
            equity_dates.append(date_idx)

        equity_curve = pd.Series(equity_values, index=equity_dates, name="equity")
        metrics = self._compute_metrics(combo, trades, equity_curve)

        return BacktestResult(
            combo=combo, etf_symbol=etf_symbol,
            trades=trades, metrics=metrics, equity_curve=equity_curve,
        )

    def _compute_metrics(
        self, combo: SignalCombination, trades: list[Trade], equity_curve: pd.Series,
    ) -> BacktestMetrics:
        if not trades:
            return BacktestMetrics(
                combo_key=combo.key, total_trades=0, win_count=0, loss_count=0,
                win_rate=0.0, avg_return=0.0, total_pnl=0.0,
                max_drawdown=0.0, sharpe_ratio=0.0, profit_factor=0.0, avg_holding_days=0.0,
            )

        pnls = [t.pnl for t in trades]
        returns = [t.return_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        if len(returns) > 1:
            sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
        else:
            sharpe = 0.0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestMetrics(
            combo_key=combo.key,
            total_trades=len(trades), win_count=len(wins), loss_count=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0.0,
            avg_return=float(np.mean(returns)),
            total_pnl=float(sum(pnls)),
            max_drawdown=max_dd,
            sharpe_ratio=float(sharpe),
            profit_factor=float(profit_factor),
            avg_holding_days=float(np.mean([t.holding_days for t in trades])),
        )

    def batch_backtest(
        self, combos: list[SignalCombination], etf_symbol: str,
        indicator_returns: pd.DataFrame, etf_prices: pd.DataFrame,
        min_trades: int = 5,
    ) -> list[BacktestResult]:
        results: list[BacktestResult] = []
        for i, combo in enumerate(combos):
            if (i + 1) % 50 == 0:
                logger.info(f"Backtesting combination {i+1}/{len(combos)}...")
            result = self.backtest_combination(combo, etf_symbol, indicator_returns, etf_prices)
            if result.metrics.total_trades >= min_trades:
                results.append(result)
        logger.info(f"Backtest: {len(results)}/{len(combos)} combos had >= {min_trades} trades")
        return results


def results_to_dataframe(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.metrics
        rows.append({
            "combo_key": m.combo_key, "label": r.combo.label, "etf": r.etf_symbol,
            "total_trades": m.total_trades, "win_count": m.win_count,
            "loss_count": m.loss_count, "win_rate": m.win_rate,
            "avg_return": m.avg_return, "total_pnl": m.total_pnl,
            "max_drawdown": m.max_drawdown, "sharpe_ratio": m.sharpe_ratio,
            "profit_factor": m.profit_factor, "avg_holding_days": m.avg_holding_days,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("win_rate", ascending=False)
    return df
