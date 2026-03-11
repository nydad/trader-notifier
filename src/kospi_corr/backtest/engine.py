"""Backtest engine: day-by-day simulation of signal-driven trades.

Entry logic: signal fires on day T (based on indicator data available at close of T).
Trade executed on day T+1: enter at open, exit at close (intraday round-trip).
Models slippage (default 10 bps) and commission (default 3 bps each way).

Supports both LONG and SHORT directions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kospi_corr.backtest.signals import SignalEvaluator
from kospi_corr.domain.types import (
    BacktestMetrics,
    BacktestParams,
    SignalCombination,
    SignalDirection,
    Trade,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete result of backtesting one signal combination on one ETF."""

    combo: SignalCombination
    etf_symbol: str
    trades: list[Trade]
    metrics: BacktestMetrics
    equity_curve: pd.Series
    daily_returns: pd.Series


class BacktestEngine:
    """Simulate trading strategy based on indicator signals.

    Workflow per day:
      1. At end-of-day T, evaluate signal conditions on indicator returns.
      2. If signal fires, schedule a trade for day T+1.
      3. On day T+1, enter at open (with slippage), exit at close (with slippage).
      4. Deduct round-trip commission.
      5. Track PnL, equity curve, and per-trade statistics.
    """

    def __init__(self, params: BacktestParams | None = None) -> None:
        self.params = params or BacktestParams()
        self._evaluator = SignalEvaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        combo: SignalCombination,
        etf_symbol: str,
        indicator_returns: pd.DataFrame,
        etf_prices: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest for one signal combination on one ETF.

        Args:
            combo: Signal combination to test (AND-logic conditions + direction).
            etf_symbol: ETF ticker / code being traded.
            indicator_returns: DataFrame indexed by date with one column per
                indicator.  Values are daily returns / changes used by
                :class:`SignalEvaluator`.
            etf_prices: DataFrame indexed by date with at least ``open`` and
                ``close`` columns for *etf_symbol*.

        Returns:
            BacktestResult containing trades, metrics, and equity curve.
        """
        trades: list[Trade] = []
        capital = self.params.initial_capital
        position_budget = capital * self.params.position_budget_ratio

        slippage_frac = self.params.slippage_bps / 10_000
        commission_frac = self.params.commission_bps / 10_000

        equity_values: list[float] = []
        equity_dates: list = []

        # Build a sorted list of dates that appear in both frames.
        common_dates = indicator_returns.index.intersection(etf_prices.index).sort_values()
        date_list = list(common_dates)

        signal_fired_today = False

        for i, today in enumerate(date_list):
            # ----------------------------------------------------------
            # Step A: Check if yesterday's signal scheduled a trade today
            # ----------------------------------------------------------
            if signal_fired_today:
                signal_fired_today = False  # consume the flag

                etf_row = etf_prices.loc[today]
                entry_price_raw = (
                    etf_row["open"] if "open" in etf_row.index else etf_row["close"]
                )
                exit_price_raw = etf_row["close"]

                if pd.notna(entry_price_raw) and pd.notna(exit_price_raw):
                    trade = self._execute_trade(
                        combo=combo,
                        etf_symbol=etf_symbol,
                        trade_date=today,
                        entry_price_raw=float(entry_price_raw),
                        exit_price_raw=float(exit_price_raw),
                        position_budget=position_budget,
                        slippage_frac=slippage_frac,
                        commission_frac=commission_frac,
                    )
                    trades.append(trade)
                    capital += trade.pnl

            # ----------------------------------------------------------
            # Step B: Evaluate signal conditions for *today's* indicator data.
            #         If signal fires, the trade will happen on the NEXT day.
            # ----------------------------------------------------------
            ind_row = indicator_returns.loc[today]
            if self._evaluator.evaluate_combination(ind_row, combo):
                # Only fire if there is a next trading day available.
                if i + 1 < len(date_list):
                    signal_fired_today = True

            equity_values.append(capital)
            equity_dates.append(today)

        equity_curve = pd.Series(equity_values, index=equity_dates, name="equity")
        daily_rets = equity_curve.pct_change().fillna(0.0)
        daily_rets.name = "daily_return"

        metrics = self._compute_metrics(combo, trades, equity_curve, daily_rets)

        return BacktestResult(
            combo=combo,
            etf_symbol=etf_symbol,
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            daily_returns=daily_rets,
        )

    def batch_run(
        self,
        combos: list[SignalCombination],
        etf_symbol: str,
        indicator_returns: pd.DataFrame,
        etf_prices: pd.DataFrame,
        min_trades: int = 5,
    ) -> list[BacktestResult]:
        """Run backtest for many signal combinations and filter by minimum trade count."""
        results: list[BacktestResult] = []
        for idx, combo in enumerate(combos):
            if (idx + 1) % 100 == 0:
                logger.info(
                    "Backtesting combination %d / %d ...", idx + 1, len(combos)
                )
            result = self.run(combo, etf_symbol, indicator_returns, etf_prices)
            if result.metrics.total_trades >= min_trades:
                results.append(result)

        logger.info(
            "Backtest complete: %d / %d combos passed min_trades=%d filter",
            len(results),
            len(combos),
            min_trades,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_trade(
        combo: SignalCombination,
        etf_symbol: str,
        trade_date,
        entry_price_raw: float,
        exit_price_raw: float,
        position_budget: float,
        slippage_frac: float,
        commission_frac: float,
    ) -> Trade:
        """Construct a single Trade with slippage and commission applied."""
        direction = combo.direction

        if direction == SignalDirection.LONG:
            # Buy at open (slippage pushes price up), sell at close (slippage pushes price down)
            entry_price = entry_price_raw * (1 + slippage_frac)
            exit_price = exit_price_raw * (1 - slippage_frac)
        else:
            # Short sell at open (slippage pushes price down), cover at close (slippage pushes price up)
            entry_price = entry_price_raw * (1 - slippage_frac)
            exit_price = exit_price_raw * (1 + slippage_frac)

        qty = position_budget / entry_price

        if direction == SignalDirection.LONG:
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        # Round-trip commission: charged on both entry and exit notional
        commission = position_budget * commission_frac * 2
        net_pnl = gross_pnl - commission
        return_pct = net_pnl / position_budget if position_budget > 0 else 0.0

        trade_dt = trade_date.date() if hasattr(trade_date, "date") else trade_date

        return Trade(
            series_symbol=etf_symbol,
            direction=direction,
            entry_date=trade_dt,
            entry_price=entry_price_raw,
            exit_date=trade_dt,
            exit_price=exit_price_raw,
            qty=float(qty),
            pnl=float(net_pnl),
            return_pct=float(return_pct),
            holding_days=1,
            exit_reason="day_close",
        )

    @staticmethod
    def _compute_metrics(
        combo: SignalCombination,
        trades: list[Trade],
        equity_curve: pd.Series,
        daily_returns: pd.Series,
    ) -> BacktestMetrics:
        """Compute summary statistics from completed trades and equity curve."""
        if not trades:
            return BacktestMetrics(
                combo_key=combo.key,
                total_trades=0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                avg_return=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                avg_holding_days=0.0,
            )

        pnls = np.array([t.pnl for t in trades])
        returns = np.array([t.return_pct for t in trades])
        holding_days = np.array([t.holding_days for t in trades])

        win_mask = pnls > 0
        win_count = int(win_mask.sum())
        loss_count = len(trades) - win_count

        # Max drawdown from equity curve
        running_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - running_max) / running_max
        max_drawdown = float(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0

        # Sharpe ratio: annualized, based on per-trade returns
        if len(returns) > 1 and np.std(returns, ddof=1) > 0:
            sharpe = float(
                (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = float(pnls[win_mask].sum()) if win_count > 0 else 0.0
        gross_loss = float(np.abs(pnls[~win_mask]).sum()) if loss_count > 0 else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        return BacktestMetrics(
            combo_key=combo.key,
            total_trades=len(trades),
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_count / len(trades),
            avg_return=float(np.mean(returns)),
            total_pnl=float(pnls.sum()),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_holding_days=float(np.mean(holding_days)),
        )


# ------------------------------------------------------------------
# Utility: convert results list to a flat DataFrame for analysis
# ------------------------------------------------------------------


def results_to_dataframe(results: list[BacktestResult]) -> pd.DataFrame:
    """Convert a list of BacktestResult into a summary DataFrame, sorted by win_rate."""
    rows = []
    for r in results:
        m = r.metrics
        rows.append(
            {
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
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("win_rate", ascending=False).reset_index(drop=True)
    return df
