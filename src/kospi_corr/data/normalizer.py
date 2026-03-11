"""Data alignment and normalization layer.

Critical responsibility: prevent look-ahead bias by properly
aligning timezone-disparate data sources.

Key rules (from Spark reviewer):
  - US market data (S&P500, NASDAQ, VIX) must be lagged by 1 KRX business day
    because US markets close AFTER KOSPI (15:30 KST).
  - Nikkei and Shanghai overlap with KOSPI hours, so same-day is valid.
  - All data normalized to KRX trading calendar dates.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Aligns multi-source data to a common KRX trading calendar."""

    def __init__(self, krx_dates: pd.DatetimeIndex | None = None):
        self._krx_dates = krx_dates

    def align_to_krx_calendar(
        self,
        etf_prices: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        indicator_lags: dict[str, int],
    ) -> pd.DataFrame:
        """Build a unified DataFrame aligned to KRX trading dates.

        Args:
            etf_prices: {symbol: DataFrame with close column}
            indicators: {indicator_key: DataFrame with value column}
            indicator_lags: {indicator_key: lag_days} for look-ahead prevention

        Returns:
            DataFrame with DatetimeIndex (KRX dates) and columns for
            each ETF close and each indicator value.
        """
        # Determine KRX trading dates from ETF data
        all_dates = set()
        for df in etf_prices.values():
            all_dates.update(df.index.normalize())
        krx_dates = sorted(all_dates)

        if not krx_dates:
            logger.warning("No KRX dates found in ETF data")
            return pd.DataFrame()

        krx_index = pd.DatetimeIndex(krx_dates, name="date")
        result = pd.DataFrame(index=krx_index)

        # Add ETF close prices
        for symbol, df in etf_prices.items():
            col_name = f"etf_{symbol}_close"
            series = df["close"] if "close" in df.columns else df.iloc[:, 0]
            series.index = series.index.normalize()
            result[col_name] = series.reindex(krx_index, method=None)

        # Add indicators with lag adjustment
        for key, df in indicators.items():
            col_name = f"ind_{key}"
            series = df["value"] if "value" in df.columns else df.iloc[:, 0]
            series.index = series.index.normalize()

            lag = indicator_lags.get(key, 0)
            if lag > 0:
                # Shift the indicator data forward by lag days
                # This means: on KRX date T, we use indicator value from T-lag
                series = series.shift(lag, freq="B")
                logger.debug(f"Applied {lag}-day lag to indicator {key}")

            result[col_name] = series.reindex(krx_index, method="ffill", limit=3)

        return result

    def compute_returns(
        self,
        aligned_df: pd.DataFrame,
        method: str = "log",
    ) -> pd.DataFrame:
        """Compute returns for correlation analysis.

        Args:
            aligned_df: output of align_to_krx_calendar
            method: 'log' for log returns, 'simple' for arithmetic returns

        Returns:
            DataFrame of daily returns (first row is NaN).
        """
        if method == "log":
            returns = np.log(aligned_df / aligned_df.shift(1))
        else:
            returns = aligned_df.pct_change()

        # Drop first NaN row
        returns = returns.iloc[1:]

        # Warn about columns with too many NaNs
        nan_pct = returns.isna().mean()
        for col in nan_pct[nan_pct > 0.3].index:
            logger.warning(f"Column {col} has {nan_pct[col]:.0%} NaN values")

        return returns

    def get_overlap_mask(self, df: pd.DataFrame, min_cols: int = 2) -> pd.Series:
        """Return boolean mask of rows where at least min_cols have data."""
        return df.notna().sum(axis=1) >= min_cols
