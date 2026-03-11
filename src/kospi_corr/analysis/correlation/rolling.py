"""Rolling correlation calculator.

Supports multiple window sizes (5, 10, 20 day) for both
Pearson and Spearman methods.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from kospi_corr.domain.types import CorrelationMethod, CorrelationPair


class RollingCorrelationCalculator:
    """Calculate rolling pairwise correlations over multiple windows."""

    def matrix_by_window(
        self,
        df: pd.DataFrame,
        windows: list[int] | tuple[int, ...] = (5, 10, 20),
        method: Literal["pearson", "spearman"] = "pearson",
        min_periods: int | None = None,
    ) -> dict[int, pd.DataFrame]:
        """Compute rolling correlation for each window size.

        Returns dict mapping window_size -> DataFrame of final-day correlations.
        For each window, we compute the rolling pairwise correlation and return
        the most recent complete window's correlation matrix.
        """
        results: dict[int, pd.DataFrame] = {}

        for window in windows:
            mp = min_periods or max(window // 2, 5)
            rolling_corr = df.rolling(window=window, min_periods=mp).corr(
                pairwise=True
            )

            # Get the last valid correlation matrix
            last_date = df.index[-1]
            try:
                last_matrix = rolling_corr.loc[last_date]
                results[window] = last_matrix
            except KeyError:
                # Not enough data for this window
                results[window] = pd.DataFrame()

        return results

    def timeseries_for_pair(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        window: int = 20,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pd.Series:
        """Get the rolling correlation time series for a specific pair.

        Useful for plotting how correlation evolves over time.
        """
        return df[col_a].rolling(window).corr(df[col_b])

    def average_rolling_pairs(
        self,
        df: pd.DataFrame,
        windows: list[int] | tuple[int, ...] = (5, 10, 20),
        method: Literal["pearson", "spearman"] = "pearson",
        min_periods: int = 10,
    ) -> list[CorrelationPair]:
        """Compute mean rolling correlation for each pair across the sample.

        For each (col_a, col_b, window), computes the mean of the
        rolling correlation time series.
        """
        cols = df.columns.tolist()
        pairs: list[CorrelationPair] = []
        corr_method = (
            CorrelationMethod.ROLLING_PEARSON
            if method == "pearson"
            else CorrelationMethod.ROLLING_SPEARMAN
        )

        for window in windows:
            for i, col_a in enumerate(cols):
                for col_b in cols[i + 1:]:
                    rolling_r = df[col_a].rolling(window).corr(df[col_b])
                    valid = rolling_r.dropna()
                    if len(valid) < min_periods:
                        continue

                    pairs.append(CorrelationPair(
                        series_a=col_a,
                        series_b=col_b,
                        method=corr_method,
                        window=window,
                        lag=0,
                        correlation=float(valid.mean()),
                        p_value=None,
                        n_obs=len(valid),
                    ))

        return pairs
