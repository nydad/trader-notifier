"""Lead-lag correlation analysis.

Two complementary approaches (as recommended by Spark reviewer):
1. Shifted correlation: exploratory, finds optimal lag
2. Granger causality: statistical test for predictive relationship

Key insight: Does WTI oil price movement LEAD KOSPI ETF movement?
If lag=-1 has highest correlation, it means yesterday's oil predicts
today's KOSPI return.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from kospi_corr.domain.types import CorrelationMethod, CorrelationPair

logger = logging.getLogger(__name__)


class LeadLagCalculator:
    """Analyze lead-lag relationships between time series."""

    def shifted_correlation(
        self,
        df: pd.DataFrame,
        max_lag: int = 5,
        method: Literal["pearson", "spearman"] = "pearson",
        min_periods: int = 20,
    ) -> list[CorrelationPair]:
        """Compute correlation at various lags for all column pairs.

        For lag=k: correlate series_a(t) with series_b(t+k).
        Positive lag means series_a leads series_b.
        Negative lag means series_b leads series_a.

        Returns list of CorrelationPair for each (pair, lag) combination.
        """
        cols = df.columns.tolist()
        pairs: list[CorrelationPair] = []
        corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr

        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        continue  # Skip zero-lag (covered by standard correlation)

                    # Shift series_b by lag
                    a = df[col_a]
                    b = df[col_b].shift(-lag)

                    # Align and drop NaN
                    mask = pd.concat([a, b], axis=1).dropna()
                    n = len(mask)

                    if n < min_periods:
                        continue

                    r, p = corr_func(mask.iloc[:, 0], mask.iloc[:, 1])

                    pairs.append(CorrelationPair(
                        series_a=col_a,
                        series_b=col_b,
                        method=CorrelationMethod.LEAD_LAG,
                        window=None,
                        lag=lag,
                        correlation=float(r),
                        p_value=float(p),
                        n_obs=n,
                    ))

        return pairs

    def optimal_lag(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        max_lag: int = 5,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> dict:
        """Find the lag that maximizes absolute correlation between two series.

        Returns dict with optimal_lag, correlation, p_value, all_lags.
        """
        corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
        results = {}

        for lag in range(-max_lag, max_lag + 1):
            a = series_a
            b = series_b.shift(-lag)
            mask = pd.concat([a, b], axis=1).dropna()

            if len(mask) < 10:
                continue

            r, p = corr_func(mask.iloc[:, 0], mask.iloc[:, 1])
            results[lag] = {"correlation": float(r), "p_value": float(p), "n": len(mask)}

        if not results:
            return {"optimal_lag": 0, "correlation": 0.0, "p_value": 1.0, "all_lags": {}}

        best_lag = max(results, key=lambda k: abs(results[k]["correlation"]))
        return {
            "optimal_lag": best_lag,
            "correlation": results[best_lag]["correlation"],
            "p_value": results[best_lag]["p_value"],
            "all_lags": results,
        }

    def granger_causality(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        max_lag: int = 5,
        significance: float = 0.05,
    ) -> dict:
        """Test Granger causality: does series_a Granger-cause series_b?

        Uses statsmodels grangercausalitytests.

        Returns dict with test results per lag and overall verdict.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        # Prepare data: both series must be stationary (use returns)
        combined = pd.concat([series_b, series_a], axis=1).dropna()

        if len(combined) < max_lag + 20:
            return {"is_causal": False, "reason": "insufficient_data", "results": {}}

        try:
            results = grangercausalitytests(combined.values, maxlag=max_lag, verbose=False)
        except Exception as exc:
            logger.warning(f"Granger test failed: {exc}")
            return {"is_causal": False, "reason": str(exc), "results": {}}

        # Check if any lag shows significance
        lag_results = {}
        any_significant = False

        for lag, tests in results.items():
            f_test = tests[0]["ssr_ftest"]
            p_value = f_test[1]
            lag_results[lag] = {
                "f_stat": float(f_test[0]),
                "p_value": float(p_value),
                "is_significant": p_value < significance,
            }
            if p_value < significance:
                any_significant = True

        return {
            "is_causal": any_significant,
            "reason": "significant" if any_significant else "not_significant",
            "results": lag_results,
        }
