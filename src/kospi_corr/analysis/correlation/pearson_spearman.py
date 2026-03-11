"""Pearson and Spearman correlation calculators."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from kospi_corr.domain.types import CorrelationMethod, CorrelationPair


class PearsonSpearmanCalculator:
    """Calculate full Pearson or Spearman correlation matrices with p-values."""

    def matrix(
        self,
        df: pd.DataFrame,
        method: Literal["pearson", "spearman"] = "pearson",
        min_periods: int = 20,
    ) -> pd.DataFrame:
        """Compute pairwise correlation matrix.

        Returns DataFrame with same row/col labels as input columns.
        """
        return df.corr(method=method, min_periods=min_periods)

    def pairs_with_pvalues(
        self,
        df: pd.DataFrame,
        method: Literal["pearson", "spearman"] = "pearson",
        min_periods: int = 20,
    ) -> list[CorrelationPair]:
        """Compute all pairwise correlations with p-values.

        Returns list of CorrelationPair for every (col_a, col_b) where a < b.
        """
        cols = df.columns.tolist()
        pairs: list[CorrelationPair] = []

        corr_method = CorrelationMethod.PEARSON if method == "pearson" else CorrelationMethod.SPEARMAN
        corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr

        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                # Drop rows where either is NaN
                mask = df[[col_a, col_b]].dropna()
                n = len(mask)

                if n < min_periods:
                    continue

                r, p = corr_func(mask[col_a], mask[col_b])

                pairs.append(CorrelationPair(
                    series_a=col_a,
                    series_b=col_b,
                    method=corr_method,
                    window=None,
                    lag=0,
                    correlation=float(r),
                    p_value=float(p),
                    n_obs=n,
                ))

        return pairs
