"""Correlation analysis service -- orchestrates all correlation methods."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from kospi_corr.domain.types import (
    CorrelationMethod,
    CorrelationPair,
    CorrelationRequest,
    CorrelationRunResult,
)
from kospi_corr.analysis.correlation.pearson_spearman import PearsonSpearmanCalculator
from kospi_corr.analysis.correlation.rolling import RollingCorrelationCalculator
from kospi_corr.analysis.correlation.lead_lag import LeadLagCalculator
from kospi_corr.analysis.correlation.partial import PartialCorrelationCalculator

logger = logging.getLogger(__name__)


class CorrelationRunService:
    """Executes a full correlation analysis run across all requested methods."""

    def __init__(self) -> None:
        self._pearson_spearman = PearsonSpearmanCalculator()
        self._rolling = RollingCorrelationCalculator()
        self._lead_lag = LeadLagCalculator()
        self._partial = PartialCorrelationCalculator()

    def run(
        self,
        returns_df: pd.DataFrame,
        request: CorrelationRequest,
    ) -> CorrelationRunResult:
        """Execute all requested correlation calculations.

        Args:
            returns_df: DataFrame of daily returns (aligned, no look-ahead bias).
            request: Correlation analysis parameters.

        Returns:
            CorrelationRunResult with all computed pairs.
        """
        all_pairs: list[CorrelationPair] = []

        # 1. Static correlations (Pearson / Spearman)
        for method in request.methods:
            if method in (CorrelationMethod.PEARSON, CorrelationMethod.SPEARMAN):
                logger.info(f"Computing {method.value} correlation matrix...")
                pairs = self._pearson_spearman.pairs_with_pvalues(
                    returns_df,
                    method=method.value,
                    min_periods=request.min_periods,
                )
                all_pairs.extend(pairs)
                logger.info(f"  -> {len(pairs)} pairs computed")

        # 2. Rolling correlations
        rolling_methods = [
            m for m in request.methods
            if m in (CorrelationMethod.ROLLING_PEARSON, CorrelationMethod.ROLLING_SPEARMAN)
        ]
        if rolling_methods or request.rolling_windows:
            base_method = "spearman" if CorrelationMethod.ROLLING_SPEARMAN in request.methods else "pearson"
            logger.info(f"Computing rolling correlations (windows={request.rolling_windows})...")
            pairs = self._rolling.average_rolling_pairs(
                returns_df,
                windows=list(request.rolling_windows),
                method=base_method,
                min_periods=request.min_periods,
            )
            all_pairs.extend(pairs)
            logger.info(f"  -> {len(pairs)} rolling pairs computed")

        # 3. Lead-lag analysis
        if CorrelationMethod.LEAD_LAG in request.methods or request.max_lag > 0:
            logger.info(f"Computing lead-lag correlations (max_lag={request.max_lag})...")
            pairs = self._lead_lag.shifted_correlation(
                returns_df,
                max_lag=request.max_lag,
                min_periods=request.min_periods,
            )
            all_pairs.extend(pairs)
            logger.info(f"  -> {len(pairs)} lead-lag pairs computed")

        # 4. Partial correlations
        if request.partial_controls:
            logger.info(f"Computing partial correlations (controls={request.partial_controls})...")
            controls = [c for c in request.partial_controls if c in returns_df.columns]
            if controls:
                pairs = self._partial.pairwise_partial(
                    returns_df,
                    controls=controls,
                    min_periods=request.min_periods,
                )
                all_pairs.extend(pairs)
                logger.info(f"  -> {len(pairs)} partial pairs computed")
            else:
                logger.warning("No control columns found in data, skipping partial correlation")

        return CorrelationRunResult(
            run_id=0,  # Assigned by storage layer
            request=request,
            pairs=all_pairs,
            created_at=datetime.now(),
        )

    def rank_pairs(
        self,
        result: CorrelationRunResult,
        top_n: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """Rank correlation pairs by absolute strength.

        Returns dict of DataFrames:
          - 'by_abs': top N by |correlation|
          - 'by_positive': top N positive correlations
          - 'by_negative': top N negative correlations
          - 'lead_lag': strongest lead-lag relationships
        """
        if not result.pairs:
            return {}

        df = pd.DataFrame([
            {
                "series_a": p.series_a,
                "series_b": p.series_b,
                "method": p.method.value,
                "window": p.window,
                "lag": p.lag,
                "correlation": p.correlation,
                "p_value": p.p_value,
                "n_obs": p.n_obs,
                "abs_corr": abs(p.correlation),
            }
            for p in result.pairs
        ])

        rankings = {}

        # Overall by absolute correlation
        rankings["by_abs"] = df.nlargest(top_n, "abs_corr")

        # Strongest positive
        positive = df[df["correlation"] > 0]
        rankings["by_positive"] = positive.nlargest(top_n, "correlation")

        # Strongest negative
        negative = df[df["correlation"] < 0]
        rankings["by_negative"] = negative.nsmallest(top_n, "correlation")

        # Lead-lag specific
        leadlag = df[df["method"] == "leadlag"]
        if not leadlag.empty:
            rankings["lead_lag"] = leadlag.nlargest(top_n, "abs_corr")

        return rankings
