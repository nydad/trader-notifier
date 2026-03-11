"""Partial correlation calculator.

Computes correlation between two variables while controlling for
the effect of other variables. Critical for isolating the true
relationship between e.g. WTI and KOSPI after removing the
influence of USD/KRW, VIX, etc.

Uses pingouin library for robust partial correlation with p-values.
Falls back to precision-matrix method if pingouin is unavailable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from kospi_corr.domain.types import CorrelationMethod, CorrelationPair

logger = logging.getLogger(__name__)


class PartialCorrelationCalculator:
    """Calculate partial correlations controlling for confounders."""

    def pairwise_partial(
        self,
        df: pd.DataFrame,
        controls: list[str],
        min_periods: int = 20,
    ) -> list[CorrelationPair]:
        """Compute partial correlation for all non-control column pairs.

        Args:
            df: DataFrame with all columns (targets + controls)
            controls: Column names to control for
            min_periods: Minimum observations required

        Returns:
            List of CorrelationPair with partial correlations.
        """
        target_cols = [c for c in df.columns if c not in controls]
        pairs: list[CorrelationPair] = []

        for i, col_a in enumerate(target_cols):
            for col_b in target_cols[i + 1:]:
                result = self._partial_corr_pair(
                    df, col_a, col_b, controls, min_periods
                )
                if result is not None:
                    pairs.append(result)

        return pairs

    def _partial_corr_pair(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        covars: list[str],
        min_periods: int,
    ) -> CorrelationPair | None:
        """Compute partial correlation between x and y controlling for covars."""
        needed = [x, y] + covars
        available = [c for c in needed if c in df.columns]
        if len(available) < len(needed):
            missing = set(needed) - set(available)
            logger.debug(f"Skipping partial corr {x}-{y}: missing {missing}")
            return None

        sub = df[available].dropna()
        if len(sub) < min_periods:
            return None

        try:
            r, p, n = self._compute_partial(sub, x, y, covars)
        except Exception as exc:
            logger.warning(f"Partial correlation failed for {x}-{y}: {exc}")
            return None

        return CorrelationPair(
            series_a=x,
            series_b=y,
            method=CorrelationMethod.PARTIAL,
            window=None,
            lag=0,
            correlation=float(r),
            p_value=float(p) if p is not None else None,
            n_obs=int(n),
        )

    def _compute_partial(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        covars: list[str],
    ) -> tuple[float, float | None, int]:
        """Attempt pingouin first, fall back to precision-matrix method."""
        try:
            return self._pingouin_partial(df, x, y, covars)
        except ImportError:
            logger.debug("pingouin not available, using precision-matrix method")
            return self._precision_matrix_partial(df, x, y, covars)

    @staticmethod
    def _pingouin_partial(
        df: pd.DataFrame, x: str, y: str, covars: list[str]
    ) -> tuple[float, float, int]:
        """Use pingouin.partial_corr for robust computation with p-values."""
        import pingouin as pg

        result = pg.partial_corr(data=df, x=x, y=y, covar=covars, method="pearson")
        r = result["r"].values[0]
        p = result["p-val"].values[0]
        n = result["n"].values[0]
        return r, p, n

    @staticmethod
    def _precision_matrix_partial(
        df: pd.DataFrame, x: str, y: str, covars: list[str]
    ) -> tuple[float, None, int]:
        """Fallback: residual-regression approach for partial correlation.

        Regress x and y each on covars, then correlate residuals.
        """
        from numpy.linalg import lstsq

        all_cols = [x, y] + covars
        sub = df[all_cols].dropna()
        n = len(sub)

        X_covars = sub[covars].values
        X_design = np.column_stack([np.ones(n), X_covars])

        # Residualize x
        x_vals = sub[x].values
        beta_x, _, _, _ = lstsq(X_design, x_vals, rcond=None)
        resid_x = x_vals - X_design @ beta_x

        # Residualize y
        y_vals = sub[y].values
        beta_y, _, _, _ = lstsq(X_design, y_vals, rcond=None)
        resid_y = y_vals - X_design @ beta_y

        # Correlate residuals
        r = np.corrcoef(resid_x, resid_y)[0, 1]
        return float(r), None, n
