"""Data preprocessing for correlation and backtesting analysis."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute daily returns from price/value levels.

    Args:
        df: DataFrame with price/value columns
        method: 'simple' for arithmetic returns, 'log' for log returns

    Returns:
        DataFrame of returns (first row dropped as NaN).
    """
    if method == "log":
        returns = np.log(df / df.shift(1))
    else:
        returns = df.pct_change()
    return returns.iloc[1:]


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Run ADF test for stationarity.

    Returns dict with test statistic, p-value, and is_stationary flag.
    """
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 20:
        return {"adf_stat": None, "p_value": None, "is_stationary": None, "n": len(clean)}

    result = adfuller(clean, autolag="AIC")
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < significance,
        "n": len(clean),
    }


def winsorize_returns(
    df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """Clip extreme returns to reduce outlier influence on correlations."""
    result = df.copy()
    for col in result.columns:
        lo, hi = result[col].quantile([lower, upper])
        result[col] = result[col].clip(lo, hi)
    return result


def ensure_overlap(
    df: pd.DataFrame, min_periods: int = 20
) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns with insufficient overlap and return cleaned DataFrame.

    Returns:
        (cleaned_df, list of dropped column names)
    """
    valid_counts = df.notna().sum()
    dropped = valid_counts[valid_counts < min_periods].index.tolist()
    if dropped:
        logger.warning(
            f"Dropping {len(dropped)} columns with < {min_periods} observations: {dropped}"
        )
    kept = df.drop(columns=dropped)
    return kept, dropped
