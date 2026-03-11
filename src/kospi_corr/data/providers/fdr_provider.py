"""FinanceDataReader provider for supplementary data.

Handles: Nikkei 225, Shanghai Composite, and fallback for other series.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from kospi_corr.domain.errors import DataProviderError
from kospi_corr.domain.types import DataSource, FetchRequest, SeriesDescriptor

logger = logging.getLogger(__name__)


class FDRFetcher:
    """Data fetcher using FinanceDataReader."""

    SOURCE = DataSource.FDR

    def supports(self, series: SeriesDescriptor) -> bool:
        return series.source == DataSource.FDR

    def fetch(self, req: FetchRequest) -> pd.DataFrame:
        symbol = req.series.source_symbol
        try:
            import FinanceDataReader as fdr

            df = fdr.DataReader(symbol, req.start, req.end)
            if df.empty:
                raise DataProviderError(
                    f"No FDR data for {symbol}",
                    source="fdr", symbol=symbol,
                )

            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            # For index data, use Close column as the value
            if "Close" in df.columns:
                return df[["Close"]].rename(columns={"Close": "value"})
            elif "close" in df.columns:
                return df[["close"]].rename(columns={"close": "value"})
            else:
                result = df.iloc[:, 0:1].copy()
                result.columns = ["value"]
                return result

        except ImportError:
            raise DataProviderError(
                "FinanceDataReader not installed", source="fdr", symbol=symbol
            )
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(
                f"FDR fetch failed for {symbol}: {exc}",
                source="fdr", symbol=symbol,
            ) from exc
