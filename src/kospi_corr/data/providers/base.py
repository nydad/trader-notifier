"""Base protocol for all market data fetchers."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from kospi_corr.domain.types import FetchRequest, SeriesDescriptor


@runtime_checkable
class MarketDataFetcher(Protocol):
    """Protocol that all data provider adapters must implement."""

    def supports(self, series: SeriesDescriptor) -> bool:
        """Return True if this fetcher can handle the given series."""
        ...

    def fetch(self, req: FetchRequest) -> pd.DataFrame:
        """Fetch data for the given request.

        Returns a DataFrame with:
          - DatetimeIndex (timezone-aware, normalized to date)
          - For ETF prices: columns [open, high, low, close, adj_close, volume]
          - For indicators: column [value]

        Raises DataProviderError on failure.
        """
        ...
