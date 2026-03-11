"""FRED data provider for international indicators."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from kospi_corr.domain.errors import DataProviderError
from kospi_corr.domain.types import DataSource, FetchRequest, SeriesDescriptor

logger = logging.getLogger(__name__)


class FREDFetcher:
    """Data fetcher for FRED (Federal Reserve Economic Data)."""

    SOURCE = DataSource.FRED
    KNOWN_SERIES = {
        "DCOILWTICO", "DCOILBRENTEU", "SP500", "NASDAQCOM",
        "VIXCLS", "DTWEXBGS", "DEXKOUS", "NIKKEI225",
    }

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    def supports(self, series: SeriesDescriptor) -> bool:
        return series.source == DataSource.FRED

    def fetch(self, req: FetchRequest) -> pd.DataFrame:
        symbol = req.series.source_symbol
        try:
            if self._api_key:
                return self._fetch_via_fredapi(symbol, req.start, req.end)
            else:
                return self._fetch_via_datareader(symbol, req.start, req.end)
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(
                f"FRED fetch failed for {symbol}: {exc}",
                source="fred", symbol=symbol,
            ) from exc

    def _fetch_via_fredapi(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from fredapi import Fred
        fred = Fred(api_key=self._api_key)
        series = fred.get_series(symbol, observation_start=start.isoformat(),
                                  observation_end=end.isoformat())
        if series is None or series.empty:
            raise DataProviderError(f"No FRED data for {symbol}", source="fred", symbol=symbol)
        df = series.to_frame(name="value").dropna()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    def _fetch_via_datareader(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import pandas_datareader.data as web
            df = web.DataReader(symbol, "fred", start, end)
            if df.empty:
                raise DataProviderError(f"No data for {symbol}", source="fred", symbol=symbol)
            df.columns = ["value"]
            df.index.name = "date"
            return df.dropna()
        except ImportError:
            raise DataProviderError("pandas_datareader not installed", source="fred")
