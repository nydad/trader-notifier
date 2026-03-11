"""yfinance provider — free fallback for global data (oil, FX, US indices)."""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from kospi_corr.domain.errors import DataProviderError
from kospi_corr.domain.types import DataSource, FetchRequest, SeriesDescriptor

logger = logging.getLogger(__name__)

# Map FRED symbols → yfinance tickers
_FRED_TO_YF: dict[str, str] = {
    "DCOILWTICO": "CL=F",
    "DCOILBRENTEU": "BZ=F",
    "SP500": "^GSPC",
    "NASDAQCOM": "^IXIC",
    "VIXCLS": "^VIX",
    "DTWEXBGS": "DX-Y.NYB",
    "DEXKOUS": "KRW=X",
}


class YFinanceFetcher:
    """Fetch data via yfinance — works without API keys."""

    SOURCE = DataSource.YFINANCE

    def supports(self, series: SeriesDescriptor) -> bool:
        return series.source == DataSource.YFINANCE

    def fetch(self, req: FetchRequest) -> pd.DataFrame:
        symbol = req.series.source_symbol
        yf_ticker = _FRED_TO_YF.get(symbol, symbol)

        try:
            import yfinance as yf

            time.sleep(0.5)
            ticker = yf.Ticker(yf_ticker)
            df = ticker.history(
                start=req.start.isoformat(),
                end=req.end.isoformat(),
                auto_adjust=True,
            )

            if df.empty:
                raise DataProviderError(
                    f"No yfinance data for {yf_ticker}",
                    source="yfinance",
                    symbol=yf_ticker,
                )

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"

            if "Close" in df.columns:
                return df[["Close"]].rename(columns={"Close": "value"})
            elif "close" in df.columns:
                return df[["close"]].rename(columns={"close": "value"})
            else:
                result = df.iloc[:, :1].copy()
                result.columns = ["value"]
                return result

        except ImportError:
            raise DataProviderError("yfinance not installed", source="yfinance")
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(
                f"yfinance fetch failed for {yf_ticker}: {exc}",
                source="yfinance",
                symbol=yf_ticker,
            ) from exc
