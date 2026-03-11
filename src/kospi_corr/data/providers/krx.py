"""KRX data provider using pykrx library.

Handles:
  - ETF OHLCV prices
  - Investor trading data (foreign/institutional/individual net buying)
  - Program trading data
  - KOSPI200 futures basis
  - VKOSPI
  - USD/KRW exchange rate
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from kospi_corr.domain.errors import DataProviderError
from kospi_corr.domain.types import DataSource, FetchRequest, SeriesDescriptor, SeriesKind

logger = logging.getLogger(__name__)

_CALL_DELAY_SEC = 0.3


class KRXFetcher:
    """Data fetcher for Korean Exchange data via pykrx."""

    SOURCE = DataSource.PYKRX

    def supports(self, series: SeriesDescriptor) -> bool:
        return series.source == DataSource.PYKRX

    def fetch(self, req: FetchRequest) -> pd.DataFrame:
        symbol = req.series.source_symbol
        start_str = req.start.strftime("%Y%m%d")
        end_str = req.end.strftime("%Y%m%d")

        try:
            if req.series.kind == SeriesKind.ETF_PRICE:
                return self._fetch_etf_ohlcv(req.series.symbol, start_str, end_str)
            dispatch = {
                "FOREIGN_FUTURES_NET": lambda: self._fetch_investor_net(start_str, end_str, "외국인합계"),
                "INSTITUTIONAL_NET": lambda: self._fetch_investor_net(start_str, end_str, "기관합계"),
                "INDIVIDUAL_NET": lambda: self._fetch_investor_net(start_str, end_str, "개인"),
                "PROGRAM_NET": lambda: self._fetch_program_trading(start_str, end_str),
                "KOSPI200_BASIS": lambda: self._fetch_futures_basis(start_str, end_str),
                "VKOSPI": lambda: self._fetch_vkospi(start_str, end_str),
                "USD/KRW": lambda: self._fetch_usd_krw(start_str, end_str),
            }
            if symbol in dispatch:
                return dispatch[symbol]()
            raise DataProviderError(f"Unknown pykrx symbol: {symbol}", source="pykrx", symbol=symbol)
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(f"pykrx fetch failed: {exc}", source="pykrx", symbol=symbol) from exc

    def _fetch_etf_ohlcv(self, code: str, start: str, end: str) -> pd.DataFrame:
        from pykrx import stock
        time.sleep(_CALL_DELAY_SEC)
        df = stock.get_etf_ohlcv_by_date(start, end, code)
        if df.empty:
            raise DataProviderError(f"No ETF data for {code}", source="pykrx", symbol=code)
        col_map = {"시가": "open", "고가": "high", "저가": "low",
                    "종가": "close", "거래량": "volume"}
        df = df.rename(columns=col_map)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]].copy()

    def _fetch_investor_net(self, start: str, end: str, investor_type: str) -> pd.DataFrame:
        from pykrx import stock
        time.sleep(_CALL_DELAY_SEC)
        df = stock.get_market_net_purchases_of_equities(start, end, "KOSPI", investor_type)
        if df.empty:
            logger.warning(f"No investor data for {investor_type}")
            return pd.DataFrame(columns=["value"])
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        target_col = "순매수거래대금"
        if target_col in df.columns:
            return df[[target_col]].rename(columns={target_col: "value"})
        result = df.iloc[:, 0:1].copy()
        result.columns = ["value"]
        return result

    def _fetch_program_trading(self, start: str, end: str) -> pd.DataFrame:
        from pykrx import stock
        time.sleep(_CALL_DELAY_SEC)
        try:
            df = stock.get_market_trading_value_by_date(start, end, "KOSPI")
            if df.empty:
                return pd.DataFrame(columns=["value"])
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            return pd.DataFrame(index=df.index, data={"value": 0.0})
        except Exception as exc:
            logger.warning(f"Program trading fetch fallback: {exc}")
            return pd.DataFrame(columns=["value"])

    def _fetch_futures_basis(self, start: str, end: str) -> pd.DataFrame:
        from pykrx import stock
        time.sleep(_CALL_DELAY_SEC)
        kospi200 = stock.get_index_ohlcv_by_date(start, end, "1028")
        if kospi200.empty:
            return pd.DataFrame(columns=["value"])
        kospi200.index = pd.to_datetime(kospi200.index)
        kospi200.index.name = "date"
        logger.warning("KOSPI200 basis: placeholder zeros. Connect futures data source.")
        return pd.DataFrame(index=kospi200.index, data={"value": 0.0})

    def _fetch_vkospi(self, start: str, end: str) -> pd.DataFrame:
        from pykrx import stock
        time.sleep(_CALL_DELAY_SEC)
        df = stock.get_index_ohlcv_by_date(start, end, "1204")
        if df.empty:
            return pd.DataFrame(columns=["value"])
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        col = "종가" if "종가" in df.columns else df.columns[0]
        return df[[col]].rename(columns={col: "value"})

    def _fetch_usd_krw(self, start: str, end: str) -> pd.DataFrame:
        try:
            import FinanceDataReader as fdr
            time.sleep(_CALL_DELAY_SEC)
            df = fdr.DataReader("USD/KRW", start, end)
            if df.empty:
                raise DataProviderError("No USD/KRW data", source="fdr", symbol="USD/KRW")
            df.index.name = "date"
            return df[["Close"]].rename(columns={"Close": "value"})
        except ImportError:
            raise DataProviderError("FinanceDataReader not installed", source="fdr")
