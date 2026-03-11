"""Orchestration layer — ties data collection, normalization, and analysis together."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from kospi_corr.config import (
    load_app_settings,
    load_indicator_descriptors,
    project_root,
)
from kospi_corr.data.normalizer import DataNormalizer
from kospi_corr.data.providers.fdr_provider import FDRFetcher
from kospi_corr.data.providers.krx import KRXFetcher
from kospi_corr.data.providers.yfinance_provider import YFinanceFetcher
from kospi_corr.data.watchlist import WatchlistLoader
from kospi_corr.domain.errors import DataProviderError
from kospi_corr.domain.types import (
    CorrelationMethod,
    CorrelationRequest,
    DataSource,
    FetchRequest,
    SeriesDescriptor,
    SeriesKind,
    Watchlist,
)
from kospi_corr.analysis.correlation.service import CorrelationRunService
from kospi_corr.analysis.preprocessing import compute_returns, ensure_overlap

logger = logging.getLogger(__name__)


class MarketOrchestrator:
    """Main pipeline: fetch → normalize → correlate → output."""

    def __init__(self, lookback_days: int | None = None) -> None:
        self.settings = load_app_settings()
        self.indicators = load_indicator_descriptors()
        self.lookback = lookback_days or self.settings.data.lookback_days

        # providers
        self._krx = KRXFetcher()
        self._fdr = FDRFetcher()
        self._yf = YFinanceFetcher()

        self._normalizer = DataNormalizer()
        self._corr_service = CorrelationRunService()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _pick_fetcher(self, source: DataSource):
        if source == DataSource.PYKRX:
            return self._krx
        if source == DataSource.FDR:
            return self._fdr
        if source == DataSource.FRED and self.settings.fred_api_key:
            from kospi_corr.data.providers.fred_provider import FREDFetcher
            return FREDFetcher(api_key=self.settings.fred_api_key)
        return self._yf  # FRED without key / YFINANCE → yfinance

    def load_watchlist(self) -> Watchlist:
        loader = WatchlistLoader()
        wl_path = project_root() / "data" / "watchlist.json"
        loader.EXPECTED_COUNT = None  # disable strict count check
        return loader.load(wl_path)

    def fetch_etf_prices(
        self, codes: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for each ETF code. Tries pykrx first, falls back to FDR."""
        results: dict[str, pd.DataFrame] = {}
        for code in codes:
            # Try pykrx first
            desc = SeriesDescriptor(
                kind=SeriesKind.ETF_PRICE,
                symbol=code,
                display_name=code,
                source=DataSource.PYKRX,
                source_symbol=code,
            )
            req = FetchRequest(series=desc, start=start, end=end)
            try:
                df = self._krx.fetch(req)
                if not df.empty:
                    results[code] = df
                    logger.info(f"ETF {code}: {len(df)} rows (pykrx)")
                    continue
            except DataProviderError:
                pass

            # Fallback to FinanceDataReader
            try:
                import FinanceDataReader as fdr
                import time
                time.sleep(0.3)
                df = fdr.DataReader(code, start, end)
                if not df.empty:
                    col_map = {"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"}
                    df = df.rename(columns=col_map)
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "date"
                    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                    results[code] = df[cols].copy()
                    logger.info(f"ETF {code}: {len(df)} rows (FDR fallback)")
                    continue
            except Exception as e2:
                logger.warning(f"ETF {code} FDR fallback failed: {e2}")

            logger.warning(f"ETF {code}: no data from any source")
        return results

    def fetch_indicators(
        self, start: date, end: date
    ) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
        """Fetch all configured indicators. Returns (data_dict, lag_dict)."""
        data: dict[str, pd.DataFrame] = {}
        lags: dict[str, int] = {}

        for key, desc in self.indicators.items():
            fetcher = self._pick_fetcher(desc.source)
            req = FetchRequest(series=desc, start=start, end=end)

            try:
                df = fetcher.fetch(req)
                if not df.empty:
                    data[key] = df
                    lags[key] = desc.lag_days
                    logger.info(f"Indicator {key}: {len(df)} rows (lag={desc.lag_days})")
            except DataProviderError as e:
                logger.warning(f"Indicator {key} ({desc.source}) failed: {e}")
                # Try yfinance as fallback
                if desc.source != DataSource.YFINANCE:
                    try:
                        yf_req = FetchRequest(series=desc, start=start, end=end)
                        df = self._yf.fetch(yf_req)
                        if not df.empty:
                            data[key] = df
                            lags[key] = desc.lag_days
                            logger.info(f"Indicator {key}: {len(df)} rows (yfinance fallback)")
                    except DataProviderError:
                        logger.warning(f"Indicator {key} yfinance fallback also failed")

        return data, lags

    # ------------------------------------------------------------------
    # Analysis pipeline
    # ------------------------------------------------------------------

    def run_correlation(
        self,
        lookback_days: int | None = None,
        methods: tuple[str, ...] = ("pearson", "spearman"),
        rolling_windows: tuple[int, ...] = (5, 10, 20),
        max_lag: int = 5,
    ) -> dict:
        """Full correlation analysis pipeline.

        Returns dict with:
          - 'aligned': aligned DataFrame
          - 'returns': returns DataFrame
          - 'result': CorrelationRunResult
          - 'rankings': ranked pairs dict
          - 'matrices': {method: correlation matrix DataFrame}
        """
        lookback = lookback_days or self.lookback
        end = date.today()
        start = end - timedelta(days=lookback)

        logger.info(f"=== Correlation Analysis: {start} → {end} ({lookback} days) ===")

        # 1. Load watchlist
        try:
            watchlist = self.load_watchlist()
            etf_codes = watchlist.codes
            logger.info(f"Watchlist: {len(etf_codes)} ETFs")
        except Exception as e:
            logger.warning(f"Watchlist load failed, using defaults: {e}")
            etf_codes = ["122630", "252670", "233740", "091170"]

        # 2. Fetch data
        logger.info("Fetching ETF prices...")
        etf_prices = self.fetch_etf_prices(etf_codes, start, end)

        logger.info("Fetching indicators...")
        indicator_data, indicator_lags = self.fetch_indicators(start, end)

        if not etf_prices:
            raise DataProviderError("No ETF data fetched", source="pipeline")

        logger.info(f"Data collected: {len(etf_prices)} ETFs, {len(indicator_data)} indicators")

        # 3. Normalize & align
        aligned = self._normalizer.align_to_krx_calendar(
            etf_prices, indicator_data, indicator_lags
        )
        logger.info(f"Aligned DataFrame: {aligned.shape}")

        # 4. Compute returns
        returns = compute_returns(aligned, method="simple")
        returns, dropped = ensure_overlap(returns, min_periods=self.settings.correlation.min_periods)
        if dropped:
            logger.warning(f"Dropped columns with insufficient data: {dropped}")
        logger.info(f"Returns DataFrame: {returns.shape}")

        # 5. Run correlation
        corr_methods = tuple(CorrelationMethod(m) for m in methods)
        request = CorrelationRequest(
            start=start,
            end=end,
            methods=corr_methods,
            rolling_windows=rolling_windows,
            max_lag=max_lag,
            min_periods=self.settings.correlation.min_periods,
        )

        result = self._corr_service.run(returns, request)
        rankings = self._corr_service.rank_pairs(result)

        # 6. Build correlation matrices for visualization
        from kospi_corr.analysis.correlation.pearson_spearman import PearsonSpearmanCalculator
        calc = PearsonSpearmanCalculator()
        matrices = {}
        for m in methods:
            matrices[m] = calc.matrix(returns, method=m, min_periods=self.settings.correlation.min_periods)

        logger.info(f"Correlation run complete: {len(result.pairs)} pairs")

        return {
            "aligned": aligned,
            "returns": returns,
            "result": result,
            "rankings": rankings,
            "matrices": matrices,
            "etf_codes": list(etf_prices.keys()),
            "indicator_keys": list(indicator_data.keys()),
            "start": start,
            "end": end,
        }
