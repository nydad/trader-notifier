"""Correlation analysis pipeline -- end-to-end orchestration."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from kospi_corr.domain.types import (
    CorrelationMethod, CorrelationRequest, DataSource,
    FetchRequest, SeriesDescriptor, SeriesKind,
)
from kospi_corr.data.watchlist import WatchlistLoader
from kospi_corr.data.normalizer import DataNormalizer
from kospi_corr.analysis.preprocessing import compute_returns, ensure_overlap
from kospi_corr.analysis.correlation.service import CorrelationRunService
from kospi_corr.visualization.heatmap import HeatmapRenderer
from kospi_corr.visualization.ranking_table import RankingTableRenderer

logger = logging.getLogger(__name__)


class CorrelationPipeline:
    """End-to-end: fetch -> align -> correlate -> rank -> visualize."""

    def __init__(
        self,
        watchlist_path: Path = Path("data/watchlist.json"),
        output_dir: Path = Path("output"),
        lookback_days: int = 90,
    ):
        self.watchlist_path = watchlist_path
        self.output_dir = output_dir
        self.lookback_days = lookback_days
        self._corr_service = CorrelationRunService()
        self._heatmap = HeatmapRenderer()
        self._ranking = RankingTableRenderer()

    def execute(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        methods: list[str] | None = None,
        rolling_windows: list[int] | None = None,
        max_lag: int = 5,
        partial_controls: list[str] | None = None,
    ) -> dict:
        end_date = end_date or date.today()
        start_date = start_date or (end_date - timedelta(days=self.lookback_days))
        logger.info(f"=== Correlation Pipeline: {start_date} to {end_date} ===")

        watchlist = WatchlistLoader().load(self.watchlist_path)
        etf_prices, indicators, lags = self._fetch_all(watchlist, start_date, end_date)

        normalizer = DataNormalizer()
        aligned = normalizer.align_to_krx_calendar(etf_prices, indicators, lags)
        returns = compute_returns(aligned, method="simple")
        returns, _ = ensure_overlap(returns, min_periods=20)

        request = CorrelationRequest(
            start=start_date, end=end_date,
            methods=tuple(CorrelationMethod(m) for m in (methods or ["pearson", "spearman"])),
            rolling_windows=tuple(rolling_windows or [5, 10, 20]),
            max_lag=max_lag,
            partial_controls=tuple(partial_controls or []),
        )
        result = self._corr_service.run(returns, request)
        rankings = self._corr_service.rank_pairs(result, top_n=20)
        artifacts = self._visualize(returns, rankings)
        return {"run_result": result, "rankings": rankings, "artifacts": artifacts}

    def _fetch_all(self, watchlist, start_date, end_date):
        etf_prices, indicators, lags = {}, {}, {}
        try:
            from kospi_corr.data.providers.krx import KRXFetcher
            krx = KRXFetcher()
            for item in watchlist.items:
                s = SeriesDescriptor(kind=SeriesKind.ETF_PRICE, symbol=item.code,
                    display_name=item.name, source=DataSource.PYKRX, source_symbol=item.code)
                try:
                    etf_prices[item.code] = krx.fetch(FetchRequest(series=s, start=start_date, end=end_date))
                except Exception as e:
                    logger.warning(f"ETF {item.code} fetch failed: {e}")
        except ImportError:
            logger.error("pykrx not installed")
        return etf_prices, indicators, lags

    def _visualize(self, returns, rankings):
        from kospi_corr.analysis.correlation.pearson_spearman import PearsonSpearmanCalculator
        calc = PearsonSpearmanCalculator()
        artifacts = {}
        for m in ["pearson", "spearman"]:
            mat = calc.matrix(returns, method=m)
            p = self.output_dir / f"corr_heatmap_{m}.png"
            self._heatmap.render_correlation_matrix(mat, p, title=f"{m.title()} Correlation")
            artifacts[f"{m}_heatmap"] = p
        for rt, df in rankings.items():
            p = self.output_dir / f"ranking_{rt}.png"
            self._ranking.render_correlation_ranking(df, p)
            artifacts[f"ranking_{rt}"] = p
        return artifacts
