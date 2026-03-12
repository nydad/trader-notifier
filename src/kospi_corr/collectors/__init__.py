"""Collectors package — data ingestion from external sources."""
from kospi_corr.collectors.base import BaseCollector
from kospi_corr.collectors.domestic import DomesticSnapshot, fetch_domestic_snapshot
from kospi_corr.collectors.naver_realtime import (
    InvestorFlowIntraday,
    NaverIndexData,
    NaverSummaryData,
    ProgramTradingData,
    fetch_investor_flow_intraday,
    fetch_main_summary,
    fetch_naver_index,
    fetch_program_trading,
)
from kospi_corr.collectors.news import NewsCollector

__all__ = [
    "BaseCollector",
    "DomesticSnapshot",
    "InvestorFlowIntraday",
    "NaverIndexData",
    "NewsCollector",
    "ProgramTradingData",
    "fetch_domestic_snapshot",
    "fetch_investor_flow_intraday",
    "fetch_main_summary",
    "fetch_naver_index",
    "fetch_program_trading",
]
