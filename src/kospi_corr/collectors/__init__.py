"""Collectors package — data ingestion from external sources."""
from kospi_corr.collectors.base import BaseCollector
from kospi_corr.collectors.news import NewsCollector

__all__ = ["BaseCollector", "NewsCollector"]
