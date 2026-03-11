"""Custom exception hierarchy for the market analysis system."""
from __future__ import annotations


class MarketSystemError(Exception):
    """Base exception for the entire system."""
    pass


# --- Configuration ---
class ConfigError(MarketSystemError):
    """Invalid or missing configuration."""
    pass


class WatchlistError(MarketSystemError):
    """Problem loading or validating the watchlist."""
    pass


# --- Data Layer ---
class DataProviderError(MarketSystemError):
    """Error from an external data provider (pykrx, FRED, etc.)."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        symbol: str | None = None,
        http_status: int | None = None,
    ):
        self.source = source
        self.symbol = symbol
        self.http_status = http_status
        super().__init__(message)


class NormalizationError(MarketSystemError):
    """Error during data alignment or normalization."""
    pass


class IngestionError(MarketSystemError):
    """Error during data ingestion into storage."""
    pass


# --- Storage ---
class StorageError(MarketSystemError):
    """Database / file storage error."""
    pass


# --- Analysis ---
class CorrelationError(MarketSystemError):
    """Error during correlation calculation."""
    pass


class InsufficientDataError(CorrelationError):
    """Not enough overlapping data points for calculation."""

    def __init__(self, message: str, available: int = 0, required: int = 0):
        self.available = available
        self.required = required
        super().__init__(message)


# --- Backtest ---
class BacktestError(MarketSystemError):
    """Error during backtesting."""
    pass


class SignalDefinitionError(BacktestError):
    """Invalid signal combination definition."""
    pass


# --- Visualization ---
class VisualizationError(MarketSystemError):
    """Error during chart/report generation."""
    pass
