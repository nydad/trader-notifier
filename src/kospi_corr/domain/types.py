"""Core domain types and enumerations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, StrEnum
from typing import Literal


class AssetCategory(StrEnum):
    CORE_LONG = "core_long"
    CORE_SHORT = "core_short"
    SECTOR_ETF = "sector_etf"


class SeriesKind(StrEnum):
    ETF_PRICE = "etf_price"
    INDICATOR = "indicator"


class IndicatorCategory(StrEnum):
    COMMODITY = "commodity"
    FX = "fx"
    KRX_DERIVATIVE = "krx_derivative"
    KRX_FLOW = "krx_flow"
    GLOBAL_INDEX = "global_index"
    VOLATILITY = "volatility"


class CorrelationMethod(StrEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    ROLLING_PEARSON = "rolling_pearson"
    ROLLING_SPEARMAN = "rolling_spearman"
    LEAD_LAG = "leadlag"
    PARTIAL = "partial"


class DataSource(StrEnum):
    PYKRX = "pykrx"
    FDR = "fdr"
    FRED = "fred"
    YFINANCE = "yfinance"


class SignalDirection(int, Enum):
    SHORT = -1
    NEUTRAL = 0
    LONG = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatchlistItem:
    """A single ETF/ETN in the watchlist."""
    code: str
    name: str
    category: AssetCategory


@dataclass(frozen=True)
class Watchlist:
    """Complete watchlist loaded from JSON."""
    items: tuple[WatchlistItem, ...]
    strategy: str
    updated_at: date

    @property
    def codes(self) -> list[str]:
        return [item.code for item in self.items]

    def by_category(self, cat: AssetCategory) -> list[WatchlistItem]:
        return [item for item in self.items if item.category == cat]


@dataclass(frozen=True)
class SeriesDescriptor:
    """Metadata for any time-series (ETF price or indicator)."""
    kind: SeriesKind
    symbol: str
    display_name: str
    source: DataSource
    source_symbol: str
    frequency: str = "daily"
    timezone: str = "Asia/Seoul"
    unit: str | None = None
    lag_days: int = 0  # For look-ahead bias prevention


@dataclass(frozen=True)
class PricePoint:
    as_of: date
    open: float | None
    high: float | None
    low: float | None
    close: float
    adj_close: float | None = None
    volume: float | None = None


@dataclass(frozen=True)
class IndicatorPoint:
    as_of: date
    value: float


@dataclass(frozen=True)
class SeriesFingerprint:
    """Cache validation fingerprint for a stored series."""
    series_id: int
    row_count: int
    earliest: date
    latest: date
    value_hash: str


@dataclass(frozen=True)
class FetchRequest:
    """Request to fetch data for one series."""
    series: SeriesDescriptor
    start: date
    end: date
    force: bool = False


@dataclass
class IngestSummary:
    series_symbol: str
    source: DataSource
    rows_written: int
    status: Literal["success", "failed", "partial"]
    error: str | None = None
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Correlation domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelationRequest:
    """Parameters for a correlation analysis run."""
    start: date
    end: date
    methods: tuple[CorrelationMethod, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
    )
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    max_lag: int = 5
    partial_controls: tuple[str, ...] = ()
    min_periods: int = 20


@dataclass
class CorrelationPair:
    """Result for one (series_a, series_b, method, window, lag) combination."""
    series_a: str
    series_b: str
    method: CorrelationMethod
    window: int | None
    lag: int
    correlation: float
    p_value: float | None
    n_obs: int


@dataclass
class CorrelationRunResult:
    """Aggregated result of a full correlation run."""
    run_id: int
    request: CorrelationRequest
    pairs: list[CorrelationPair]
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Backtest domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalCondition:
    """A single indicator condition, e.g. 'WTI daily change > 0'."""
    indicator: str
    operator: Literal["gt", "lt", "gte", "lte", "eq", "change_gt", "change_lt"]
    threshold: float


@dataclass(frozen=True)
class SignalCombination:
    """AND-combined set of conditions that generate a trade signal."""
    conditions: tuple[SignalCondition, ...]
    direction: SignalDirection
    label: str

    @property
    def key(self) -> str:
        parts = [f"{c.indicator}_{c.operator}_{c.threshold}" for c in self.conditions]
        return f"{self.direction.name}:{'&'.join(sorted(parts))}"


@dataclass(frozen=True)
class BacktestParams:
    initial_capital: float = 100_000_000
    position_budget_ratio: float = 0.1
    slippage_bps: float = 10
    commission_bps: float = 3
    hold_cost_bps: float = 0
    allow_short: bool = True


@dataclass
class Trade:
    series_symbol: str
    direction: SignalDirection
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    qty: float = 0
    pnl: float = 0
    return_pct: float = 0
    holding_days: int = 0
    exit_reason: str = ""


@dataclass
class BacktestMetrics:
    """Summary statistics for one signal combination backtest."""
    combo_key: str
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_return: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_holding_days: float
