"""Pydantic models for configuration validation and serialization."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    path: str = "data/market.db"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    foreign_keys: bool = True
    busy_timeout_ms: int = 5000


class CorrelationConfig(BaseModel):
    methods: list[str] = ["pearson", "spearman"]
    rolling_windows: list[int] = [5, 10, 20]
    lead_lag_max: int = 5
    granger_max_lag: int = 5
    granger_significance: float = 0.05
    partial_control_groups: dict[str, list[str]] = Field(default_factory=dict)
    min_periods: int = 20
    cache_ttl_days: dict[str, int] = Field(
        default_factory=lambda: {"static": 1, "rolling": 1, "lead_lag": 2, "partial": 7}
    )


class BacktestConfig(BaseModel):
    default_capital: float = 100_000_000
    position_budget_ratio: float = 0.1
    slippage_bps: float = 10
    commission_bps: float = 3
    hold_cost_bps: float = 0
    max_signal_depth: int = 3
    min_signal_observations: int = 20
    allow_short: bool = True


class DataConfig(BaseModel):
    lookback_days: int = 90
    min_lookback_days: int = 40
    timezone: str = "Asia/Seoul"
    krx_open: str = "09:00"
    krx_close: str = "15:30"


class VisualizationConfig(BaseModel):
    output_dir: str = "output/"
    dpi: int = 150
    figsize: tuple[int, int] = (16, 12)
    heatmap_cmap: str = "RdBu_r"
    font_family: str = "Malgun Gothic"


class AppSettings(BaseSettings):
    """Top-level application settings, loadable from env + yaml."""
    env: str = "dev"
    log_level: str = "INFO"
    log_format: str = "json"
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    fred_api_key: str = ""

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


class IndicatorDefinition(BaseModel):
    """Schema for one indicator in indicators.yaml."""
    display_name: str
    category: str
    source: str
    source_symbol: str
    frequency: str = "daily"
    timezone: str = "America/New_York"
    unit: str | None = None
    lag_days: int = 0
    fallback_source: str | None = None
    fallback_symbol: str | None = None
    notes: str | None = None
