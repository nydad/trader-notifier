"""Configuration loader — reads YAML files and produces validated Pydantic models."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from kospi_corr.domain.errors import ConfigError
from kospi_corr.domain.models import AppSettings, IndicatorDefinition
from kospi_corr.domain.types import (
    DataSource,
    SeriesDescriptor,
    SeriesKind,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # market/


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_app_settings(config_dir: Path | None = None) -> AppSettings:
    """Load app.yaml and merge with environment."""
    config_dir = config_dir or _PROJECT_ROOT / "config"
    raw = _load_yaml(config_dir / "app.yaml")

    app_raw = raw.get("app", {})
    db_raw = raw.get("database", {})
    corr_raw = raw.get("correlation", {})
    bt_raw = raw.get("backtest", {})
    data_raw = raw.get("data", {})
    vis_raw = raw.get("visualization", {})

    # Flatten lead_lag / partial sub-dicts
    if "lead_lag" in corr_raw:
        ll = corr_raw.pop("lead_lag")
        corr_raw["lead_lag_max"] = ll.get("max_lag", 5)
        corr_raw["granger_max_lag"] = ll.get("granger_max_lag", 5)
        corr_raw["granger_significance"] = ll.get("granger_significance", 0.05)
    if "partial" in corr_raw:
        corr_raw["partial_control_groups"] = corr_raw.pop("partial").get("control_groups", {})
    if "cache_ttl_days" not in corr_raw:
        corr_raw["cache_ttl_days"] = {"static": 1, "rolling": 1, "lead_lag": 2, "partial": 7}

    fred_key = os.environ.get("FRED_API_KEY", "")

    return AppSettings(
        env=app_raw.get("env", "dev"),
        log_level=app_raw.get("log_level", "INFO"),
        log_format=app_raw.get("log_format", "json"),
        db=db_raw,
        correlation=corr_raw,
        backtest=bt_raw,
        data=data_raw,
        visualization=vis_raw,
        fred_api_key=fred_key,
    )


def load_indicator_descriptors(
    config_dir: Path | None = None,
) -> dict[str, SeriesDescriptor]:
    """Load indicators.yaml and return {key: SeriesDescriptor} mapping."""
    config_dir = config_dir or _PROJECT_ROOT / "config"
    raw = _load_yaml(config_dir / "indicators.yaml")
    indicators_raw = raw.get("indicators", {})

    descriptors: dict[str, SeriesDescriptor] = {}

    for key, defn in indicators_raw.items():
        source_str = defn.get("source", "yfinance")
        try:
            source = DataSource(source_str)
        except ValueError:
            source = DataSource.YFINANCE

        descriptors[key] = SeriesDescriptor(
            kind=SeriesKind.INDICATOR,
            symbol=key,
            display_name=defn.get("display_name", key),
            source=source,
            source_symbol=defn.get("source_symbol", key),
            frequency=defn.get("frequency", "daily"),
            timezone=defn.get("timezone", "Asia/Seoul"),
            unit=defn.get("unit"),
            lag_days=defn.get("lag_days", 0),
        )

    return descriptors


def project_root() -> Path:
    return _PROJECT_ROOT
