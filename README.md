# KOSPI ETF Correlation Analysis & Backtesting System

## Overview

Short-term trading tool for Korean ETF/ETN instruments that analyzes correlation
between 12 watchlist stocks and 15 market indicators, then backtests signal
combinations to identify high-probability trading setups.

## Architecture

```
E:/workspace/market/
+-- config/
|   +-- app.yaml                 # App configuration
|   +-- indicators.yaml          # 15 indicator definitions with data sources
|   +-- logging.yaml             # Logging configuration
+-- data/
|   +-- watchlist.json           # 12 ETF/ETN watchlist (existing)
|   +-- market.db                # SQLite database (generated)
+-- output/                      # Generated charts and reports
+-- scripts/
|   +-- init_db.py               # Database initialization
+-- src/kospi_corr/
|   +-- domain/
|   |   +-- types.py             # Core types: Watchlist, Series, Correlation, Signal, Trade
|   |   +-- models.py            # Pydantic settings and validation models
|   |   +-- errors.py            # Exception hierarchy
|   +-- data/
|   |   +-- watchlist.py         # Watchlist JSON loader
|   |   +-- normalizer.py        # Timezone-aware data alignment (anti look-ahead bias)
|   |   +-- providers/
|   |       +-- base.py          # MarketDataFetcher protocol
|   |       +-- krx.py           # pykrx: ETF prices, investor flow, VKOSPI, FX
|   |       +-- fred_provider.py # FRED: WTI, Brent, S&P500, NASDAQ, VIX, DXY
|   |       +-- fdr_provider.py  # FinanceDataReader: Nikkei, Shanghai, fallbacks
|   +-- storage/
|   |   +-- sqlite/
|   |       +-- migrations/
|   |       |   +-- 0001_init.sql   # Complete DB schema (17 tables)
|   |       +-- repositories/       # Data access layer
|   +-- analysis/
|   |   +-- preprocessing.py        # Returns, stationarity, winsorization
|   |   +-- correlation/
|   |       +-- pearson_spearman.py  # Static correlation with p-values
|   |       +-- rolling.py          # Rolling correlation (5/10/20 day windows)
|   |       +-- lead_lag.py         # Shifted correlation + Granger causality
|   |       +-- partial.py          # Partial correlation (pingouin + fallback)
|   |       +-- service.py          # Orchestrates all correlation methods
|   +-- backtest/
|   |   +-- signals.py              # Signal conditions and combination generator
|   |   +-- simulator.py            # Day-trade P&L simulation engine
|   +-- visualization/
|   |   +-- heatmap.py              # Correlation heatmaps (matplotlib/seaborn)
|   |   +-- ranking_table.py        # Ranked results tables
|   +-- orchestration/
|   |   +-- correlation_pipeline.py # End-to-end correlation analysis
|   |   +-- backtest_pipeline.py    # End-to-end signal backtesting
|   +-- cli/
|       +-- main.py                 # Click CLI commands
+-- tests/
+-- pyproject.toml
+-- .env.example
```

## Data Flow

```
[Data Sources]                    [Normalization]              [Analysis]
pykrx ----+                      +-- KRX Calendar --+         +-- Pearson/Spearman
FRED -----+-> Fetch -> Normalize --> Lag Adjustment -+-> Returns -> Rolling (5/10/20)
FDR ------+                      +-- Gap Fill ------+         +-- Lead-Lag + Granger
                                                              +-- Partial Correlation
                                                                       |
[Storage]                          [Backtesting]               [Output]
SQLite <-- Raw data          Signal Generator --> Simulator --> Heatmaps
       <-- Correlation runs  (AND combinations)   (Day-trade)  Rankings
       <-- Backtest results  Win Rate + P&L calc               CSV Reports
```

## Critical Design Decisions

### 1. Look-Ahead Bias Prevention
US market data (S&P 500, NASDAQ, VIX) is lagged by 1 KRX business day because
US markets close AFTER Korean markets (15:30 KST). This is enforced in
`data/normalizer.py` via the `indicator_lags` parameter.

### 2. Correlation Methods
- **Pearson**: Linear relationships between daily returns
- **Spearman**: Rank-based, captures non-linear monotonic relationships
- **Rolling (5/10/20)**: How correlation evolves over time
- **Lead-Lag**: Does WTI move BEFORE KOSPI? Tested at lags -5 to +5 days
- **Partial**: True WTI-KOSPI relationship after controlling for VIX, DXY, etc.

### 3. Signal Backtesting
Combinatorial search of signal conditions (max depth 3) with safeguards:
- Contradictory conditions filtered out
- Minimum observation count required
- Walk-forward validation recommended (not in-sample only)

### 4. SQLite Schema
17 tables covering: data catalog, ETF prices, indicators, series fingerprints
(for cache invalidation), correlation runs/pairs/rankings, backtest runs/trades/metrics,
fetch logs, and visualization artifacts.

## Watchlist (12 Instruments)

| Category | Code | Name |
|----------|------|------|
| Core Long | 122630 | KODEX Leverage |
| Core Long | 233740 | KODEX KOSDAQ150 Leverage |
| Core Long | 091170 | KODEX Semiconductor Leverage |
| Core Long | 396520 | TIGER Semiconductor TOP10 Leverage |
| Core Long | 472170 | KODEX Defense TOP10 Leverage |
| Core Short | 252670 | KODEX 200 Futures Inverse 2X |
| Core Short | 530031 | Samsung Inverse 2X KOSDAQ150 Futures ETN |
| Sector ETF | 091160 | KODEX Semiconductor |
| Sector ETF | 229200 | KODEX KOSDAQ150 |
| Sector ETF | 457990 | KoAct KOSDAQ Active |
| Sector ETF | 459580 | TIME KOSDAQ Active |
| Sector ETF | 472160 | KODEX Defense TOP10 |

## 15 Market Indicators

| Category | Indicator | Source | Lag |
|----------|-----------|--------|-----|
| Commodity | WTI Crude Oil | FRED (DCOILWTICO) | 0 |
| Commodity | Brent Crude Oil | FRED (DCOILBRENTEU) | 0 |
| FX | USD/KRW | pykrx/FDR | 0 |
| FX | DXY Dollar Index | FRED (DTWEXBGS) | 0 |
| KRX Derivative | KOSPI200 Futures Basis | pykrx | 0 |
| KRX Flow | Foreign Futures Net | pykrx | 0 |
| KRX Flow | Institutional Net | pykrx | 0 |
| KRX Flow | Individual Net | pykrx | 0 |
| KRX Flow | Program Trading Net | pykrx | 0 |
| Global Index | S&P 500 | FRED (SP500) | 1 |
| Global Index | NASDAQ | FRED (NASDAQCOM) | 1 |
| Global Index | Nikkei 225 | FDR (N225) | 0 |
| Global Index | Shanghai Composite | FDR (SSEC) | 0 |
| Volatility | VIX | FRED (VIXCLS) | 1 |
| Volatility | VKOSPI | pykrx | 0 |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Initialize database
kospi-corr init-db

# View watchlist info
kospi-corr info

# Run correlation analysis (90-day lookback)
kospi-corr correlate --days 90

# Python API usage
from kospi_corr.orchestration.correlation_pipeline import CorrelationPipeline
pipeline = CorrelationPipeline()
result = pipeline.execute()
```

## Dependencies

Core: pandas, numpy, scipy, statsmodels, pingouin
Data: pykrx, finance-datareader, fredapi, yfinance
Visualization: matplotlib, seaborn
Config: pyyaml, pydantic, pydantic-settings
CLI: click, rich
