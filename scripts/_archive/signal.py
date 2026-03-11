#!/usr/bin/env python3
"""Standalone helper script for KOSPI ETF signal generation.

Collects current market data, runs the signal engine (WeightedScorer),
and outputs LONG/SHORT probability for each watchlist ETF.

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/signal.py
    PYTHONPATH=src python scripts/signal.py --verbose
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Ensure src/ is on sys.path when invoked standalone
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# Direction hints: +1 = higher value is bullish for KOSPI, -1 = bearish
_DIRECTION_HINTS: dict[str, int] = {
    "sp500": 1,
    "nasdaq": 1,
    "nikkei225": 1,
    "shanghai_composite": 1,
    "wti_crude": 1,
    "brent_crude": 1,
    "usd_krw": -1,       # KRW weakening is bearish
    "dxy": -1,            # Strong USD is bearish for EM
    "vix": -1,            # High VIX is bearish
    "vkospi": -1,         # High VKOSPI is bearish
    "kospi200_futures_basis": 1,
    "foreign_futures_net": 1,
    "institutional_net": 1,
    "individual_net": -1,  # Retail often contrarian
    "program_trading_net": 1,
}

# Default weights (importance)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "sp500": 2.0,
    "nasdaq": 1.5,
    "nikkei225": 0.8,
    "shanghai_composite": 0.5,
    "wti_crude": 1.0,
    "brent_crude": 0.5,
    "usd_krw": 1.8,
    "dxy": 0.8,
    "vix": 1.2,
    "vkospi": 1.0,
    "kospi200_futures_basis": 0.8,
    "foreign_futures_net": 1.5,
    "institutional_net": 1.2,
    "individual_net": 0.5,
    "program_trading_net": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KOSPI ETF Signal Generator - LONG/SHORT Probabilities",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Days of historical data for normalization stats (default: 30)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def _compute_daily_return(series: "pd.Series") -> float:
    """Compute the latest daily return from a price series."""
    if len(series) < 2:
        return float("nan")
    last = series.iloc[-1]
    prev = series.iloc[-2]
    if prev == 0 or np.isnan(prev) or np.isnan(last):
        return float("nan")
    return (last - prev) / prev


def main() -> int:
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("signal")

    try:
        import pandas as pd
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        console = Console()
    except ImportError:
        print("Error: rich and pandas are required. pip install rich pandas",
              file=sys.stderr)
        return 1

    console.print()
    console.print(Panel(
        "[bold cyan]KOSPI ETF Signal Generator[/bold cyan]\n"
        f"Date: {date.today()} | Lookback: {args.lookback} days",
        title="Signal Engine",
        border_style="blue",
    ))

    try:
        from kospi_corr.orchestration import MarketOrchestrator
        from kospi_corr.engine.scorer import WeightedScorer, SignalInput
        from kospi_corr.domain.types import SignalDirection

        # 1. Collect data
        logger.info("Initializing data collection...")
        orch = MarketOrchestrator(lookback_days=args.lookback)

        end = date.today()
        start = end - timedelta(days=args.lookback)

        # Fetch indicators
        logger.info("Fetching market indicators...")
        indicator_data, indicator_lags = orch.fetch_indicators(start, end)
        logger.info(f"Fetched {len(indicator_data)} indicators")

        # Show indicator status
        status_table = Table(
            title="Indicator Data Status",
            box=box.SIMPLE,
            show_lines=False,
        )
        status_table.add_column("Indicator", style="cyan")
        status_table.add_column("Rows", justify="right")
        status_table.add_column("Latest", style="dim")
        status_table.add_column("Return", justify="right")

        # Compute returns for each indicator
        indicator_returns: dict[str, float] = {}
        indicator_stats: dict[str, dict[str, float]] = {}

        for key, df in indicator_data.items():
            if df.empty:
                continue

            # Get the close/value column
            if "close" in df.columns:
                series = df["close"].dropna()
            elif "value" in df.columns:
                series = df["value"].dropna()
            elif len(df.columns) == 1:
                series = df.iloc[:, 0].dropna()
            else:
                continue

            if len(series) < 2:
                continue

            ret = _compute_daily_return(series)
            indicator_returns[key] = ret

            # Compute normalization stats (mean/std of daily returns)
            returns_series = series.pct_change().dropna()
            if len(returns_series) > 5:
                indicator_stats[key] = {
                    "mean": float(returns_series.mean()),
                    "std": float(returns_series.std()),
                }

            latest_date = series.index[-1]
            if hasattr(latest_date, "strftime"):
                latest_str = latest_date.strftime("%Y-%m-%d")
            else:
                latest_str = str(latest_date)

            color = "green" if not np.isnan(ret) and ret > 0 else "red"
            ret_str = (
                f"[{color}]{ret:+.4f}[/{color}]"
                if not np.isnan(ret)
                else "[dim]N/A[/dim]"
            )
            status_table.add_row(key, str(len(series)), latest_str, ret_str)

        console.print(status_table)

        if not indicator_returns:
            console.print("[bold red]No indicator data available.[/bold red]")
            return 1

        # 2. Run signal engine
        logger.info("Running WeightedScorer...")
        scorer = WeightedScorer(sigmoid_scale=4.0, neutral_band=0.05)

        # Build signal inputs
        signals: list[SignalInput] = []
        for key, ret in indicator_returns.items():
            if np.isnan(ret):
                continue

            weight = _DEFAULT_WEIGHTS.get(key, 1.0)
            direction = _DIRECTION_HINTS.get(key, 1)

            # Use historical stats for normalization if available
            norm_kw: dict[str, float] = {}
            if key in indicator_stats:
                norm_kw["norm_mean"] = indicator_stats[key]["mean"]
                norm_kw["norm_std"] = indicator_stats[key]["std"]

            signals.append(SignalInput(
                name=key,
                raw_value=ret,
                weight=weight,
                direction_hint=direction,
                **norm_kw,
            ))

        result = scorer.score(signals)

        # 3. Display results
        console.print()

        # Overall market signal
        dir_color = {
            SignalDirection.LONG: "bold green",
            SignalDirection.SHORT: "bold red",
            SignalDirection.NEUTRAL: "bold yellow",
        }
        dir_label = {
            SignalDirection.LONG: "LONG",
            SignalDirection.SHORT: "SHORT",
            SignalDirection.NEUTRAL: "NEUTRAL",
        }

        console.print(Panel(
            f"[{dir_color[result.direction]}]{dir_label[result.direction]}[/{dir_color[result.direction]}]\n\n"
            f"LONG  Probability: [green]{result.long_probability:.1%}[/green]\n"
            f"SHORT Probability: [red]{result.short_probability:.1%}[/red]\n"
            f"Raw Score: {result.raw_score:+.6f}",
            title="Market Direction Signal (KOSPI)",
            border_style=dir_color[result.direction].replace("bold ", ""),
        ))

        # Signal contributions
        contrib_table = Table(
            title="Signal Contributions",
            box=box.ROUNDED,
            show_lines=True,
        )
        contrib_table.add_column("Signal", style="cyan")
        contrib_table.add_column("Return", justify="right")
        contrib_table.add_column("Weight", justify="right", style="dim")
        contrib_table.add_column("Direction", justify="center")
        contrib_table.add_column("Contribution", justify="right")

        # Sort by absolute contribution
        sorted_contribs = sorted(
            result.signal_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        for name, contrib in sorted_contribs:
            ret = indicator_returns.get(name, float("nan"))
            weight = _DEFAULT_WEIGHTS.get(name, 1.0)
            direction = _DIRECTION_HINTS.get(name, 1)
            dir_str = "[green]+1[/green]" if direction > 0 else "[red]-1[/red]"
            color = "green" if contrib > 0 else "red"

            ret_str = f"{ret:+.4f}" if not np.isnan(ret) else "N/A"

            contrib_table.add_row(
                name,
                ret_str,
                f"{weight:.1f}",
                dir_str,
                f"[{color}]{contrib:+.6f}[/{color}]",
            )

        console.print(contrib_table)

        # 4. ETF-specific signals
        console.print()
        logger.info("Loading watchlist for ETF-specific signals...")

        try:
            watchlist = orch.load_watchlist()

            etf_table = Table(
                title="ETF Watchlist Signal Summary",
                box=box.ROUNDED,
                show_lines=True,
            )
            etf_table.add_column("Code", style="cyan")
            etf_table.add_column("Name")
            etf_table.add_column("Category", style="dim")
            etf_table.add_column("Direction", justify="center")
            etf_table.add_column("LONG %", justify="right")
            etf_table.add_column("SHORT %", justify="right")

            for item in watchlist.items:
                # For now, all ETFs share the same market signal
                # In Phase 3, sector-specific signals will differentiate
                d = result.direction
                d_str = f"[{dir_color[d]}]{dir_label[d]}[/{dir_color[d]}]"

                # Adjust for inverse ETFs
                is_inverse = "인버스" in item.name or "inverse" in item.name.lower()
                if is_inverse:
                    # Flip direction for inverse ETFs
                    if d == SignalDirection.LONG:
                        d_str = f"[bold red]SHORT[/bold red]"
                        long_p = result.short_probability
                        short_p = result.long_probability
                    elif d == SignalDirection.SHORT:
                        d_str = f"[bold green]LONG[/bold green]"
                        long_p = result.long_probability
                        short_p = result.short_probability
                    else:
                        long_p = result.long_probability
                        short_p = result.short_probability
                else:
                    long_p = result.long_probability
                    short_p = result.short_probability

                etf_table.add_row(
                    item.code,
                    item.name,
                    item.category.value,
                    d_str,
                    f"[green]{long_p:.1%}[/green]",
                    f"[red]{short_p:.1%}[/red]",
                )

            console.print(etf_table)
        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")

        # Disclaimer
        console.print()
        console.print(
            "[dim]* 현재 Phase 2 WeightedScorer 기반 시그널입니다. "
            "Phase 3 Bayesian Log-Odds 엔진이 구현되면 더 정밀한 확률이 산출됩니다.\n"
            "* 현실적 승률: 50-58%. 투자 판단의 참고용으로만 사용하세요.[/dim]"
        )
        console.print()

        return 0

    except Exception as exc:
        logger.error(f"Signal generation failed: {exc}", exc_info=args.verbose)
        if console:
            console.print(f"[bold red]Error:[/bold red] {exc}")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
