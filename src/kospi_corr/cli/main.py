"""CLI entry point for the KOSPI correlation and backtest system."""
from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def cli(log_level: str) -> None:
    """KOSPI ETF Correlation Analysis & Backtesting System."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--days", default=180, help="Lookback days (calendar)")
@click.option("--output", default="output", help="Output directory")
@click.option("--max-lag", default=5, help="Max lag for lead-lag analysis")
@click.option("--no-heatmap", is_flag=True, help="Skip heatmap generation")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def correlate(days: int, output: str, max_lag: int, no_heatmap: bool, verbose: bool) -> None:
    """Run correlation analysis pipeline."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from kospi_corr.orchestration import MarketOrchestrator
    from kospi_corr.visualization import generate_heatmap, print_rich_summary

    click.echo(f"Starting correlation analysis ({days} days lookback)...")
    orch = MarketOrchestrator(lookback_days=days)
    result = orch.run_correlation(lookback_days=days, max_lag=max_lag)

    # Rich CLI output
    print_rich_summary(
        rankings=result["rankings"],
        matrices=result["matrices"],
        etf_codes=result["etf_codes"],
        indicator_keys=result["indicator_keys"],
        start=result["start"],
        end=result["end"],
    )

    # Heatmap
    if not no_heatmap and result["matrices"]:
        out = Path(output)
        for method, matrix in result["matrices"].items():
            if not matrix.empty:
                path = generate_heatmap(
                    matrix,
                    title=f"{method.title()} Correlation ({result['start']} ~ {result['end']})",
                    output_path=out / f"corr_{method}.png",
                )
                click.echo(f"Heatmap saved: {path}")

    click.echo(f"Done. {len(result['result'].pairs)} correlation pairs computed.")


@cli.command()
def watchlist() -> None:
    """Show current ETF watchlist."""
    from kospi_corr.orchestration import MarketOrchestrator

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        orch = MarketOrchestrator()
        wl = orch.load_watchlist()

        table = Table(title="ETF Watchlist")
        table.add_column("Code", style="cyan")
        table.add_column("Name")
        table.add_column("Category", style="dim")

        for item in wl.items:
            table.add_row(item.code, item.name, item.category.value)

        console.print(table)
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
def indicators() -> None:
    """Show configured market indicators."""
    from kospi_corr.config import load_indicator_descriptors

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        descs = load_indicator_descriptors()

        table = Table(title="Market Indicators")
        table.add_column("Key", style="cyan")
        table.add_column("Name")
        table.add_column("Source", style="yellow")
        table.add_column("Symbol", style="dim")
        table.add_column("Lag", justify="right")

        for key, desc in descs.items():
            table.add_row(key, desc.display_name, desc.source.value, desc.source_symbol, str(desc.lag_days))

        console.print(table)
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.option("--discord", is_flag=True, help="Also send results to Discord")
@click.option("--lookback", default=30, help="Lookback days for normalization stats")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def premarket(discord: bool, lookback: int, verbose: bool) -> None:
    """Run pre-market check routine (장 시작 전 체크)."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/premarket.py", "--lookback", str(lookback)]
    if discord:
        cmd.append("--discord")
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[3]))
    raise SystemExit(result.returncode)


@cli.command()
def info() -> None:
    """Show system configuration and status."""
    click.echo("KOSPI ETF Correlation & Backtest System v0.1.0")
    click.echo(f"Watchlist: data/watchlist.json")
    click.echo(f"Config: config/app.yaml, config/indicators.yaml")
    click.echo(f"Output: output/")

    wl_path = Path("data/watchlist.json")
    if wl_path.exists():
        from kospi_corr.data.watchlist import WatchlistLoader
        loader = WatchlistLoader()
        loader.EXPECTED_COUNT = None
        wl = loader.load(wl_path)
        click.echo(f"Watchlist items: {len(wl.items)}")
        for item in wl.items:
            click.echo(f"  [{item.category.value}] {item.code} {item.name}")


if __name__ == "__main__":
    cli()
