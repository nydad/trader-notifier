"""CLI entry point using Typer."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="kospi-corr",
    help="KOSPI ETF Correlation Analysis & Trading Signal System",
    no_args_is_help=True,
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def correlate(
    days: int = typer.Option(180, "--days", "-d", help="Lookback period in calendar days"),
    max_lag: int = typer.Option(5, "--max-lag", help="Max lead-lag days"),
    heatmap: bool = typer.Option(True, "--heatmap/--no-heatmap", help="Generate heatmap PNG"),
    output_dir: str = typer.Option("output", "--out", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run full correlation analysis on watchlist ETFs vs market indicators."""
    _setup_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    from kospi_corr.orchestration import MarketOrchestrator
    from kospi_corr.visualization import generate_heatmap, print_rich_summary

    logger.info("Starting correlation analysis...")
    orch = MarketOrchestrator(lookback_days=days)
    result = orch.run_correlation(
        lookback_days=days,
        max_lag=max_lag,
    )

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
    if heatmap and result["matrices"]:
        out = Path(output_dir)
        for method, matrix in result["matrices"].items():
            if not matrix.empty:
                path = generate_heatmap(
                    matrix,
                    title=f"{method.title()} Correlation ({result['start']} ~ {result['end']})",
                    output_path=out / f"corr_{method}.png",
                )
                typer.echo(f"Heatmap saved: {path}")

    typer.echo("Done.")


@app.command()
def watchlist() -> None:
    """Show current watchlist."""
    from rich.console import Console
    from rich.table import Table
    from kospi_corr.orchestration import MarketOrchestrator

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


@app.command()
def indicators() -> None:
    """Show configured market indicators."""
    from rich.console import Console
    from rich.table import Table
    from kospi_corr.config import load_indicator_descriptors

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
