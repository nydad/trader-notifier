#!/usr/bin/env python3
"""Standalone helper script for KOSPI ETF correlation analysis.

Runs the full correlation pipeline: data collection, alignment, correlation
computation, rich CLI output, and heatmap generation.

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/analyze.py --days 180
    PYTHONPATH=src python scripts/analyze.py --days 365 --no-heatmap
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure src/ is on sys.path when invoked standalone
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KOSPI ETF Correlation Analysis Pipeline",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Lookback period in calendar days (default: 180)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag in days for lead-lag analysis (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "output"),
        help="Output directory for heatmap PNGs (default: output/)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip heatmap PNG generation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("analyze")

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
    except ImportError:
        console = None

    # Header
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    if console:
        console.print()
        console.print(Panel(
            f"[bold cyan]KOSPI ETF Correlation Analysis[/bold cyan]\n"
            f"Period: {start_date} ~ {end_date} ({args.days} days)\n"
            f"Max Lag: {args.max_lag} days | Heatmap: {'OFF' if args.no_heatmap else 'ON'}",
            title="Analyze",
            border_style="blue",
        ))
    else:
        print(f"\n=== KOSPI ETF Correlation Analysis ===")
        print(f"Period: {start_date} ~ {end_date} ({args.days} days)")

    # Run pipeline
    try:
        from kospi_corr.orchestration import MarketOrchestrator
        from kospi_corr.visualization import generate_heatmap, print_rich_summary

        logger.info("Initializing MarketOrchestrator...")
        orch = MarketOrchestrator(lookback_days=args.days)

        logger.info("Running correlation analysis...")
        result = orch.run_correlation(
            lookback_days=args.days,
            max_lag=args.max_lag,
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

        # Heatmap generation
        if not args.no_heatmap and result["matrices"]:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)

            for method, matrix in result["matrices"].items():
                if not matrix.empty:
                    title = (
                        f"{method.title()} Correlation "
                        f"({result['start']} ~ {result['end']})"
                    )
                    heatmap_path = generate_heatmap(
                        matrix,
                        title=title,
                        output_path=out_dir / f"corr_{method}.png",
                    )
                    logger.info(f"Heatmap saved: {heatmap_path}")
                    if console:
                        console.print(
                            f"  [green]Heatmap[/green]: {heatmap_path}"
                        )

        # Summary
        n_pairs = len(result["result"].pairs)
        n_etfs = len(result["etf_codes"])
        n_indicators = len(result["indicator_keys"])

        if console:
            console.print()
            console.print(Panel(
                f"[bold green]Analysis Complete[/bold green]\n"
                f"ETFs: {n_etfs} | Indicators: {n_indicators} | "
                f"Correlation Pairs: {n_pairs}",
                border_style="green",
            ))
        else:
            print(f"\nDone. {n_etfs} ETFs, {n_indicators} indicators, "
                  f"{n_pairs} pairs computed.")

        return 0

    except Exception as exc:
        logger.error(f"Analysis failed: {exc}", exc_info=args.verbose)
        if console:
            console.print(f"[bold red]Error:[/bold red] {exc}")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
