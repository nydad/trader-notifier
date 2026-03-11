"""Visualization module — seaborn heatmaps and rich CLI output."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    output_path: Path | str = "output/correlation_heatmap.png",
    figsize: tuple[int, int] = (16, 12),
    cmap: str = "RdBu_r",
    dpi: int = 150,
    font_family: str = "Malgun Gothic",
) -> Path:
    """Generate and save a seaborn correlation heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean column names for readability
    rename = {}
    for col in corr_matrix.columns:
        short = col.replace("etf_", "").replace("ind_", "").replace("_close", "")
        rename[col] = short
    matrix = corr_matrix.rename(columns=rename, index=rename)

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Heatmap saved: {output_path}")
    return output_path


def generate_rolling_chart(
    returns_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int = 20,
    output_path: Path | str = "output/rolling_corr.png",
    figsize: tuple[int, int] = (14, 6),
    font_family: str = "Malgun Gothic",
) -> Path:
    """Generate rolling correlation time-series chart for a pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rolling_r = returns_df[col_a].rolling(window).corr(returns_df[col_b])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling_r.index, rolling_r.values, linewidth=1.5, color="#2196F3")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(
        rolling_r.index, rolling_r.values, 0,
        where=rolling_r.values > 0, alpha=0.15, color="green"
    )
    ax.fill_between(
        rolling_r.index, rolling_r.values, 0,
        where=rolling_r.values < 0, alpha=0.15, color="red"
    )

    short_a = col_a.replace("etf_", "").replace("ind_", "").replace("_close", "")
    short_b = col_b.replace("etf_", "").replace("ind_", "").replace("_close", "")
    ax.set_title(f"Rolling {window}d Correlation: {short_a} vs {short_b}", fontsize=12)
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_rich_summary(
    rankings: dict[str, pd.DataFrame],
    matrices: dict[str, pd.DataFrame],
    etf_codes: list[str],
    indicator_keys: list[str],
    start,
    end,
) -> None:
    """Print a rich CLI summary of correlation results."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]KOSPI ETF Correlation Analysis[/bold]\n"
            f"Period: {start} ~ {end}\n"
            f"ETFs: {len(etf_codes)} | Indicators: {len(indicator_keys)}",
            title="Market Analysis",
            border_style="blue",
        )
    )

    # Top correlations by absolute value
    if "by_abs" in rankings and not rankings["by_abs"].empty:
        table = Table(
            title="Top Correlations (by |r|)",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Series A", style="cyan")
        table.add_column("Series B", style="cyan")
        table.add_column("Method", style="dim")
        table.add_column("r", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("N", justify="right")

        for _, row in rankings["by_abs"].head(15).iterrows():
            r_val = row["correlation"]
            color = "green" if r_val > 0 else "red"
            short_a = str(row["series_a"]).replace("etf_", "").replace("ind_", "").replace("_close", "")
            short_b = str(row["series_b"]).replace("etf_", "").replace("ind_", "").replace("_close", "")
            p_str = f"{row['p_value']:.4f}" if pd.notna(row.get("p_value")) else "-"
            table.add_row(
                short_a,
                short_b,
                str(row["method"]),
                f"[{color}]{r_val:+.4f}[/{color}]",
                p_str,
                str(row["n_obs"]),
            )
        console.print(table)

    # Lead-lag highlights
    if "lead_lag" in rankings and not rankings["lead_lag"].empty:
        table = Table(
            title="Lead-Lag Relationships",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Leader", style="yellow")
        table.add_column("Follower", style="yellow")
        table.add_column("Lag (days)", justify="right")
        table.add_column("r", justify="right")
        table.add_column("p-value", justify="right")

        for _, row in rankings["lead_lag"].head(10).iterrows():
            r_val = row["correlation"]
            color = "green" if r_val > 0 else "red"
            lag = row["lag"]
            leader = str(row["series_a"]) if lag > 0 else str(row["series_b"])
            follower = str(row["series_b"]) if lag > 0 else str(row["series_a"])
            leader = leader.replace("etf_", "").replace("ind_", "").replace("_close", "")
            follower = follower.replace("etf_", "").replace("ind_", "").replace("_close", "")
            p_str = f"{row['p_value']:.4f}" if pd.notna(row.get("p_value")) else "-"
            table.add_row(
                leader,
                follower,
                str(abs(lag)),
                f"[{color}]{r_val:+.4f}[/{color}]",
                p_str,
            )
        console.print(table)

    # Correlation matrix summary (pearson)
    if "pearson" in matrices and not matrices["pearson"].empty:
        mat = matrices["pearson"]
        # Show ETF-vs-indicator sub-matrix
        etf_cols = [c for c in mat.columns if c.startswith("etf_")]
        ind_cols = [c for c in mat.columns if c.startswith("ind_")]

        if etf_cols and ind_cols:
            sub = mat.loc[etf_cols, ind_cols]
            table = Table(
                title="ETF vs Indicator (Pearson)",
                box=box.SIMPLE_HEAVY,
            )
            table.add_column("ETF", style="bold")
            for ic in ind_cols:
                table.add_column(
                    ic.replace("ind_", ""),
                    justify="right",
                    max_width=8,
                )

            for etf in etf_cols:
                vals = []
                for ic in ind_cols:
                    v = sub.loc[etf, ic]
                    if pd.isna(v):
                        vals.append("[dim]-[/dim]")
                    else:
                        color = "green" if v > 0.1 else ("red" if v < -0.1 else "white")
                        vals.append(f"[{color}]{v:+.2f}[/{color}]")
                table.add_row(etf.replace("etf_", "").replace("_close", ""), *vals)
            console.print(table)

    console.print()
