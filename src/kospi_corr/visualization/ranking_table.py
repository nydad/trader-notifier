"""Ranking table renderer for correlation and backtest results."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class RankingTableRenderer:
    """Render ranked results as publication-quality tables."""

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def render_correlation_ranking(
        self,
        ranking_df: pd.DataFrame,
        output_path: Path,
        title: str = "Top Correlated Indicator Pairs",
        top_n: int = 20,
    ) -> Path:
        """Render top-N correlation pairs as a styled table image.

        Args:
            ranking_df: DataFrame with correlation pair data
            output_path: Where to save the image

        Returns:
            Path to saved image.
        """
        display_df = ranking_df.head(top_n).copy()

        # Format columns
        if "correlation" in display_df.columns:
            display_df["correlation"] = display_df["correlation"].map("{:.4f}".format)
        if "p_value" in display_df.columns:
            display_df["p_value"] = display_df["p_value"].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )

        # Select display columns
        show_cols = [c for c in [
            "series_a", "series_b", "method", "window", "lag",
            "correlation", "p_value", "n_obs"
        ] if c in display_df.columns]

        display_df = display_df[show_cols]

        fig, ax = plt.subplots(figsize=(14, max(2, 0.4 * len(display_df) + 1)))
        ax.axis("off")
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(show_cols))))

        # Style header
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

        ax.set_title(title, fontsize=12, pad=20)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Ranking table saved to {output_path}")
        return output_path

    def render_backtest_ranking(
        self,
        results_df: pd.DataFrame,
        output_path: Path,
        title: str = "Signal Combination Backtest Results",
        top_n: int = 20,
    ) -> Path:
        """Render backtest results as a ranked table."""
        display_df = results_df.head(top_n).copy()

        # Format numeric columns
        fmt_map = {
            "win_rate": "{:.1%}",
            "avg_return": "{:.2%}",
            "total_pnl": "{:,.0f}",
            "max_drawdown": "{:.2%}",
            "sharpe_ratio": "{:.2f}",
            "profit_factor": "{:.2f}",
        }
        for col, fmt in fmt_map.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].map(fmt.format)

        show_cols = [c for c in [
            "label", "etf", "total_trades", "win_rate",
            "avg_return", "total_pnl", "sharpe_ratio", "profit_factor",
        ] if c in display_df.columns]

        display_df = display_df[show_cols]

        fig, ax = plt.subplots(figsize=(16, max(2, 0.4 * len(display_df) + 1)))
        ax.axis("off")
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(show_cols))))

        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

        ax.set_title(title, fontsize=12, pad=20)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Backtest ranking table saved to {output_path}")
        return output_path
