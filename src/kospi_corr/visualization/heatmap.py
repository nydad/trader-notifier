"""Correlation heatmap renderer using matplotlib and seaborn."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class HeatmapRenderer:
    """Renders correlation matrices as annotated heatmaps."""

    def __init__(
        self,
        figsize: tuple[int, int] = (16, 12),
        dpi: int = 150,
        cmap: str = "RdBu_r",
        font_family: str = "Malgun Gothic",
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        try:
            plt.rcParams["font.family"] = font_family
        except Exception:
            logger.warning(f"Font {font_family} not available, using default")
        plt.rcParams["axes.unicode_minus"] = False

    def render_correlation_matrix(
        self,
        matrix: pd.DataFrame,
        output_path: Path,
        title: str = "Correlation Matrix",
        annotate: bool = True,
        mask_diagonal: bool = True,
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> Path:
        """Render a full correlation matrix as a heatmap.

        Args:
            matrix: Square correlation matrix (DataFrame)
            output_path: Where to save the PNG
            title: Chart title
            annotate: Show correlation values on cells
            mask_diagonal: Hide the diagonal (self-correlation = 1)

        Returns:
            Path to saved image.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        mask = None
        if mask_diagonal:
            mask = np.eye(len(matrix), dtype=bool)

        sns.heatmap(
            matrix,
            mask=mask,
            annot=annotate,
            fmt=".2f",
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            ax=ax,
        )

        ax.set_title(title, fontsize=14, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Heatmap saved to {output_path}")
        return output_path

    def render_lead_lag_heatmap(
        self,
        pairs_df: pd.DataFrame,
        target_series: str,
        output_path: Path,
        title: str = "Lead-Lag Correlation",
    ) -> Path:
        """Render lead-lag correlations for one target vs all indicators.

        Args:
            pairs_df: DataFrame with columns [series_a, series_b, lag, correlation]
            target_series: The series to show on y-axis
            output_path: Where to save

        Returns:
            Path to saved image.
        """
        # Filter to pairs involving the target
        mask = (pairs_df["series_a"] == target_series) | (pairs_df["series_b"] == target_series)
        subset = pairs_df[mask].copy()

        if subset.empty:
            logger.warning(f"No lead-lag data for {target_series}")
            return output_path

        # Build pivot: indicator x lag -> correlation
        subset["other"] = subset.apply(
            lambda r: r["series_b"] if r["series_a"] == target_series else r["series_a"],
            axis=1,
        )
        pivot = subset.pivot_table(
            index="other", columns="lag", values="correlation", aggfunc="first"
        )
        pivot = pivot.reindex(columns=sorted(pivot.columns))

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=self.cmap,
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Indicator")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Lead-lag heatmap saved to {output_path}")
        return output_path

    def render_rolling_evolution(
        self,
        rolling_series: dict[str, pd.Series],
        output_path: Path,
        title: str = "Rolling Correlation Over Time",
    ) -> Path:
        """Plot rolling correlation time series for multiple pairs.

        Args:
            rolling_series: {pair_label: Series of rolling correlations}
            output_path: Where to save

        Returns:
            Path to saved image.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for label, series in rolling_series.items():
            ax.plot(series.index, series.values, label=label, alpha=0.7)

        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Correlation")
        ax.set_xlabel("Date")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Rolling evolution chart saved to {output_path}")
        return output_path
