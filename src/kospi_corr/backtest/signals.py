"""Signal definition and combination generator for backtesting.

Defines how indicator conditions combine to generate trade signals.
Example: "WTI daily return > 0 AND Foreign futures net > 0" -> LONG

Addresses multiple testing concern (from Spark reviewer):
  - Max combination depth capped at 3 conditions
  - Requires minimum observation count for validity
"""
from __future__ import annotations

import itertools
import logging
from typing import Literal

import numpy as np
import pandas as pd

from kospi_corr.domain.types import SignalCombination, SignalCondition, SignalDirection

logger = logging.getLogger(__name__)


# Predefined signal conditions based on common trading hypotheses
def build_default_conditions(indicator_columns: list[str]) -> list[SignalCondition]:
    """Generate standard conditions for each indicator.

    For each indicator, creates:
      - change_gt 0 (indicator went up)
      - change_lt 0 (indicator went down)
    """
    conditions: list[SignalCondition] = []
    for col in indicator_columns:
        conditions.append(SignalCondition(indicator=col, operator="change_gt", threshold=0))
        conditions.append(SignalCondition(indicator=col, operator="change_lt", threshold=0))
    return conditions


class SignalEvaluator:
    """Evaluate signal conditions against a row of data."""

    @staticmethod
    def evaluate_condition(row: pd.Series, condition: SignalCondition) -> bool:
        """Check if a single condition is met for a data row."""
        val = row.get(condition.indicator)
        if val is None or pd.isna(val):
            return False

        op = condition.operator
        thresh = condition.threshold

        if op == "gt":
            return val > thresh
        elif op == "lt":
            return val < thresh
        elif op == "gte":
            return val >= thresh
        elif op == "lte":
            return val <= thresh
        elif op == "eq":
            return val == thresh
        elif op == "change_gt":
            # The value IS already a return/change
            return val > thresh
        elif op == "change_lt":
            return val < thresh
        else:
            logger.warning(f"Unknown operator: {op}")
            return False

    @staticmethod
    def evaluate_combination(
        row: pd.Series, combo: SignalCombination
    ) -> bool:
        """Check if ALL conditions in a combination are met (AND logic)."""
        return all(
            SignalEvaluator.evaluate_condition(row, cond)
            for cond in combo.conditions
        )


class SignalCombinationGenerator:
    """Generate signal combinations from conditions.

    Controls combinatorial explosion by:
      - Limiting max_depth (default 3)
      - Filtering contradictory conditions
      - Generating both LONG and SHORT directions
    """

    def generate(
        self,
        conditions: list[SignalCondition],
        max_depth: int = 3,
        directions: tuple[SignalDirection, ...] = (SignalDirection.LONG,),
    ) -> list[SignalCombination]:
        """Generate all valid signal combinations up to max_depth.

        Args:
            conditions: Available conditions
            max_depth: Maximum conditions per combination (1 to 3)
            directions: Which trade directions to generate

        Returns:
            List of SignalCombination objects.
        """
        combos: list[SignalCombination] = []

        for depth in range(1, min(max_depth + 1, 4)):
            for cond_tuple in itertools.combinations(conditions, depth):
                # Skip contradictory conditions (same indicator, opposite direction)
                if self._has_contradiction(cond_tuple):
                    continue

                for direction in directions:
                    label = self._make_label(cond_tuple, direction)
                    combo = SignalCombination(
                        conditions=cond_tuple,
                        direction=direction,
                        label=label,
                    )
                    combos.append(combo)

        logger.info(
            f"Generated {len(combos)} signal combinations "
            f"(depth 1..{max_depth}, {len(conditions)} conditions)"
        )
        return combos

    @staticmethod
    def _has_contradiction(conditions: tuple[SignalCondition, ...]) -> bool:
        """Check if conditions contain contradictions on same indicator."""
        seen: dict[str, set[str]] = {}
        for c in conditions:
            if c.indicator not in seen:
                seen[c.indicator] = set()
            seen[c.indicator].add(c.operator)

        for indicator, ops in seen.items():
            if "change_gt" in ops and "change_lt" in ops:
                return True
            if "gt" in ops and "lt" in ops:
                return True
        return False

    @staticmethod
    def _make_label(
        conditions: tuple[SignalCondition, ...], direction: SignalDirection
    ) -> str:
        parts = [f"{c.indicator} {c.operator} {c.threshold}" for c in conditions]
        return f"{direction.name}: {' AND '.join(parts)}"
