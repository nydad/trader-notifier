"""Watchlist loader: reads data/watchlist.json and produces domain Watchlist."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from kospi_corr.domain.errors import WatchlistError
from kospi_corr.domain.types import AssetCategory, Watchlist, WatchlistItem


class WatchlistLoader:
    """Loads and validates the ETF watchlist from JSON."""

    EXPECTED_COUNT = 12

    def load(self, path: Path) -> Watchlist:
        """Load watchlist from a JSON file.

        Raises WatchlistError if the file is missing, malformed,
        or fails validation.
        """
        if not path.exists():
            raise WatchlistError(f"Watchlist file not found: {path}")

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WatchlistError(f"Failed to parse watchlist: {exc}") from exc

        items: list[WatchlistItem] = []
        watchlist_data = raw.get("watchlist", {})

        category_map = {
            "core_long": AssetCategory.CORE_LONG,
            "core_short": AssetCategory.CORE_SHORT,
            "sector_etf": AssetCategory.SECTOR_ETF,
        }

        for cat_key, category in category_map.items():
            for entry in watchlist_data.get(cat_key, []):
                items.append(WatchlistItem(
                    code=entry["code"],
                    name=entry["name"],
                    category=category,
                ))

        updated_at = date.fromisoformat(raw.get("updated_at", date.today().isoformat()))
        strategy = raw.get("strategy", "")

        watchlist = Watchlist(
            items=tuple(items),
            strategy=strategy,
            updated_at=updated_at,
        )
        return self.validate(watchlist)

    def validate(self, watchlist: Watchlist, expected: int | None = None) -> Watchlist:
        """Validate item count and code uniqueness."""
        expected = expected if expected is not None else self.EXPECTED_COUNT

        if expected is not None and len(watchlist.items) != expected:
            raise WatchlistError(
                f"Expected {expected} items, got {len(watchlist.items)}"
            )

        codes = [item.code for item in watchlist.items]
        if len(codes) != len(set(codes)):
            dupes = [c for c in codes if codes.count(c) > 1]
            raise WatchlistError(f"Duplicate codes: {set(dupes)}")

        return watchlist
