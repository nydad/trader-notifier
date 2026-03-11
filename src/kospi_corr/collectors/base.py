"""Abstract base class for all data collectors."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base for every collector in the system.

    Each collector fetches data from one external source (RSS, API, scraper)
    and returns it as a pandas DataFrame with a DatetimeIndex.
    """

    @abstractmethod
    def collect(self, start: date, end: date) -> pd.DataFrame:
        """Collect data for the given date range.

        Parameters
        ----------
        start : date
            Inclusive start date.
        end : date
            Inclusive end date.

        Returns
        -------
        pd.DataFrame
            DataFrame with a DatetimeIndex (``date``) and source-specific
            columns.  Concrete subclasses document the exact schema.

        Raises
        ------
        kospi_corr.domain.errors.DataProviderError
            If the external source is unreachable or returns invalid data.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable collector name (defaults to class name)."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.name}>"
