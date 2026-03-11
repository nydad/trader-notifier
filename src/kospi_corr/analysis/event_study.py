"""Event study analysis for ETF returns around event dates.

Given a set of event dates (e.g., keyword spike dates, macro announcements),
compute average returns and cumulative abnormal returns (CAR) in a window
around each event.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class EventStudyConfig:
    """Parameters for event study analysis."""

    pre_window: int = 5   # Days before event to include
    post_window: int = 5  # Days after event to include
    estimation_window: int = 120  # Days before pre_window for normal-return estimation
    min_estimation_obs: int = 60  # Minimum observations for estimation window


@dataclass
class EventStudyResult:
    """Result of event study analysis for one ETF / event set."""

    etf_symbol: str
    event_label: str
    n_events: int
    n_usable_events: int
    avg_return_by_day: pd.Series     # Index: relative day (-pre..+post), value: mean return
    cumulative_car: pd.Series        # Cumulative average abnormal return by day
    t_stats: pd.Series               # t-statistic per relative day
    p_values: pd.Series              # p-value per relative day (two-sided)
    detail_df: pd.DataFrame          # Full matrix: rows=events, cols=relative days
    car_at_event: float              # CAR at day 0
    car_post: float                  # CAR at +post_window
    car_total: float                 # CAR over full window


class EventStudyAnalyzer:
    """Analyze ETF returns around event dates.

    Workflow:
      1. For each event date, extract a window of ETF daily returns.
      2. Estimate "normal" (expected) returns from a pre-event estimation window
         using a simple mean-return model.
      3. Compute abnormal returns (AR) = actual - expected.
      4. Average AR across events for each relative day -> AAR.
      5. Cumulate AAR -> CAR.
      6. Test statistical significance with cross-sectional t-tests.

    Example::

        analyzer = EventStudyAnalyzer()
        result = analyzer.analyze(
            etf_returns=returns_series,
            event_dates=[date(2025,9,15), date(2025,10,1)],
            etf_symbol="122630",
            event_label="WTI spike",
        )
        print(result.avg_return_by_day)
        print(result.cumulative_car)
    """

    def __init__(self, config: EventStudyConfig | None = None) -> None:
        self.config = config or EventStudyConfig()

    def analyze(
        self,
        etf_returns: pd.Series,
        event_dates: list,
        etf_symbol: str = "",
        event_label: str = "event",
    ) -> EventStudyResult:
        """Run event study for one ETF and a set of event dates.

        Args:
            etf_returns: Daily return series for the ETF, indexed by date.
                         Can be raw returns or log returns.
            event_dates: List of event dates (datetime.date or Timestamp).
            etf_symbol: Identifier for the ETF (used in output labels).
            event_label: Human-readable label for the event type.

        Returns:
            EventStudyResult with AAR, CAR, and significance statistics.
        """
        cfg = self.config
        pre = cfg.pre_window
        post = cfg.post_window
        total_window = pre + post + 1
        relative_days = list(range(-pre, post + 1))

        # Ensure index is sorted
        etf_returns = etf_returns.sort_index()
        all_dates = etf_returns.index

        # Collect abnormal returns per event
        ar_matrix: list[list[float]] = []
        usable_events = 0

        for ev_date in event_dates:
            ev_date = pd.Timestamp(ev_date)

            # Find the position of the event date (or nearest trading day)
            pos = all_dates.searchsorted(ev_date)
            if pos >= len(all_dates):
                pos = len(all_dates) - 1

            # Snap to nearest trading day
            if all_dates[pos] != ev_date:
                # Try the previous date
                if pos > 0 and abs((all_dates[pos - 1] - ev_date).days) <= abs(
                    (all_dates[pos] - ev_date).days
                ):
                    pos = pos - 1

            actual_ev_date = all_dates[pos]

            # Check that we have enough data for the event window
            window_start = pos - pre
            window_end = pos + post
            if window_start < 0 or window_end >= len(all_dates):
                logger.debug(
                    "Skipping event %s: insufficient window data.", ev_date
                )
                continue

            # Estimation window: the period before the event window
            est_end = window_start - 1
            est_start = est_end - cfg.estimation_window + 1
            if est_start < 0:
                est_start = 0

            est_returns = etf_returns.iloc[est_start : est_end + 1]
            if len(est_returns) < cfg.min_estimation_obs:
                logger.debug(
                    "Skipping event %s: estimation window too short (%d obs).",
                    ev_date,
                    len(est_returns),
                )
                continue

            # Normal return model: simple mean
            expected_return = float(est_returns.mean())

            # Extract event-window returns
            event_window_returns = etf_returns.iloc[
                window_start : window_end + 1
            ].values

            if len(event_window_returns) != total_window:
                logger.debug(
                    "Skipping event %s: window length mismatch.", ev_date
                )
                continue

            # Abnormal returns
            ar = event_window_returns - expected_return
            ar_matrix.append(list(ar))
            usable_events += 1

        # Build results
        if usable_events == 0:
            logger.warning(
                "No usable events for %s / %s. Returning empty result.",
                etf_symbol,
                event_label,
            )
            empty_series = pd.Series(
                np.zeros(total_window), index=relative_days, dtype=float
            )
            empty_df = pd.DataFrame(columns=relative_days)
            return EventStudyResult(
                etf_symbol=etf_symbol,
                event_label=event_label,
                n_events=len(event_dates),
                n_usable_events=0,
                avg_return_by_day=empty_series,
                cumulative_car=empty_series,
                t_stats=empty_series,
                p_values=pd.Series(
                    np.ones(total_window), index=relative_days, dtype=float
                ),
                detail_df=empty_df,
                car_at_event=0.0,
                car_post=0.0,
                car_total=0.0,
            )

        ar_array = np.array(ar_matrix)  # shape: (n_events, total_window)

        # Average abnormal return (AAR) per relative day
        aar = ar_array.mean(axis=0)
        avg_return_series = pd.Series(aar, index=relative_days, name="AAR")

        # Cumulative average abnormal return (CAR)
        car = np.cumsum(aar)
        car_series = pd.Series(car, index=relative_days, name="CAR")

        # Cross-sectional t-test per relative day
        t_values = np.zeros(total_window)
        p_values_arr = np.ones(total_window)

        if usable_events >= 2:
            for j in range(total_window):
                col = ar_array[:, j]
                t_stat, p_val = sp_stats.ttest_1samp(col, 0.0)
                t_values[j] = t_stat
                p_values_arr[j] = p_val

        t_stats_series = pd.Series(t_values, index=relative_days, name="t_stat")
        p_values_series = pd.Series(p_values_arr, index=relative_days, name="p_value")

        # Detail DataFrame: rows = events, columns = relative days
        detail_df = pd.DataFrame(ar_array, columns=relative_days)
        detail_df.index.name = "event_idx"

        # Key CAR values
        day0_idx = relative_days.index(0)
        car_at_event = float(car[day0_idx])
        car_post = float(car[-1])
        car_total = float(car[-1])

        return EventStudyResult(
            etf_symbol=etf_symbol,
            event_label=event_label,
            n_events=len(event_dates),
            n_usable_events=usable_events,
            avg_return_by_day=avg_return_series,
            cumulative_car=car_series,
            t_stats=t_stats_series,
            p_values=p_values_series,
            detail_df=detail_df,
            car_at_event=car_at_event,
            car_post=car_post,
            car_total=car_total,
        )

    def analyze_multiple_etfs(
        self,
        etf_returns_dict: dict[str, pd.Series],
        event_dates: list,
        event_label: str = "event",
    ) -> dict[str, EventStudyResult]:
        """Run event study for multiple ETFs against the same event dates.

        Args:
            etf_returns_dict: ``{etf_symbol: returns_series}``
            event_dates: Shared event dates.
            event_label: Label for these events.

        Returns:
            ``{etf_symbol: EventStudyResult}``
        """
        results: dict[str, EventStudyResult] = {}
        for symbol, returns in etf_returns_dict.items():
            results[symbol] = self.analyze(
                etf_returns=returns,
                event_dates=event_dates,
                etf_symbol=symbol,
                event_label=event_label,
            )
        return results

    def results_to_summary_df(
        self, results: dict[str, EventStudyResult]
    ) -> pd.DataFrame:
        """Convert multiple event study results to a summary DataFrame.

        One row per ETF with key CAR statistics.
        """
        rows = []
        for symbol, r in results.items():
            # Find significance at day 0 and at end of window
            p_at_0 = float(r.p_values.get(0, 1.0))
            p_at_end = float(r.p_values.iloc[-1]) if len(r.p_values) > 0 else 1.0

            rows.append(
                {
                    "etf_symbol": symbol,
                    "event_label": r.event_label,
                    "n_events": r.n_events,
                    "n_usable_events": r.n_usable_events,
                    "car_at_event": r.car_at_event,
                    "car_post": r.car_post,
                    "car_total": r.car_total,
                    "p_value_day0": p_at_0,
                    "p_value_end": p_at_end,
                    "significant_day0": p_at_0 < 0.05,
                    "significant_end": p_at_end < 0.05,
                }
            )
        return pd.DataFrame(rows)
