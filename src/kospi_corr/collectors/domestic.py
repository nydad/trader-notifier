"""Domestic Korean market real-time data collector.

Collects LIVE data during KOSPI trading hours (09:00-15:30 KST):
  - KOSPI200 index price + change + z-score normalization
  - VKOSPI volatility index (multi-source fallback)
  - Foreign/Institutional investor flow with P(LONG) signal
  - Program trading (차익/비차익) — Naver Finance real-time
  - Naver Polling API — real-time KOSPI index as KOSPI200 fallback

All fetches are best-effort: partial data is acceptable, crashes are not.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
}

# Rate-limit sleep between network calls (seconds)
_RATE_LIMIT_SLEEP = 0.1


# ---------------------------------------------------------------------------
# Sigmoid helper (avoid numpy import just for this)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# Dataclass for snapshot result
# ---------------------------------------------------------------------------

@dataclass
class DomesticSnapshot:
    """Snapshot of all domestic real-time data."""

    # KOSPI200
    kospi200_current: float | None = None
    kospi200_prev_close: float | None = None
    kospi200_change_pct: float | None = None
    kospi200_mean_ret: float = 0.0
    kospi200_std_ret: float = 1.0

    # KOSDAQ
    kosdaq_current: float | None = None
    kosdaq_change_pct: float | None = None

    # VKOSPI
    vkospi: float = 20.0
    vkospi_source: str = "default"

    # Foreign/Institutional flow (unit: 억원)
    foreign_net: float = 0.0
    institution_net: float = 0.0
    individual_net: float = 0.0
    foreign_flow_p_long: float = 0.5
    foreign_flow_combined_score: float = 0.0

    # Program trading (unit: 억원)
    program_arb_net: float = 0.0       # 차익 순매수
    program_nonarb_net: float = 0.0    # 비차익 순매수
    program_total_net: float = 0.0     # 전체 프로그램 순매수
    program_p_long: float = 0.5        # Program trading P(LONG)
    program_time: str = ""             # Last update time label

    # Metadata
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(KST)
    )

    def to_dict(self) -> dict:
        """Serialize to plain dict for downstream consumers."""
        return {
            "kospi200_current": self.kospi200_current,
            "kospi200_prev_close": self.kospi200_prev_close,
            "kospi200_change_pct": self.kospi200_change_pct,
            "kospi200_mean_ret": self.kospi200_mean_ret,
            "kospi200_std_ret": self.kospi200_std_ret,
            "kosdaq_current": self.kosdaq_current,
            "kosdaq_change_pct": self.kosdaq_change_pct,
            "vkospi": self.vkospi,
            "vkospi_source": self.vkospi_source,
            "foreign_net": self.foreign_net,
            "institution_net": self.institution_net,
            "individual_net": self.individual_net,
            "foreign_flow_p_long": self.foreign_flow_p_long,
            "foreign_flow_combined_score": self.foreign_flow_combined_score,
            "program_arb_net": self.program_arb_net,
            "program_nonarb_net": self.program_nonarb_net,
            "program_total_net": self.program_total_net,
            "program_p_long": self.program_p_long,
            "program_time": self.program_time,
            "fetched_at": self.fetched_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# 1. KOSPI200 Real-time via yfinance
# ---------------------------------------------------------------------------

def fetch_kospi200() -> dict:
    """Fetch real-time KOSPI200 index data via yfinance.

    Returns dict with keys:
        current, prev_close, change_pct, mean_ret, std_ret
    Returns empty dict on failure.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker("^KS200")

        # Primary: fast_info for live price
        current = None
        prev_close = None

        try:
            fi = ticker.fast_info
            if fi is not None:
                current = fi.get("lastPrice") or fi.get("last_price")
                prev_close = fi.get("previousClose") or fi.get("previous_close")
        except Exception as e:
            logger.debug("KOSPI200 fast_info failed: %s", e)

        # Fallback: history last row
        if current is None:
            try:
                hist = ticker.history(period="5d")
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    current = float(hist["Close"].iloc[-1])
                    if len(hist) >= 2:
                        prev_close = float(hist["Close"].iloc[-2])
                    logger.info("KOSPI200 from history fallback: %.2f", current)
            except Exception as e:
                logger.debug("KOSPI200 history fallback failed: %s", e)

        if current is None:
            logger.warning("KOSPI200: could not retrieve current price")
            return {}

        # Compute change_pct as decimal fraction (0.005 = 0.5%)
        change_pct = None
        if prev_close and prev_close > 0:
            change_pct = (current - prev_close) / prev_close

        # Compute 20-day rolling statistics for z-score normalization
        mean_ret = 0.0
        std_ret = 1.0
        try:
            hist_long = ticker.history(period="1mo")
            if hist_long is not None and len(hist_long) >= 5:
                returns = hist_long["Close"].pct_change().dropna()
                if len(returns) >= 5:
                    mean_ret = float(returns.mean())
                    std_ret = float(returns.std())
                    if std_ret < 1e-8:
                        std_ret = 1.0
        except Exception as e:
            logger.debug("KOSPI200 rolling stats failed: %s", e)

        result = {
            "current": float(current),
            "prev_close": float(prev_close) if prev_close else None,
            "change_pct": round(change_pct, 4) if change_pct is not None else None,
            "mean_ret": mean_ret,
            "std_ret": std_ret,
        }
        logger.info(
            "KOSPI200 fetched: %.2f (%.2f%%)",
            result["current"],
            result["change_pct"] or 0.0,
        )
        return result

    except Exception as e:
        logger.warning("KOSPI200 fetch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# 2. VKOSPI Real-time (multi-source fallback)
# ---------------------------------------------------------------------------

def _fetch_vkospi_naver() -> dict:
    """Fetch VKOSPI from Naver Finance sise_index page.

    URL: https://finance.naver.com/sise/sise_index.naver?code=VKOSPI
    Parses the current index value from the page.

    Returns {"value": float, "source": "naver"} or empty dict.
    """
    url = "https://finance.naver.com/sise/sise_index.naver?code=VKOSPI"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Naver index page: the current value is in <em id="now_value">
        now_val = soup.find("em", {"id": "now_value"})
        if now_val:
            text = now_val.get_text(strip=True).replace(",", "")
            value = float(text)
            # VKOSPI should be 5-100 range. If >100, it's returning KOSPI instead.
            if 1.0 < value < 100.0:
                logger.info("VKOSPI from Naver: %.2f", value)
                return {"value": value, "source": "naver"}
            else:
                logger.debug(
                    "VKOSPI Naver: value %.2f out of range (5-100), "
                    "page likely returning KOSPI instead", value,
                )

        logger.debug("VKOSPI: Naver page parsed but valid value not found")
        return {}

    except Exception as e:
        logger.debug("VKOSPI Naver fetch failed: %s", e)
        return {}


def _fetch_vkospi_investing() -> dict:
    """Fetch VKOSPI from Investing.com (existing logic).

    Returns {"value": float, "source": "investing.com"} or empty dict.
    """
    url = "https://www.investing.com/indices/kospi-volatility"
    headers = {
        **_HEADERS,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        price_el = soup.find("div", {"data-test": "instrument-price-last"})
        if price_el:
            value = float(price_el.get_text(strip=True).replace(",", ""))
            logger.info("VKOSPI from Investing.com: %.2f", value)
            return {"value": value, "source": "investing.com"}

        # Fallback selectors
        for sel in [
            "span.text-2xl",
            "span[data-test='instrument-price-last']",
        ]:
            el = soup.select_one(sel)
            if el:
                try:
                    value = float(el.get_text(strip=True).replace(",", ""))
                    logger.info("VKOSPI from Investing.com (alt): %.2f", value)
                    return {"value": value, "source": "investing.com"}
                except ValueError:
                    continue

        logger.debug("VKOSPI: Investing.com page parsed but value not found")
        return {}

    except Exception as e:
        logger.debug("VKOSPI Investing.com fetch failed: %s", e)
        return {}


def _fetch_vkospi_vix_proxy() -> dict:
    """Compute VKOSPI proxy from VIX * 1.2.

    Returns {"value": float, "source": "vix_proxy"} or empty dict.
    """
    try:
        import yfinance as yf

        vix_ticker = yf.Ticker("^VIX")
        fi = vix_ticker.fast_info
        vix_val = fi.get("lastPrice") or fi.get("last_price")

        if vix_val is None:
            hist = vix_ticker.history(period="5d")
            if hist is not None and not hist.empty:
                vix_val = float(hist["Close"].iloc[-1])

        if vix_val is not None:
            proxy = float(vix_val) * 1.2
            logger.info("VKOSPI proxy from VIX: %.2f (VIX=%.2f)", proxy, vix_val)
            return {"value": proxy, "source": "vix_proxy"}

        return {}

    except Exception as e:
        logger.debug("VIX proxy for VKOSPI failed: %s", e)
        return {}


def fetch_vkospi_realtime() -> dict:
    """Fetch VKOSPI with cascading fallbacks.

    Order:
      1. Naver Finance (primary — Korean source, most reliable)
      2. Investing.com (secondary)
      3. VIX * 1.2 proxy
      4. Default: 20.0

    Returns {"value": float, "source": str}
    """
    # Try Naver first
    result = _fetch_vkospi_naver()
    if result:
        return result

    time.sleep(_RATE_LIMIT_SLEEP)

    # Try Investing.com
    result = _fetch_vkospi_investing()
    if result:
        return result

    time.sleep(_RATE_LIMIT_SLEEP)

    # VIX proxy
    result = _fetch_vkospi_vix_proxy()
    if result:
        return result

    # Final default
    logger.warning("VKOSPI: all sources failed, using default 20.0")
    return {"value": 20.0, "source": "default"}


# ---------------------------------------------------------------------------
# 3. Foreign/Institutional Flow Real-time
# ---------------------------------------------------------------------------

def fetch_investor_flow() -> dict:
    """Fetch foreign/institutional investor flow and compute P(LONG) signal.

    Reuses the Naver Finance investor trend data from naver_scraper.
    Converts raw flow data to a sigmoid-based LONG probability.

    Returns dict with keys:
        foreign, institution, individual, p_long, combined_score
    Returns neutral defaults on failure.
    """
    try:
        from kospi_corr.data.providers.naver_scraper import fetch_investor_trend

        data = fetch_investor_trend()

        if not data:
            logger.warning("Investor flow: no data from Naver, returning neutral")
            return {
                "foreign": 0.0,
                "institution": 0.0,
                "individual": 0.0,
                "p_long": 0.5,
                "combined_score": 0.0,
            }

        foreign = data.get("foreign", 0.0)
        institution = data.get("institution", 0.0)
        individual = data.get("individual", 0.0)

        # Weighted combination: foreign has more market impact
        combined = foreign * 1.5 + institution * 1.0

        # Sigmoid mapping: P(LONG) = sigmoid(combined / 3000)
        # typical daily range: -10000 to +10000 억원
        p_long = _sigmoid(combined / 3000.0)
        p_long = max(0.05, min(0.95, p_long))  # clip

        result = {
            "foreign": foreign,
            "institution": institution,
            "individual": individual,
            "p_long": round(p_long, 4),
            "combined_score": round(combined, 2),
        }
        logger.info(
            "Investor flow: foreign=%.0f, inst=%.0f, P(LONG)=%.3f",
            foreign, institution, p_long,
        )
        return result

    except Exception as e:
        logger.warning("Investor flow fetch failed: %s", e)
        return {
            "foreign": 0.0,
            "institution": 0.0,
            "individual": 0.0,
            "p_long": 0.5,
            "combined_score": 0.0,
        }


# ---------------------------------------------------------------------------
# Main snapshot function
# ---------------------------------------------------------------------------

def fetch_domestic_snapshot() -> DomesticSnapshot:
    """Fetch all domestic data sources into a single snapshot.

    Best-effort collection — partial data is acceptable.
    Each source is independently fetched with try/except protection.
    Rate-limited with 0.3s sleep between network calls.

    Returns
    -------
    DomesticSnapshot
        Snapshot with all available domestic market data.
        Missing values use safe defaults.
    """
    snapshot = DomesticSnapshot()

    # 1. KOSPI200
    kospi_data = fetch_kospi200()
    if kospi_data:
        snapshot.kospi200_current = kospi_data.get("current")
        snapshot.kospi200_prev_close = kospi_data.get("prev_close")
        snapshot.kospi200_change_pct = kospi_data.get("change_pct")
        snapshot.kospi200_mean_ret = kospi_data.get("mean_ret", 0.0)
        snapshot.kospi200_std_ret = kospi_data.get("std_ret", 1.0)

    time.sleep(_RATE_LIMIT_SLEEP)

    # 2. VKOSPI
    vkospi_data = fetch_vkospi_realtime()
    snapshot.vkospi = vkospi_data.get("value", 20.0)
    snapshot.vkospi_source = vkospi_data.get("source", "default")

    time.sleep(_RATE_LIMIT_SLEEP)

    # 3. Naver Polling API — real-time KOSPI200 (faster than yfinance)
    try:
        from kospi_corr.collectors.naver_realtime import fetch_naver_index

        naver_idx = fetch_naver_index()
        kpi200 = naver_idx.get("KPI200")
        if kpi200 is not None and kpi200.current > 0:
            if snapshot.kospi200_current is None:
                snapshot.kospi200_current = kpi200.current
                snapshot.kospi200_change_pct = kpi200.change_rate / 100.0
                logger.info(
                    "KOSPI200 from Naver polling: %.2f (%.2f%%)",
                    kpi200.current, kpi200.change_rate,
                )
        kosdaq = naver_idx.get("KOSDAQ")
        if kosdaq is not None and kosdaq.current > 0:
            snapshot.kosdaq_current = kosdaq.current
            snapshot.kosdaq_change_pct = kosdaq.change_rate / 100.0
    except Exception as e:
        logger.debug("Naver polling augmentation failed: %s", e)

    time.sleep(_RATE_LIMIT_SLEEP)

    # 4. mainSummary API — investor flow + program trading (single call)
    _summary_fetched = False
    try:
        from kospi_corr.collectors.naver_realtime import fetch_main_summary

        summary = fetch_main_summary()
        if summary is not None:
            _summary_fetched = True

            # Investor flow from mainSummary (KOSPI = index 0)
            if summary.investor_flows:
                kospi_flow = summary.investor_flows[0]
                snapshot.foreign_net = kospi_flow.foreign
                snapshot.institution_net = kospi_flow.institution
                snapshot.individual_net = kospi_flow.individual

                combined = kospi_flow.foreign * 1.5 + kospi_flow.institution * 1.0
                p_long = _sigmoid(combined / 3000.0)
                snapshot.foreign_flow_p_long = max(0.05, min(0.95, p_long))
                snapshot.foreign_flow_combined_score = round(combined, 2)

            # Program trading from mainSummary
            prog = summary.program
            snapshot.program_arb_net = prog.arb_net
            snapshot.program_nonarb_net = prog.nonarb_net
            snapshot.program_total_net = prog.total_net
            snapshot.program_time = prog.bizdate

            if prog.total_net != 0:
                snapshot.program_p_long = _sigmoid(prog.total_net / 2000.0)
                snapshot.program_p_long = max(0.05, min(0.95, snapshot.program_p_long))
    except Exception as e:
        logger.debug("mainSummary fetch failed: %s", e)

    # 5. Fallback: old naver_scraper for investor flow (if mainSummary failed)
    if not _summary_fetched:
        flow_data = fetch_investor_flow()
        snapshot.foreign_net = flow_data.get("foreign", 0.0)
        snapshot.institution_net = flow_data.get("institution", 0.0)
        snapshot.individual_net = flow_data.get("individual", 0.0)
        snapshot.foreign_flow_p_long = flow_data.get("p_long", 0.5)
        snapshot.foreign_flow_combined_score = flow_data.get("combined_score", 0.0)

    # Timestamp
    snapshot.fetched_at = datetime.now(KST)

    logger.info(
        "Domestic snapshot complete: KOSPI200=%s, VKOSPI=%.1f(%s), "
        "Flow P(LONG)=%.3f, Program net=%.0f P(LONG)=%.3f",
        snapshot.kospi200_current,
        snapshot.vkospi,
        snapshot.vkospi_source,
        snapshot.foreign_flow_p_long,
        snapshot.program_total_net,
        snapshot.program_p_long,
    )

    return snapshot


# ---------------------------------------------------------------------------
# Convenience: VKOSPI level classification (improved from naver_scraper)
# ---------------------------------------------------------------------------

def get_vkospi_level(vkospi_value: float | None = None) -> tuple[float, str]:
    """Classify VKOSPI into a regime label.

    If no value provided, fetches live data via fetch_vkospi_realtime().

    VKOSPI ranges:
      Normal: < 20, Stable: 20-25, Elevated: 25-35, High Stress: 35-50,
      Crisis: 50+

    Returns (value, regime_label).
    """
    if vkospi_value is None:
        data = fetch_vkospi_realtime()
        vkospi_value = data.get("value", 20.0)

    v = vkospi_value

    if v >= 50:
        label = "위기 (Crisis)"
    elif v >= 35:
        label = "고스트레스 (High Stress)"
    elif v >= 25:
        label = "경계 (Elevated)"
    elif v >= 20:
        label = "안정 (Stable)"
    else:
        label = "정상 (Normal)"

    return v, label
