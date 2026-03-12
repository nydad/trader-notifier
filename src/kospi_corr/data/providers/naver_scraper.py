"""Korean market data scraper — Naver Finance + Investing.com.

Provides:
  - 외국인/기관/개인 매매동향 (Naver Finance)
  - VKOSPI 실시간 (Investing.com)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

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


def _parse_amount(text: str) -> float:
    """Parse Korean number format like '-5,086' to float (unit: 억원)."""
    text = text.strip().replace(",", "").replace("+", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# 외국인/기관/개인 매매동향
# ---------------------------------------------------------------------------

def fetch_investor_trend(target_date: date | None = None) -> dict:
    """Fetch 외국인/기관/개인 net buying data from Naver Finance.

    URL: https://finance.naver.com/sise/investorDealTrendDay.naver
    Data unit: 억원 (100M KRW)

    Returns dict with keys:
        date, individual, foreign, institution, pension, insurance, trust
    Returns empty dict on failure.
    """
    if target_date is None:
        target_date = datetime.now(KST).date()

    date_str = target_date.strftime("%Y%m%d")
    url = f"https://finance.naver.com/sise/investorDealTrendDay.naver?bizdate={date_str}&page=1"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", {"class": "type_1"})
        if not table:
            logger.warning("Investor trend table not found")
            return {}

        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 7:
                continue

            date_text = cols[0].get_text(strip=True)
            if not date_text or "." not in date_text:
                continue

            # First valid data row = most recent
            values = [cols[i].get_text(strip=True) for i in range(7)]
            return {
                "date": date_text,
                "individual": _parse_amount(values[1]),
                "foreign": _parse_amount(values[2]),
                "institution": _parse_amount(values[3]),
                "pension": _parse_amount(values[4]),
                "insurance": _parse_amount(values[5]),
                "trust": _parse_amount(values[6]),
                "source": "naver",
            }

        logger.warning("No data rows found in investor trend table")
        return {}

    except Exception as e:
        logger.warning(f"Naver investor trend scraping failed: {e}")
        return {}


def fetch_investor_trend_multi(days: int = 5) -> list[dict]:
    """Fetch multiple days of investor trend data for trend analysis."""
    today = datetime.now(KST).date()
    results = []
    page = 1
    url = f"https://finance.naver.com/sise/investorDealTrendDay.naver?bizdate={today.strftime('%Y%m%d')}&page={page}"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", {"class": "type_1"})
        if not table:
            return []

        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 7:
                continue
            date_text = cols[0].get_text(strip=True)
            if not date_text or "." not in date_text:
                continue

            values = [cols[i].get_text(strip=True) for i in range(7)]
            results.append({
                "date": date_text,
                "individual": _parse_amount(values[1]),
                "foreign": _parse_amount(values[2]),
                "institution": _parse_amount(values[3]),
                "pension": _parse_amount(values[4]),
            })
            if len(results) >= days:
                break

        return results

    except Exception as e:
        logger.warning(f"Naver multi-day fetch failed: {e}")
        return []


def interpret_investor_flow(data: dict) -> str:
    """Generate Korean text interpretation of investor flow data."""
    if not data:
        return "투자자 동향 데이터 없음"

    foreign = data.get("foreign", 0)
    institution = data.get("institution", 0)
    individual = data.get("individual", 0)

    parts = []
    # Foreign
    if abs(foreign) > 500:
        action = "순매수" if foreign > 0 else "순매도"
        parts.append(f"외국인 {action} {abs(foreign):,.0f}억")
    else:
        parts.append(f"외국인 소규모 ({foreign:+,.0f}억)")

    # Institution
    if abs(institution) > 500:
        action = "순매수" if institution > 0 else "순매도"
        parts.append(f"기관 {action} {abs(institution):,.0f}억")
    else:
        parts.append(f"기관 소규모 ({institution:+,.0f}억)")

    # Assessment
    if foreign > 1000 and institution > 1000:
        parts.append("→ 외국인+기관 동반 매수 (강한 상승 신호)")
    elif foreign < -1000 and institution < -1000:
        parts.append("→ 외국인+기관 동반 매도 (강한 하락 신호)")
    elif foreign > 1000 and institution < -500:
        parts.append("→ 외국인 매수 vs 기관 매도 (혼조)")
    elif foreign < -1000 and institution > 500:
        parts.append("→ 외국인 매도 vs 기관 매수 (기관 방어)")
    elif abs(foreign) < 300 and abs(institution) < 300:
        parts.append("→ 수급 중립")

    return "\n".join(parts)


def get_investor_signal_score(data: dict) -> float:
    """Convert investor flow to a LONG probability signal (0.0~1.0).

    Logic:
    - Foreign + Institution net buying = bullish for KOSPI = LONG
    - Foreign + Institution net selling = bearish = SHORT
    Returns P(LONG) as 0.0-1.0.
    """
    if not data:
        return 0.5  # neutral

    foreign = data.get("foreign", 0)
    institution = data.get("institution", 0)

    # Weighted: foreign has more impact
    combined = foreign * 1.5 + institution * 1.0

    # Normalize: typical daily range is -10000 to +10000 억원
    # Use sigmoid-like mapping
    import numpy as np
    score = 1 / (1 + np.exp(-combined / 3000))
    return float(np.clip(score, 0.05, 0.95))


# ---------------------------------------------------------------------------
# VKOSPI (Investing.com scraping)
# ---------------------------------------------------------------------------

def fetch_vkospi() -> dict:
    """Fetch real-time VKOSPI from Investing.com.

    URL: https://www.investing.com/indices/kospi-volatility
    Returns: {"value": float, "change": str, "range": str, "source": "investing.com"}
    Returns empty dict on failure.

    VKOSPI typical ranges:
      Normal: 15-25, Elevated: 25-40, High stress: 40-60, Crisis: 60-85+
    """
    url = "https://www.investing.com/indices/kospi-volatility"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the main price element
        price_el = soup.find("div", {"data-test": "instrument-price-last"})
        if price_el:
            value = float(price_el.get_text(strip=True).replace(",", ""))
        else:
            # Fallback: look for price in various selectors
            for sel in ["span.text-2xl", "span[data-test='instrument-price-last']"]:
                el = soup.select_one(sel)
                if el:
                    try:
                        value = float(el.get_text(strip=True).replace(",", ""))
                        break
                    except ValueError:
                        continue
            else:
                # Try pandas read_html as last resort
                import pandas as pd
                tables = pd.read_html(resp.text)
                if tables:
                    for tbl in tables:
                        for col in tbl.columns:
                            if "prev" in str(col).lower() or "close" in str(col).lower():
                                try:
                                    value = float(str(tbl[col].iloc[0]).replace(",", ""))
                                    break
                                except (ValueError, IndexError):
                                    continue
                        else:
                            continue
                        break
                    else:
                        logger.warning("VKOSPI: could not find price on page")
                        return {}
                else:
                    return {}

        result = {"value": value, "source": "investing.com"}

        # Try to get change
        change_el = soup.find("span", {"data-test": "instrument-price-change-percent"})
        if change_el:
            result["change_pct"] = change_el.get_text(strip=True)

        logger.info(f"VKOSPI fetched: {value:.2f}")
        return result

    except Exception as e:
        logger.warning(f"VKOSPI fetch from Investing.com failed: {e}")
        return {}


def get_vkospi_level(vkospi_data: dict | None = None) -> tuple[float, str]:
    """Get VKOSPI value and regime label.

    Returns (value, regime_label).
    Falls back to VIX * 2.5 proxy if Investing.com fails.
    """
    if vkospi_data and "value" in vkospi_data:
        v = vkospi_data["value"]
    else:
        # Try fetching
        data = fetch_vkospi()
        if data and "value" in data:
            v = data["value"]
        else:
            # Fallback: use VIX as proxy
            try:
                import yfinance as yf
                vix = yf.Ticker("^VIX").fast_info.get("lastPrice", 20)
                v = float(vix) * 2.5  # rough VKOSPI proxy
                logger.info(f"VKOSPI proxy from VIX: {v:.1f}")
            except Exception:
                v = 25.0  # safe default

    # Classify
    if v >= 60:
        label = "위기 (Crisis)"
    elif v >= 40:
        label = "고스트레스 (High Stress)"
    elif v >= 25:
        label = "상승 (Elevated)"
    else:
        label = "정상 (Normal)"

    return v, label
