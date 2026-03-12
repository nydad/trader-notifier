"""Naver Finance real-time data collectors.

Uses Naver's APIs for real-time Korean market data:
  1. Polling API — KOSPI/KOSDAQ/KPI200 real-time index (~7s delay)
  2. mainSummary JSON API — Program trading + Investor flow (single call)
  3. programDealTrendTime — Intraday program trading time-series (fallback)

All fetches are best-effort with graceful fallback to empty results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}

# mainSummary API requires browser-like headers with XHR
_SUMMARY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/sise/sise_program.naver",
    "Accept": "application/json,text/plain,*/*",
    "X-Requested-With": "XMLHttpRequest",
}

# Naver polling API transmits values as integers: actual_value × 100
_NAVER_SCALE = 100.0

# Won to 억원 conversion
_WON_TO_UK = 1e8


# ---------------------------------------------------------------------------
# 1. Naver Polling API — Real-time KOSPI/KOSDAQ/KPI200
# ---------------------------------------------------------------------------

@dataclass
class NaverIndexData:
    """Real-time index data from Naver polling API."""
    code: str           # KOSPI, KOSDAQ, KPI200
    current: float      # Current value (scaled)
    change: float       # Change value (scaled)
    change_rate: float  # Change rate in % (e.g. -0.79)
    volume: int         # Accumulated volume


def fetch_naver_index() -> dict[str, NaverIndexData]:
    """Fetch real-time KOSPI, KOSDAQ, KPI200 from Naver polling API.

    URL: polling.finance.naver.com/api/realtime?query=SERVICE_INDEX:KOSPI,KOSDAQ,KPI200
    Response fields per index:
      cd = code, nv = now value (×100), cv = change value (×100),
      cr = change rate (%), aq = accumulated quantity

    Returns dict keyed by code (KOSPI, KOSDAQ, KPI200).
    """
    url = (
        "https://polling.finance.naver.com/api/realtime"
        "?query=SERVICE_INDEX:KOSPI,KOSDAQ,KPI200"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json()

        result: dict[str, NaverIndexData] = {}
        areas = data.get("result", {}).get("areas", [])

        for area in areas:
            for item in area.get("datas", []):
                code = item.get("cd", "")
                if not code:
                    continue

                result[code] = NaverIndexData(
                    code=code,
                    current=float(item.get("nv", 0)) / _NAVER_SCALE,
                    change=float(item.get("cv", 0)) / _NAVER_SCALE,
                    change_rate=float(item.get("cr", 0.0)),
                    volume=int(item.get("aq", 0)),
                )

        if result:
            for code, idx in result.items():
                logger.info(
                    "Naver %s: %.2f (%+.2f, %+.2f%%)",
                    code, idx.current, idx.change, idx.change_rate,
                )
        return result

    except Exception as e:
        logger.warning("Naver polling API failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# 2. mainSummary JSON API — Program Trading + Investor Flow
# ---------------------------------------------------------------------------

@dataclass
class ProgramTradingData:
    """Intraday program trading snapshot."""
    arb_buy: float = 0.0       # 차익 매수 (억원)
    arb_sell: float = 0.0      # 차익 매도 (억원)
    arb_net: float = 0.0       # 차익 순매수 (억원)
    nonarb_buy: float = 0.0    # 비차익 매수 (억원)
    nonarb_sell: float = 0.0   # 비차익 매도 (억원)
    nonarb_net: float = 0.0    # 비차익 순매수 (억원)
    total_buy: float = 0.0
    total_sell: float = 0.0
    total_net: float = 0.0
    bizdate: str = ""
    source: str = "naver_api"


@dataclass
class InvestorFlowIntraday:
    """Intraday investor flow data from mainSummary API."""
    individual: float = 0.0    # 개인 순매수 (억원)
    foreign: float = 0.0       # 외국인 순매수 (억원)
    institution: float = 0.0   # 기관 순매수 (억원)
    market: str = "KOSPI"      # KOSPI, KOSDAQ, or ALL
    source: str = "naver_api"


@dataclass
class NaverSummaryData:
    """Combined data from mainSummary API (single API call)."""
    program: ProgramTradingData
    investor_flows: list[InvestorFlowIntraday]  # [KOSPI, KOSDAQ, ALL]


def fetch_main_summary() -> NaverSummaryData | None:
    """Fetch program trading + investor flow from Naver mainSummary JSON API.

    URL: api.finance.naver.com/service/mainSummary.naver
    Requires browser-like headers with X-Requested-With: XMLHttpRequest.

    Returns NaverSummaryData with program trading and investor flow.
    Returns None on failure.
    """
    url = "https://api.finance.naver.com/service/mainSummary.naver"
    try:
        resp = requests.get(
            url, headers=_SUMMARY_HEADERS, timeout=15, verify=False,
        )
        resp.raise_for_status()

        if not resp.text:
            logger.debug("mainSummary: empty response (wrong headers?)")
            return None

        data = resp.json()
        result = data.get("message", {}).get("result", {})
        if not result:
            logger.debug("mainSummary: no result in response")
            return None

        # --- Program Trading (kospiTrendProgram) ---
        program = _parse_program_trading(result.get("kospiTrendProgram", {}))

        # --- Investor Flow (todayIndexDealTrendList) ---
        flows = _parse_investor_flows(
            result.get("todayIndexDealTrendList", []),
        )

        logger.info(
            "mainSummary: program net=%.0f억 (arb=%.0f, nonarb=%.0f), "
            "KOSPI flow: foreign=%.0f억, inst=%.0f억",
            program.total_net, program.arb_net, program.nonarb_net,
            flows[0].foreign if flows else 0,
            flows[0].institution if flows else 0,
        )
        return NaverSummaryData(program=program, investor_flows=flows)

    except Exception as e:
        logger.warning("mainSummary API failed: %s", e)
        return None


def _parse_program_trading(ktp: dict) -> ProgramTradingData:
    """Parse kospiTrendProgram dict into ProgramTradingData.

    Amount fields are in 원 (won). Convert to 억원 (÷1e8).
    Net = Buy(Consign+Self) - Sell(Consign+Self)
    """
    if not ktp:
        return ProgramTradingData()

    # Arbitrage (차익)
    arb_buy = (
        ktp.get("differenceBuyConsignAmount", 0)
        + ktp.get("differenceBuySelfAmount", 0)
    ) / _WON_TO_UK
    arb_sell = (
        ktp.get("differenceSellConsignAmount", 0)
        + ktp.get("differenceSellSelfAmount", 0)
    ) / _WON_TO_UK
    arb_net = arb_buy - arb_sell

    # Non-Arbitrage (비차익)
    nonarb_buy = (
        ktp.get("biDifferenceBuyConsignAmount", 0)
        + ktp.get("biDifferenceBuySelfAmount", 0)
    ) / _WON_TO_UK
    nonarb_sell = (
        ktp.get("biDifferenceSellConsignAmount", 0)
        + ktp.get("biDifferenceSellSelfAmount", 0)
    ) / _WON_TO_UK
    nonarb_net = nonarb_buy - nonarb_sell

    return ProgramTradingData(
        arb_buy=round(arb_buy, 1),
        arb_sell=round(arb_sell, 1),
        arb_net=round(arb_net, 1),
        nonarb_buy=round(nonarb_buy, 1),
        nonarb_sell=round(nonarb_sell, 1),
        nonarb_net=round(nonarb_net, 1),
        total_buy=round(arb_buy + nonarb_buy, 1),
        total_sell=round(arb_sell + nonarb_sell, 1),
        total_net=round(arb_net + nonarb_net, 1),
        bizdate=str(ktp.get("bizdate", "")),
    )


def _parse_investor_flows(trend_list: list) -> list[InvestorFlowIntraday]:
    """Parse todayIndexDealTrendList into InvestorFlowIntraday list.

    Index mapping: [0]=KOSPI, [1]=KOSDAQ, [2]=ALL
    Values are already in 억원.
    """
    labels = ["KOSPI", "KOSDAQ", "ALL"]
    flows: list[InvestorFlowIntraday] = []

    for i, item in enumerate(trend_list[:3]):
        flows.append(InvestorFlowIntraday(
            individual=float(item.get("personalValue", 0)),
            foreign=float(item.get("foreignValue", 0)),
            institution=float(item.get("institutionalValue", 0)),
            market=labels[i] if i < len(labels) else f"idx{i}",
        ))

    return flows


# ---------------------------------------------------------------------------
# 3. Convenience: fetch program trading (with mainSummary fallback)
# ---------------------------------------------------------------------------

def fetch_program_trading() -> ProgramTradingData:
    """Fetch program trading data, preferring mainSummary JSON API."""
    summary = fetch_main_summary()
    if summary is not None:
        return summary.program
    return ProgramTradingData()


def fetch_investor_flow_intraday() -> InvestorFlowIntraday:
    """Fetch KOSPI investor flow, preferring mainSummary JSON API."""
    summary = fetch_main_summary()
    if summary is not None and summary.investor_flows:
        return summary.investor_flows[0]  # KOSPI
    return InvestorFlowIntraday()
