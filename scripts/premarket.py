#!/usr/bin/env python3
"""Pre-market check for KOSPI ETF leverage day-trading.

Fetches REAL-TIME quotes from yfinance, compares vs previous close,
and runs Bayesian signal engine for today's direction.

Priority:
  1. USD/KRW overnight movement (most important)
  2. S&P500/NASDAQ close (gap direction)
  3. Oil price direction (FX impact size)
  4. Iran/Trump news (FX-shaking events)

Usage:
    cd E:/workspace/market
    PYTHONPATH=src python scripts/premarket.py
    PYTHONPATH=src python scripts/premarket.py --discord
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as _time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger("premarket")

# ---------------------------------------------------------------------------
# yfinance tickers for real-time quotes
# ---------------------------------------------------------------------------

REALTIME_TICKERS = {
    "usd_krw":     {"yf": "KRW=X",     "label": "USD/KRW 환율",      "priority": 1, "dir": -1, "weight": 2.5, "cat": "direction_driver", "invert_quote": True},
    "sp500":       {"yf": "^GSPC",      "label": "S&P500",            "priority": 2, "dir": +1, "weight": 2.0, "cat": "direction_driver"},
    "nasdaq":      {"yf": "^IXIC",      "label": "NASDAQ",            "priority": 2, "dir": +1, "weight": 1.5, "cat": "direction_driver"},
    "sp500_fut":   {"yf": "ES=F",       "label": "S&P500 선물(E-mini)","priority": 2, "dir": +1, "weight": 1.8, "cat": "direction_driver"},
    "nasdaq_fut":  {"yf": "NQ=F",       "label": "NASDAQ 선물",        "priority": 2, "dir": +1, "weight": 1.3, "cat": "direction_driver"},
    "wti_crude":   {"yf": "CL=F",       "label": "WTI 유가",          "priority": 3, "dir": +1, "weight": 1.2, "cat": "direction_driver"},
    "brent_crude": {"yf": "BZ=F",       "label": "Brent 유가",        "priority": 3, "dir": +1, "weight": 0.6, "cat": "direction_driver"},
    "dxy":         {"yf": "DX-Y.NYB",   "label": "달러인덱스(DXY)",    "priority": 3, "dir": -1, "weight": 1.0, "cat": "direction_driver"},
    "vix":         {"yf": "^VIX",       "label": "VIX",               "priority": 4, "dir": -1, "weight": 1.0, "cat": "sentiment"},
    "gold":        {"yf": "GC=F",       "label": "금",                "priority": 4, "dir": -1, "weight": 0.5, "cat": "sentiment"},
}


def _fetch_realtime(ticker_map: dict) -> dict[str, dict]:
    """Fetch latest quote + previous close from yfinance for each ticker.

    Returns {key: {"current": float, "prev_close": float, "change_pct": float, ...}}
    """
    import yfinance as yf

    results = {}
    for key, info in ticker_map.items():
        yf_sym = info["yf"]
        try:
            _time.sleep(0.3)
            t = yf.Ticker(yf_sym)

            # Try fast_info for real-time quote
            try:
                fi = t.fast_info
                current = fi.get("lastPrice") or fi.get("last_price")
                prev = fi.get("previousClose") or fi.get("previous_close")
            except Exception:
                current = None
                prev = None

            # Fallback: use recent history
            if current is None or prev is None:
                hist = t.history(period="5d")
                if hist.empty or len(hist) < 2:
                    logger.warning(f"{yf_sym}: no data")
                    continue
                close_col = "Close" if "Close" in hist.columns else "close"
                current = float(hist[close_col].iloc[-1])
                prev = float(hist[close_col].iloc[-2])

            # For KRW=X, yfinance gives USD/KRW as "1 USD = X KRW"
            # We want: positive change = won weakening = dollar stronger
            if info.get("invert_quote"):
                # KRW=X: higher = weaker won, which is what we want
                pass

            change_pct = (current - prev) / prev if prev != 0 else 0.0

            # Get 30-day history for z-score
            hist30 = t.history(period="1mo")
            if hist30 is not None and len(hist30) > 5:
                close_col = "Close" if "Close" in hist30.columns else "close"
                rets = hist30[close_col].pct_change().dropna()
                mean_ret = float(rets.mean())
                std_ret = float(rets.std())
            else:
                mean_ret = 0.0
                std_ret = 0.01

            results[key] = {
                "current": float(current),
                "prev_close": float(prev),
                "change_pct": float(change_pct),
                "mean_ret": mean_ret,
                "std_ret": std_ret,
                "label": info["label"],
                "priority": info["priority"],
                "dir": info["dir"],
                "weight": info["weight"],
                "cat": info["cat"],
            }
            logger.info(f"{info['label']}: {current:.2f} ({change_pct:+.2%})")

        except Exception as e:
            logger.warning(f"{yf_sym} failed: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def _interpret(key: str, data: dict, all_data: dict) -> str:
    ret = data["change_pct"]
    if key == "usd_krw":
        if abs(ret) < 0.001:
            return "환율 보합 → 영향 미미"
        direction = "원화 약세(달러 강세)" if ret > 0 else "원화 강세(달러 약세)"
        impact = "KOSPI 하방 압력" if ret > 0 else "KOSPI 상방 압력"
        strength = "강한" if abs(ret) >= 0.008 else ("의미있는" if abs(ret) >= 0.003 else "약한")
        return f"{direction} {ret:+.2%} → {strength} {impact}"

    if key in ("sp500", "nasdaq", "sp500_fut", "nasdaq_fut"):
        if abs(ret) < 0.002:
            return f"보합({ret:+.2%}) → 갭 없이 출발"
        d = "상승" if ret > 0 else "하락"
        return f"미국 {d}({ret:+.2%}) → 갭 {d} 출발 가능"

    if key in ("wti_crude", "brent_crude"):
        if abs(ret) < 0.01:
            return f"보합({ret:+.2%}) → 환율 영향 미미"
        if abs(ret) >= 0.03:
            return f"급변({ret:+.2%}) → 환율 2차 영향 주시"
        return f"변동({ret:+.2%}) → 환율 간접 영향"

    if key == "dxy":
        if ret > 0.002: return f"달러 강세({ret:+.2%}) → 원화 약세 압력"
        if ret < -0.002: return f"달러 약세({ret:+.2%}) → 원화 강세 지지"
        return "보합"

    if key == "vix":
        if ret > 0.05: return f"급등({ret:+.2%}) → 위험회피 심화"
        if ret < -0.05: return f"하락({ret:+.2%}) → 위험선호 회복"
        return f"보합({ret:+.2%})"

    if key == "gold":
        if ret > 0.01: return f"상승({ret:+.2%}) → 안전자산 선호"
        if ret < -0.01: return f"하락({ret:+.2%}) → 위험자산 선호"
        return f"보합({ret:+.2%})"

    return f"{ret:+.2%}"


# ---------------------------------------------------------------------------
# Discord send
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1481273084916011008"  # #코스피 channel


def _send_discord_message(content: str = "", embed: dict | None = None):
    """Send message to Discord #코스피 channel via bot API."""
    import requests
    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]

    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    if resp.status_code not in (200, 201):
        logger.warning(f"Discord send failed: {resp.status_code} {resp.text[:200]}")
        return False
    return True


def _build_discord_embed(direction: str, long_p: float, short_p: float,
                         confidence: float, data: dict, news_text: str) -> dict:
    """Build Discord embed for pre-market signal."""
    color = {"LONG": 0x00FF00, "SHORT": 0xFF0000, "NEUTRAL": 0xFFFF00}
    dir_kr = {"LONG": "매수(LONG)", "SHORT": "매도(SHORT)", "NEUTRAL": "관망(NEUTRAL)"}

    fields = []
    # Priority items
    for key in ["usd_krw", "sp500", "nasdaq", "sp500_fut", "wti_crude", "vix"]:
        if key in data:
            d = data[key]
            fields.append({
                "name": d["label"],
                "value": f"{d['current']:,.2f} ({d['change_pct']:+.2%})",
                "inline": True,
            })

    fields.append({
        "name": "LONG / SHORT",
        "value": f"**{long_p:.1%}** / {short_p:.1%} (신뢰도 {confidence:.0%})",
        "inline": False,
    })

    if news_text:
        fields.append({"name": "뉴스", "value": news_text[:200], "inline": False})

    return {
        "title": f"장전 체크 — {dir_kr.get(direction, direction)}",
        "color": color.get(direction, 0xFFFF00),
        "fields": fields,
        "footer": {"text": f"KOSPI Signal Bot | {datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')}"},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="KOSPI Pre-Market Check (실시간)")
    parser.add_argument("--discord", action="store_true", help="Send to Discord")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        console = Console()
    except ImportError:
        print("pip install rich", file=sys.stderr)
        return 1

    kst = timezone(timedelta(hours=9))
    kst_now = datetime.now(kst)
    console.print()
    console.print(Panel(
        f"[bold cyan]KOSPI ETF Pre-Market Check (실시간)[/bold cyan]\n"
        f"{kst_now.strftime('%Y-%m-%d %H:%M KST')}",
        title="장 시작 전 체크리스트",
        border_style="blue",
    ))

    # 1. Fetch real-time quotes
    console.print("\n[bold]실시간 호가 수집 중...[/bold]")
    data = _fetch_realtime(REALTIME_TICKERS)

    if not data:
        console.print("[bold red]데이터 수집 실패[/bold red]")
        return 1

    # 2. Display table (priority order)
    tbl = Table(title="Pre-Market Check (우선순위순)", box=box.HEAVY_HEAD, show_lines=True)
    tbl.add_column("#", width=3, style="bold")
    tbl.add_column("항목", width=20, style="cyan")
    tbl.add_column("현재가", width=14, justify="right")
    tbl.add_column("전일비", width=10, justify="right")
    tbl.add_column("해석", width=48)

    sorted_items = sorted(data.items(), key=lambda x: x[1]["priority"])
    for key, d in sorted_items:
        ret = d["change_pct"]
        # Color: green if bullish for KOSPI, red if bearish
        bullish = ret * d["dir"]
        if abs(bullish) < 0.001:
            color = "yellow"
        elif bullish > 0:
            color = "green"
        else:
            color = "red"

        tbl.add_row(
            str(d["priority"]),
            d["label"],
            f"{d['current']:,.2f}",
            f"[{color}]{ret:+.2%}[/{color}]",
            _interpret(key, d, data),
        )

    console.print(tbl)

    # 3. News check
    console.print()
    console.print("[bold yellow]#4 뉴스 이벤트 체크[/bold yellow]")
    news_text = ""
    news_signal = None
    try:
        from kospi_corr.collectors.news import NewsCollector
        news = NewsCollector()
        news_signal = news.collect_signal()

        u_colors = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "bold red"}
        uc = u_colors.get(news_signal.urgency_level, "white")
        console.print(f"  Urgency: [{uc}]{news_signal.urgency_level}[/{uc}]"
                      f"  |  Sentiment: {news_signal.sentiment_score:.2f}"
                      f"  |  Scanned: {news_signal.total_articles_scanned}")

        hits = {k: v for k, v in news_signal.keyword_hits.items() if v > 0}
        if hits:
            console.print(f"  Keywords: {', '.join(f'{k}={v}' for k, v in hits.items())}")

        for art in news_signal.top_articles[:5]:
            kw = ", ".join(art.matched_keywords[:3])
            console.print(f"    [{art.source}] {art.title}  [dim]({kw})[/dim]")
            news_text += f"[{art.source}] {art.title}\n"
    except Exception as e:
        console.print(f"  [dim]뉴스 수집 실패: {e}[/dim]")

    # 4. Bayesian signal
    console.print()
    try:
        from kospi_corr.engine.bayesian import BayesianEngine, BayesianSignal
        from kospi_corr.domain.types import SignalDirection

        signals = []
        for key, d in data.items():
            ret = d["change_pct"]
            std = d["std_ret"] if d["std_ret"] > 0 else 0.01
            mean = d["mean_ret"]

            # z-score → direction-adjusted → sigmoid = P(LONG)
            z = (ret - mean) / std
            z_dir = z * d["dir"]
            prob = float(1 / (1 + np.exp(-z_dir)))
            prob = float(np.clip(prob, 0.01, 0.99))

            signals.append(BayesianSignal(
                name=key,
                probability=prob,
                weight=d["weight"],
                category=d["cat"],
                age_seconds=0.0,
                tau_seconds=14400.0,
            ))

        engine = BayesianEngine(prior=0.5, neutral_band=0.05)
        # Fetch real VKOSPI
        try:
            from kospi_corr.data.providers.naver_scraper import get_vkospi_level
            vkospi, vkospi_label = get_vkospi_level()
            console.print(f"  VKOSPI: [bold]{vkospi:.1f}[/bold] ({vkospi_label})")
        except Exception:
            vkospi = 25.0
        sp_ret = data.get("sp500", {}).get("change_pct", 0.0)

        result = engine.compute(signals, vkospi=vkospi, trend_slope=sp_ret)

        dcolors = {SignalDirection.LONG: "bold green", SignalDirection.SHORT: "bold red", SignalDirection.NEUTRAL: "bold yellow"}
        dlabels = {SignalDirection.LONG: "LONG (매수)", SignalDirection.SHORT: "SHORT (매도)", SignalDirection.NEUTRAL: "NEUTRAL (관망)"}
        d = result.direction

        console.print(Panel(
            f"[{dcolors[d]}]{dlabels[d]}[/{dcolors[d]}]\n\n"
            f"LONG  확률: [green]{result.long_probability:.1%}[/green]\n"
            f"SHORT 확률: [red]{result.short_probability:.1%}[/red]\n"
            f"신뢰도:     {result.confidence:.1%}\n"
            f"레짐:       {result.regime.regime.value} (VKOSPI {result.regime.vkospi_level:.1f})",
            title="Bayesian Signal — 종합 방향 판정",
            border_style=dcolors[d].replace("bold ", ""),
        ))

        # Contributions
        if result.key_signals:
            st = Table(title="Signal Contributions (기여도순)", box=box.SIMPLE)
            st.add_column("Signal", style="cyan")
            st.add_column("P(LONG)", justify="right")
            st.add_column("Weight", justify="right", style="dim")
            st.add_column("Contribution", justify="right")
            for s in result.key_signals[:8]:
                c = s["contribution"]
                clr = "green" if c > 0 else "red"
                st.add_row(s["name"], f"{s['probability']:.1%}", f"{s['weight']:.2f}", f"[{clr}]{c:+.4f}[/{clr}]")
            console.print(st)

        # 5. Action summary
        console.print()
        fx = data.get("usd_krw", {}).get("change_pct", 0)
        sp = data.get("sp500", {}).get("change_pct", 0)
        oil = data.get("wti_crude", {}).get("change_pct", 0)

        actions = []
        if abs(fx) >= 0.003:
            if fx > 0: actions.append("환율 상승(원화 약세) → 인버스/숏 유리")
            else: actions.append("환율 하락(원화 강세) → 레버리지/롱 유리")
        if abs(sp) > 0.005:
            d_str = "상승" if sp > 0 else "하락"
            actions.append(f"미국 {d_str}({sp:+.2%}) → 갭 {d_str} 출발")
        if abs(oil) >= 0.02:
            actions.append(f"유가 급변({oil:+.2%}) → 환율 2차 영향 주시")
        if news_signal and news_signal.urgency_level in ("HIGH", "CRITICAL"):
            actions.append(f"뉴스 {news_signal.urgency_level} → 포지션 축소")
        if not actions:
            actions.append("특이사항 없음 — 정상 트레이딩")

        console.print(Panel(
            "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions)),
            title="Action Items", border_style="cyan",
        ))

        # 6. Discord
        if args.discord:
            dir_str = {SignalDirection.LONG: "LONG", SignalDirection.SHORT: "SHORT", SignalDirection.NEUTRAL: "NEUTRAL"}[result.direction]
            embed = _build_discord_embed(
                dir_str, result.long_probability, result.short_probability,
                result.confidence, data, news_text,
            )
            if _send_discord_message(embed=embed):
                console.print("[green]Discord #코스피 채널로 전송 완료[/green]")
            else:
                console.print("[yellow]Discord 전송 실패[/yellow]")

    except Exception as e:
        logger.error(f"Signal engine failed: {e}", exc_info=True)
        console.print(f"[red]시그널 엔진 에러: {e}[/red]")

    console.print("\n[dim]* 승률 50-58%. 참고용.[/dim]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
