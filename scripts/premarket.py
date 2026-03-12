#!/usr/bin/env python3
"""Pre-market check for KOSPI ETF leverage day-trading.

Delegates all signal generation to SignalPipeline.
Displays results via Rich and optionally sends to Discord.

Usage:
    PYTHONPATH=src python scripts/premarket.py
    PYTHONPATH=src python scripts/premarket.py --discord
    PYTHONPATH=src python scripts/premarket.py -v
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from kospi_corr.engine.market_phase import MarketPhase
from kospi_corr.engine.signal_pipeline import SignalPipeline, SignalResult

logger = logging.getLogger("premarket")

KST = timezone(timedelta(hours=9))
CHANNEL_ID = "1481273084916011008"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_result(console, result: SignalResult) -> None:
    """Show signal result as a Rich table + summary panel."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    # -- Signal contributions table --
    if result.key_signals:
        tbl = Table(
            title="Signal Contributions (기여도순)",
            box=box.HEAVY_HEAD,
            show_lines=True,
        )
        tbl.add_column("Signal", style="cyan", width=16)
        tbl.add_column("P(LONG)", justify="right", width=10)
        tbl.add_column("Weight", justify="right", style="dim", width=8)
        tbl.add_column("Contribution", justify="right", width=14)

        for sig in result.key_signals[:10]:
            c = sig["contribution"]
            clr = "green" if c > 0 else "red"
            tbl.add_row(
                sig["name"],
                f"{sig['probability']:.1%}",
                f"{sig['weight']:.2f}",
                f"[{clr}]{c:+.4f}[/{clr}]",
            )
        console.print(tbl)

    # -- Direction panel --
    dir_colors = {"LONG": "green", "SHORT": "red", "NEUTRAL": "yellow"}
    dc = dir_colors.get(result.direction, "yellow")

    console.print(Panel(
        f"[bold {dc}]{result.direction_kr}[/bold {dc}]\n\n"
        f"LONG  확률: [green]{result.long_probability:.1%}[/green]\n"
        f"SHORT 확률: [red]{result.short_probability:.1%}[/red]\n"
        f"신뢰도:     {result.confidence:.1%}\n"
        f"레짐:       {result.regime}"
        f" (VKOSPI {result.vkospi:.1f})",
        title="Bayesian Signal",
        border_style=dc,
    ))

    # -- Gate warnings --
    gate = result.gate_status
    if not gate.get("should_trade", True):
        for reason in gate.get("reasons", []):
            console.print(f"  [bold red]{reason}[/bold red]")


# ---------------------------------------------------------------------------
# Trading interpretation
# ---------------------------------------------------------------------------

def print_interpretation(console, result: SignalResult) -> None:
    """Print actionable trading interpretation."""
    console.print()
    if result.direction == "SHORT":
        console.print("[red bold]-> 인버스2X(252670) 매수 기회[/red bold]")
        console.print("  KOSPI 하방 압력, 숏 진입 유리")
    elif result.direction == "LONG":
        console.print("[green bold]-> 레버리지 ETF 매수 유리[/green bold]")
        console.print("  KOSPI 상방 압력")
        # Suggest sector based on key signals
        top_names = [s["name"] for s in result.key_signals[:3]]
        if any("nasdaq" in n for n in top_names):
            console.print("  반도체 레버리지(091170) 추천 — 나스닥 연동")
        else:
            console.print("  KODEX 레버리지(122630) 추천 — KOSPI200 연동")
    else:
        console.print("[yellow]-> 관망[/yellow]")

    for c in result.considerations:
        console.print(f"  [dim]{c}[/dim]")

    console.print("\n[dim]※ 참고용 — 투자 판단은 본인 책임 (승률 50-58%)[/dim]")


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def send_to_discord(result: SignalResult) -> bool:
    """Build embed and send to Discord #코스피 channel."""
    import requests

    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.warning("DISCORD_BOT_TOKEN not set")
        return False

    color_map = {"LONG": 0x00FF00, "SHORT": 0xFF0000, "NEUTRAL": 0xFFFF00}

    fields = []

    # Key signal contributors
    if result.key_signals:
        top = result.key_signals[:5]
        contrib_text = "\n".join(
            f"{'↑' if s['contribution'] > 0 else '↓'} {s['name']}: "
            f"P(L)={s['probability']:.0%}, 기여={s['contribution']:+.3f}"
            for s in top
        )
        fields.append({"name": "주요 기여", "value": contrib_text, "inline": False})

    fields.append({
        "name": "LONG / SHORT",
        "value": (
            f"**{result.long_probability:.1%}** / {result.short_probability:.1%}"
            f" (신뢰도 {result.confidence:.0%})"
        ),
        "inline": False,
    })

    # Considerations as summary
    if result.considerations:
        cons_text = "\n".join(f"• {c}" for c in result.considerations[:4])
        fields.append({"name": "시장 판단", "value": cons_text, "inline": False})

    embed = {
        "title": f"장전 체크 — {result.direction_kr}",
        "color": color_map.get(result.direction, 0xFFFF00),
        "fields": fields,
        "footer": {
            "text": (
                f"KOSPI Signal Bot | "
                f"{result.timestamp.strftime('%Y-%m-%d %H:%M KST')}"
            ),
        },
    }

    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json={"embeds": [embed]}, headers=headers, timeout=10)
    if resp.status_code not in (200, 201):
        logger.warning("Discord send failed: %s %s", resp.status_code, resp.text[:200])
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="KOSPI Pre-Market Check")
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
    except ImportError:
        print("pip install rich", file=sys.stderr)
        return 1

    console = Console()
    kst_now = datetime.now(KST)

    console.print()
    console.print(Panel(
        f"[bold cyan]KOSPI ETF Pre-Market Check[/bold cyan]\n"
        f"{kst_now.strftime('%Y-%m-%d %H:%M KST')}",
        title="장전 체크",
        border_style="blue",
    ))

    # Generate signal
    console.print("\n[bold]시그널 생성 중...[/bold]")
    pipeline = SignalPipeline(enable_news=True, enable_gate=True)
    result = pipeline.generate(phase_override=MarketPhase.PREMARKET)

    # Display
    display_result(console, result)
    print_interpretation(console, result)

    # Discord
    if args.discord:
        if send_to_discord(result):
            console.print("\n[green]Discord #코스피 채널로 전송 완료[/green]")
        else:
            console.print("\n[yellow]Discord 전송 실패[/yellow]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
