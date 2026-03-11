"""Trading signal -> Discord embed formatter."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from kospi_corr.domain.types import SignalDirection


class SignalCategory(str, Enum):
    """Signal family used for embed title."""

    KOSPI200 = "kospi200"
    SEMICONDUCTOR = "semiconductor"


class TradingSignal(BaseModel):
    """Domain-facing payload for a generated signal."""

    symbol: str
    long_probability: float = Field(..., ge=0.0, le=1.0)
    short_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_signals: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None
    direction: SignalDirection = SignalDirection.NEUTRAL
    category: SignalCategory = SignalCategory.KOSPI200
    timezone: str = "Asia/Seoul"


class DiscordEmbedField(BaseModel):
    name: str
    value: str
    inline: bool = True


class DiscordEmbed(BaseModel):
    """JSON-ready structure that matches Discord embed payload schema."""

    title: str
    color: int
    fields: list[DiscordEmbedField]
    footer: dict[str, str]
    description: str | None = None


class SignalFormatter:
    """Convert a TradingSignal to a Discord embed dict."""

    LONG_COLOR = 0x00FF00
    SHORT_COLOR = 0xFF0000
    NEUTRAL_COLOR = 0xFFFF00

    TITLE_BY_CATEGORY = {
        SignalCategory.KOSPI200: "KOSPI200 ETF Signal",
        SignalCategory.SEMICONDUCTOR: "Semiconductor ETF Signal",
    }

    def build_title(self, signal: TradingSignal) -> str:
        return self.TITLE_BY_CATEGORY.get(signal.category, "KOSPI ETF Signal")

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    @staticmethod
    def _to_kst(timestamp: datetime | None, tz: str) -> datetime:
        zone = ZoneInfo(tz)
        if timestamp is None:
            return datetime.now(zone)
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=zone)
        return timestamp.astimezone(zone)

    def color_for_direction(self, direction: SignalDirection) -> int:
        if direction == SignalDirection.LONG:
            return self.LONG_COLOR
        if direction == SignalDirection.SHORT:
            return self.SHORT_COLOR
        return self.NEUTRAL_COLOR

    def format_key_signals(self, key_signals: list[str]) -> str:
        if not key_signals:
            return "No key signals available."
        return "\n".join(f"- {item}" for item in key_signals)

    def to_embed(self, signal: TradingSignal) -> DiscordEmbed:
        timestamp = self._to_kst(signal.timestamp, signal.timezone)
        title = self.build_title(signal)
        color = self.color_for_direction(signal.direction)

        fields: list[DiscordEmbedField] = [
            DiscordEmbedField(name="Long Prob", value=self._format_pct(signal.long_probability), inline=True),
            DiscordEmbedField(name="Short Prob", value=self._format_pct(signal.short_probability), inline=True),
            DiscordEmbedField(name="Confidence", value=self._format_pct(signal.confidence), inline=True),
            DiscordEmbedField(name="Key Signals", value=self.format_key_signals(signal.key_signals), inline=False),
            DiscordEmbedField(
                name="Timestamp",
                value=timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"),
                inline=False,
            ),
        ]

        return DiscordEmbed(
            title=title,
            color=color,
            fields=fields,
            footer={"text": "KOSPI Signal Bot | Not financial advice"},
            description=f"Symbol: {signal.symbol}",
        )

    def to_payload(self, signal: TradingSignal) -> dict[str, Any]:
        """Return full payload expected by Discord API."""

        return {"embeds": [self.to_embed(signal).model_dump(exclude_none=True)]}
