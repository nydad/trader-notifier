"""KRX-hour-aware dispatch layer for Discord signal notifications."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from kospi_corr.domain.types import SignalDirection

from .discord import AlertLevel, DiscordConfig, DiscordNotifier, color_for_direction
from .formatter import SignalFormatter, TradingSignal


class AlertScheduler:
    """Send alerts only when market-hour policy permits."""

    _LEVEL_TO_MIN_CONFIDENCE = {
        AlertLevel.ALL: 0.0,
        AlertLevel.IMPORTANT: 0.55,
        AlertLevel.CRITICAL: 0.8,
    }

    def __init__(
        self,
        config: DiscordConfig,
        notifier: DiscordNotifier,
        formatter: SignalFormatter | None = None,
    ) -> None:
        self.config = config
        self.notifier = notifier
        self.formatter = formatter or SignalFormatter()
        self._tz = ZoneInfo(config.timezone)
        self._market_open = self._parse_time(config.market_open)
        self._market_close = self._parse_time(config.market_close)
        if self._market_open >= self._market_close:
            raise ValueError("market_open must be earlier than market_close")

    @staticmethod
    def _parse_time(value: str) -> time:
        hour, minute = [int(v) for v in value.split(":")]
        return time(hour=hour, minute=minute)

    def _now_kst(self, now: datetime | None = None) -> datetime:
        current = now or datetime.now(self._tz)
        if current.tzinfo is None:
            return current.replace(tzinfo=self._tz)
        return current.astimezone(self._tz)

    def is_market_hours(self, now: datetime | None = None) -> bool:
        current = self._now_kst(now)
        if current.weekday() >= 5:
            return False
        if not (self._market_open <= current.time() <= self._market_close):
            return False
        return True

    def _meets_alert_level(self, signal: TradingSignal) -> bool:
        min_conf = self._LEVEL_TO_MIN_CONFIDENCE.get(self.config.alert_level, 0.0)
        if signal.confidence < min_conf:
            return False
        if signal.direction == SignalDirection.NEUTRAL and not self.config.send_neutral:
            return False
        return True

    def should_send(self, signal: TradingSignal, now: datetime | None = None) -> bool:
        if not self.config.enabled:
            return False
        if not self.is_market_hours(now):
            return False
        return self._meets_alert_level(signal)

    def dispatch(self, signal: TradingSignal, now: datetime | None = None) -> bool:
        if not self.should_send(signal, now=now):
            return False
        embed = self.formatter.to_embed(signal).model_dump(exclude_none=True)
        self.notifier.send(embed)
        return True

    def dispatch_with_payload(self, signal: TradingSignal, now: datetime | None = None) -> bool:
        if not self.should_send(signal, now=now):
            return False
        payload = self.formatter.to_payload(signal)
        self.notifier.send(payload["embeds"][0])
        return True

    def color(self, direction: SignalDirection) -> int:
        return color_for_direction(direction)
