"""Discord notification transport implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

import requests
from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, ValidationInfo, field_validator, model_validator

from kospi_corr.domain.types import SignalDirection


class NotifierMode(str, Enum):
    """Select how Discord messages are delivered."""

    WEBHOOK = "webhook"
    BOT = "bot"


class AlertLevel(str, Enum):
    """Notification aggressiveness by confidence threshold."""

    ALL = "all"
    IMPORTANT = "important"
    CRITICAL = "critical"


class DiscordNotificationError(RuntimeError):
    """Base error class for Discord transport failures."""


class DiscordConfigError(DiscordNotificationError):
    """Configuration validation error for Discord notifier."""


class DiscordConfigValidatorError(DiscordNotificationError):
    """Raised when runtime dependencies are not ready."""


class DiscordConfig(BaseModel):
    """Pydantic configuration model for Discord integration."""

    enabled: bool = True
    provider: NotifierMode = NotifierMode.WEBHOOK
    webhook_url: AnyHttpUrl | None = None
    bot_token: SecretStr | None = None
    channel_id: int | None = Field(default=None, gt=0)
    alert_level: AlertLevel = AlertLevel.ALL
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    send_neutral: bool = False
    timezone: str = "Asia/Seoul"
    market_open: str = "09:00"
    market_close: str = "15:30"
    dry_run: bool = False
    http_timeout: float = Field(default=8.0, gt=0.0, le=60.0)
    api_base_url: str = "https://discord.com/api/v10"
    webhook_username: str | None = None
    webhook_avatar_url: AnyHttpUrl | None = None
    include_direction_hint: bool = True

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise DiscordConfigError(f"Invalid timezone: {value}") from exc
        return value

    @field_validator("market_open", "market_close")
    @classmethod
    def _validate_hhmm(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2:
            raise DiscordConfigError("Market time must be HH:MM")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as exc:
            raise DiscordConfigError("Market time must be HH:MM") from exc
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise DiscordConfigError("Market time out of range")
        return value

    @model_validator(mode="after")
    def _validate_provider_config(self) -> "DiscordConfig":
        if not self.enabled:
            return self

        if self.provider == NotifierMode.WEBHOOK and not self.webhook_url:
            raise DiscordConfigError("webhook_url is required when provider='webhook'")
        if self.provider == NotifierMode.BOT:
            if not self.bot_token:
                raise DiscordConfigError("bot_token is required when provider='bot'")
            if not self.channel_id:
                raise DiscordConfigError("channel_id is required when provider='bot'")
        return self

    @field_validator("alert_level", mode="before")
    @classmethod
    def _validate_level(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class DiscordNotifier(ABC):
    """Common notifier interface."""

    def __init__(self, config: DiscordConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self._session = session or requests.Session()

    @abstractmethod
    def send(self, embed: Mapping[str, Any]) -> None:
        """Send one Discord embed payload."""

    def close(self) -> None:
        """Close HTTP session."""
        self._session.close()


@dataclass(frozen=True)
class _DirectionColors:
    long: int = 0x00FF00
    short: int = 0xFF0000
    neutral: int = 0xFFFF00


class DiscordWebhookNotifier(DiscordNotifier):
    """Send alerts via incoming webhook URL."""

    def send(self, embed: Mapping[str, Any]) -> None:
        if self.config.dry_run:
            return

        if not self.config.webhook_url:
            raise DiscordConfigValidatorError("Webhook URL missing")

        payload: dict[str, Any] = {
            "embeds": [dict(embed)],
        }
        if self.config.webhook_username:
            payload["username"] = self.config.webhook_username
        if self.config.webhook_avatar_url:
            payload["avatar_url"] = str(self.config.webhook_avatar_url)

        try:
            response = self._session.post(
                str(self.config.webhook_url),
                json=payload,
                timeout=self.config.http_timeout,
            )
        except requests.RequestException as exc:
            raise DiscordNotificationError(f"Webhook request failed: {exc}") from exc

        if response.status_code not in {200, 204}:
            raise DiscordNotificationError(
                f"Webhook request failed with {response.status_code}: {response.text}"
            )


class DiscordBotNotifier(DiscordNotifier):
    """Send alerts via bot token endpoint."""

    def send(self, embed: Mapping[str, Any]) -> None:
        if self.config.dry_run:
            return
        if not self.config.bot_token:
            raise DiscordConfigValidatorError("Bot token missing")
        if not self.config.channel_id:
            raise DiscordConfigValidatorError("Channel ID missing")

        url = (
            f"{self.config.api_base_url.rstrip('/')}"
            f"/channels/{self.config.channel_id}/messages"
        )
        headers = {
            "Authorization": f"Bot {self.config.bot_token.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {"embeds": [dict(embed)]}

        try:
            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.config.http_timeout,
            )
        except requests.RequestException as exc:
            raise DiscordNotificationError(f"Discord bot request failed: {exc}") from exc

        if response.status_code not in {200, 201, 204}:
            raise DiscordNotificationError(
                f"Discord bot request failed with {response.status_code}: {response.text}"
            )


class DiscordNotifierFactory:
    """Select and instantiate notifier based on configured mode."""

    @staticmethod
    def create(config: DiscordConfig) -> DiscordNotifier:
        if config.provider == NotifierMode.BOT:
            return DiscordBotNotifier(config)
        return DiscordWebhookNotifier(config)


def color_for_direction(direction: SignalDirection) -> int:
    """Small helper shared across modules for embed color."""

    if direction == SignalDirection.LONG:
        return _DirectionColors().long
    if direction == SignalDirection.SHORT:
        return _DirectionColors().short
    return _DirectionColors().neutral
