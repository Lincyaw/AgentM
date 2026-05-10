"""ChannelManager — instantiates enabled channels and fans out outbound messages.

The manager is the *channel-side* half of the gateway; the *agent-side*
half lives in :mod:`agentm_channels.gateway`. Together they own the
:class:`MessageBus`: channels publish inbound, gateway publishes
outbound, manager consumes outbound and dispatches to the right
channel by ``msg.channel``.

Config shape (under any larger config; the manager only reads the
``channels`` section)::

    channels:
      feishu:
        enabled: true
        app_id: cli_xxx
        app_secret: xxxx
        allow_from: ['*']
      slack:
        enabled: false

Channels not present in the config are simply not started — the same
shape as nanobot.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import BaseChannel
from .bus import MessageBus, OutboundMessage
from .registry import discover_all


logger = logging.getLogger(__name__)


DEFAULT_SEND_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)
"""Backoff schedule between outbound retries. The number of entries is
the number of *retries* after the initial attempt (so 3 entries → up to
4 sends total). Override at :class:`ChannelManager` construction time."""


class ChannelManager:
    def __init__(
        self,
        channels_config: dict[str, Any],
        bus: MessageBus,
        *,
        send_retry_delays: tuple[float, ...] = DEFAULT_SEND_RETRY_DELAYS,
    ) -> None:
        self._config = channels_config
        self._bus = bus
        self._send_retry_delays = send_retry_delays
        self._channels: dict[str, BaseChannel] = {}
        self._tasks: list[asyncio.Task[Any]] = []
        self._dispatch_task: asyncio.Task[Any] | None = None
        self._init_channels()

    @property
    def channels(self) -> dict[str, BaseChannel]:
        return self._channels

    def _init_channels(self) -> None:
        for name, cls in discover_all().items():
            section = self._config.get(name)
            if not section:
                continue
            if not section.get("enabled", False):
                continue
            try:
                self._channels[name] = cls(section, self._bus)
            except Exception:
                logger.exception("init channel %r failed", name)

    async def start(self) -> None:
        if not self._channels:
            logger.warning(
                "ChannelManager: no channels enabled — gateway will idle. "
                "Add `channels.<name>.enabled: true` to your config."
            )
        for name, ch in self._channels.items():
            self._tasks.append(asyncio.create_task(self._safe_start(ch), name=f"ch-{name}"))
        self._dispatch_task = asyncio.create_task(self._dispatch_loop(), name="ch-dispatch")

    async def stop(self) -> None:
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except (asyncio.CancelledError, Exception):
                pass
        for ch in self._channels.values():
            try:
                await ch.stop()
            except Exception:
                logger.exception("stop channel %r failed", ch.name)
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _safe_start(self, ch: BaseChannel) -> None:
        try:
            await ch.start()
        except Exception:
            logger.exception("channel %r start raised", ch.name)

    async def _dispatch_loop(self) -> None:
        while True:
            msg = await self._bus.consume_outbound()
            ch = self._channels.get(msg.channel)
            if ch is None:
                logger.warning(
                    "outbound dropped: no channel %r registered (msg to %s)",
                    msg.channel,
                    msg.chat_id,
                )
                continue
            await self._send_with_retry(ch, msg)

    async def _send_with_retry(self, ch: BaseChannel, msg: OutboundMessage) -> None:
        last: BaseException | None = None
        for delay in (0.0, *self._send_retry_delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                await ch.send(msg)
                return
            except Exception as exc:
                last = exc
                logger.warning(
                    "channel %r send failed (will retry after %.1fs): %s",
                    ch.name,
                    delay,
                    exc,
                )
        logger.error("channel %r gave up after retries: %s", ch.name, last)
