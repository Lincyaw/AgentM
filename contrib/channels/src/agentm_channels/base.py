"""Abstract :class:`BaseChannel` that all chat adapters extend.

The contract is intentionally small:

* :meth:`start` — connect / subscribe; long-running async task.
* :meth:`stop` — clean teardown; idempotent.
* :meth:`send` — deliver one :class:`OutboundMessage`.

Optional surface:

* :meth:`send_delta` — streaming text. Channels that don't support it
  fall back to whole-turn :meth:`send` calls.
* :attr:`name` / :attr:`display_name` — module identity.
* :meth:`default_config` — schema hint for an interactive setup wizard
  (matches nanobot's pattern; not used by the AgentM CLI today but
  cheap to keep).

The base class also defines :meth:`_handle_message`, the helper every
concrete channel calls when it receives a user message. It does the
permission check (``allow_from``), normalizes the payload, and pushes
an :class:`InboundMessage` onto the bus.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .bus import InboundMessage, MessageBus, OutboundMessage


logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    name: str = "base"
    """Module name; must match the file (``feishu.py`` → ``"feishu"``)."""

    display_name: str = "Base"

    def __init__(self, config: Any, bus: MessageBus) -> None:
        self.config = config
        self.bus = bus
        self._running = False

    # --- lifecycle ----------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Connect, subscribe, and stay alive until :meth:`stop` is called."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear down. Must be idempotent."""

    # --- outbound -----------------------------------------------------

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Deliver one outbound message. Raise on failure (manager retries)."""

    async def send_delta(
        self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Streaming chunk; default no-op. Subclasses opt in by overriding."""
        return None

    @property
    def supports_streaming(self) -> bool:
        return type(self).send_delta is not BaseChannel.send_delta

    # --- inbound helper -----------------------------------------------

    def is_allowed(self, sender_id: str) -> bool:
        """Permission check against ``config.allow_from``.

        Empty list **denies all** (intentional fail-closed default —
        anyone who deploys this gateway must opt in their user IDs);
        ``"*"`` permits everyone.
        """
        cfg = self.config
        if isinstance(cfg, dict):
            allow = cfg.get("allow_from", cfg.get("allowFrom", []))
        else:
            allow = getattr(cfg, "allow_from", [])
        if not allow:
            logger.warning(
                "[%s] allow_from is empty — denying %s; "
                "set allow_from: ['*'] to permit everyone, or list explicit ids",
                self.name,
                sender_id,
            )
            return False
        if "*" in allow:
            return True
        return str(sender_id) in {str(x) for x in allow}

    async def _handle_message(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """Concrete channels call this from their inbound callback."""
        if not self.is_allowed(sender_id):
            return
        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender_id=str(sender_id),
                chat_id=str(chat_id),
                content=content,
                media=media or [],
                metadata=meta,
                session_key_override=session_key,
            )
        )

    # --- introspection ------------------------------------------------

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Default config payload for an interactive setup wizard."""
        return {"enabled": False, "allow_from": []}

    @property
    def is_running(self) -> bool:
        return self._running
