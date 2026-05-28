"""SessionInbox — the single entry point for messages reaching the loop.

Design: ``.claude/designs/session-inbox.md``. This is step 1 (the spine):
every message that enters the agent loop — user input today, background
completions / ticker / monitor fires / subagent findings in later steps —
arrives through :meth:`SessionInbox.push`. A turn-boundary handler drains the
inbox and renders each item into an :class:`AgentMessage` according to its
``source``.

The inbox lives in ``core.runtime`` (substrate), not in an atom: it is
mechanism, not policy, so the §11 single-file-extension contract does not
apply. Producers (atoms in later steps) push items; the runtime-owned
handlers registered by the session drain them.

``push`` is a plain synchronous list append (task/thread-safe enough for the
single-threaded asyncio model AgentM runs under). ``wait_nonempty`` is backed
by an :class:`asyncio.Event` so a future persistent driver (step 5) can block
while idle without burning CPU or LLM calls; it is implemented now but unused
by step 1, whose only driver is ``prompt``/``tick``'s run-to-idle.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    TextContent,
    UserMessage,
)

# ``InboxSource`` is a mechanism-level routing tag, not a subjective
# classification — it decides *how* an item lands (UserMessage vs synthetic
# tool_result vs system-reminder note), which is objective plumbing. It is an
# open string set: step 1 only renders ``"user"``; later steps add
# ``"background"`` / ``"ticker"`` / ``"monitor"`` / ``"subagent"`` without
# touching this type.
InboxSource = str


@dataclass(slots=True)
class InboxItem:
    """One pending message-to-be, awaiting drain at a turn boundary.

    ``payload`` is rendered into an :class:`AgentMessage` according to
    ``source`` (see :func:`SessionInbox._render_item`). A producer that
    supersedes its own prior, not-yet-drained item sets ``dedup_key``: a
    later ``push`` with the same key replaces the earlier item in place
    rather than stacking (e.g. a ticker's "still running" status line).
    """

    source: InboxSource
    payload: Any
    dedup_key: str | None = None


class SessionInbox:
    """FIFO queue of out-of-band items, drained at each turn boundary.

    Producer side: :meth:`push` (synchronous append, optional dedup).
    Driver side: :meth:`wait_nonempty` (block while idle — step-5 driver).
    Loop side: :meth:`drain` (take everything at a turn boundary) and
    :meth:`is_empty` (non-blocking emptiness check for the keep-alive floor).
    """

    def __init__(self) -> None:
        self._items: list[InboxItem] = []
        self._nonempty = asyncio.Event()

    def push(self, item: InboxItem) -> None:
        """Enqueue ``item``. With a ``dedup_key``, replace the same-key
        undrained item in place (no stacking) — the latest wins, keeping its
        arrival position so a stuck-in-a-long-turn agent never finds a pile of
        stale status lines."""

        if item.dedup_key is not None:
            for idx, existing in enumerate(self._items):
                if existing.dedup_key == item.dedup_key:
                    self._items[idx] = item
                    self._nonempty.set()
                    return
        self._items.append(item)
        self._nonempty.set()

    def drain(self) -> list[InboxItem]:
        """Remove and return every queued item in arrival order (FIFO)."""

        items = self._items
        self._items = []
        self._nonempty.clear()
        return items

    def is_empty(self) -> bool:
        """Non-blocking emptiness check (used by the keep-alive floor)."""

        return not self._items

    async def wait_nonempty(self) -> None:
        """Block until the inbox holds at least one item.

        Returns immediately if already non-empty. Backs a future persistent
        driver (step 5); unused by step 1.
        """

        await self._nonempty.wait()


def render_item(item: InboxItem) -> AgentMessage:
    """Render an :class:`InboxItem` into an :class:`AgentMessage`.

    Step 1 handles ``source="user"`` → :class:`UserMessage`. Other sources
    are reserved for later steps (background completion → synthetic
    tool_result, ticker/monitor → ``<system-reminder>`` note, etc.) and raise
    until then so a mis-routed item fails loudly rather than landing wrong.
    """

    if item.source == "user":
        content: list[Any]
        if isinstance(item.payload, str):
            content = [TextContent(type="text", text=item.payload)]
        else:
            content = list(item.payload)
        return UserMessage(role="user", content=content, timestamp=time.time())

    raise NotImplementedError(
        f"SessionInbox.render_item: source {item.source!r} is not handled in "
        f"step 1 (only 'user'); later steps add background/ticker/monitor/"
        f"subagent rendering."
    )


__all__ = ["InboxItem", "InboxSource", "SessionInbox", "render_item"]
