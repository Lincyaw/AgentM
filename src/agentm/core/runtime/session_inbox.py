"""SessionInbox — the single entry point for messages reaching the loop.

Design: ``.claude/designs/session-inbox.md``. Every message that enters the
agent loop through the runtime entry point — user input, background tool
completion/status, monitor fires, and subagent findings — arrives through
:meth:`SessionInbox.push`. A turn-boundary handler drains the inbox and renders
each item into an :class:`AgentMessage` according to its ``source``.

The inbox lives in ``core.runtime`` (substrate), not in an atom: it is
mechanism, not policy, so the single-file-extension contract does not
apply. Producers push items; the runtime-owned handlers registered by the
session drain them.

``push`` is a plain synchronous list append (task/thread-safe enough for the
single-threaded asyncio model AgentM runs under). ``wait_nonempty`` is backed
by an :class:`asyncio.Event` so the persistent driver can block while idle
without burning CPU or LLM calls.
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
# open string set: current renderers cover ``"user"``, ``"background"``,
# ``"monitor"``, and ``"subagent"`` without constraining future producers to a
# closed enum.
InboxSource = str


@dataclass(slots=True)
class InboxItem:
    """One pending message-to-be, awaiting drain at a turn boundary.

    ``payload`` is rendered into an :class:`AgentMessage` according to
    ``source`` (see :func:`SessionInbox._render_item`). A producer that
    supersedes its own prior, not-yet-drained item sets ``dedup_key``: a
    later ``push`` with the same key replaces the earlier item in place
    rather than stacking (e.g. a ticker's "still running" status line).

    ``terminal`` carries a *terminate intent* into the loop (#177). A
    backgrounded tool that ultimately returns :class:`ToolTerminate` posts its
    completion with ``terminal=True``; the runtime drain seam records that and
    the keep-alive floor stops the loop with ``ToolTerminated`` once the item
    has been delivered, instead of keeping the agent alive on the non-empty
    inbox. The message still lands in the conversation first (so the agent sees
    the terminal tool's final result) — only the *next* turn boundary stops.
    """

    source: InboxSource
    payload: Any
    dedup_key: str | None = None
    terminal: bool = False


class SessionInbox:
    """FIFO queue of out-of-band items, drained at each turn boundary.

    Producer side: :meth:`push` (synchronous append, optional dedup).
    Driver side: :meth:`wait_nonempty` (block while idle).
    Loop side: :meth:`drain` (take everything at a turn boundary) and
    :meth:`is_empty` (non-blocking emptiness check for the keep-alive floor).
    """

    def __init__(self) -> None:
        self._items: list[InboxItem] = []
        self._nonempty = asyncio.Event()
        # #179: outstanding detached background work (auto-backgrounded tools,
        # child subagent sessions) that can still post a LATE inbox item after
        # the agent has ended its turn. A one-shot host (``agentm -p``) must NOT
        # exit while this is non-zero, or the late completion is dropped with no
        # event loop to receive it. Producers bracket their detached unit with
        # ``note_work_started`` / ``note_work_finished`` (via
        # ``ExtensionAPI.track_background``). ``_no_pending_work`` is SET while
        # the count is zero so ``wait_idle`` can block until the last unit ends.
        # Recurring signals (monitor wakeups / condition polls / tickers) do
        # NOT count — they are not work to drain before exit, and counting them
        # would keep a one-shot host alive forever.
        self._pending_work = 0
        self._no_pending_work = asyncio.Event()
        self._no_pending_work.set()

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

    def is_empty(self, sources: frozenset[str] | None = None) -> bool:
        """Non-blocking emptiness check (used by the keep-alive floor).

        With ``sources=None`` (default) this reports whether the inbox holds
        *any* item — the keep-alive floor's "inbox non-empty ⇒ keep running"
        semantics. With a ``sources`` set it reports whether any pending item
        comes from one of those producer classes, so a caller can ask a
        narrower question ("is there pending USER input?") without draining or
        being fooled by unrelated items (e.g. a producer's own
        ``source="background"`` ticker/completion echoes).
        """

        if sources is None:
            return not self._items
        return not any(item.source in sources for item in self._items)

    async def wait_nonempty(self) -> None:
        """Block until the inbox holds at least one item (or :meth:`kick`).

        Returns immediately if already non-empty / kicked.
        """

        await self._nonempty.wait()

    def note_work_started(self) -> None:
        """Record one detached background unit as live (#179).

        Clears the no-pending-work gate so ``wait_idle`` blocks until the
        matching :meth:`note_work_finished`. Idempotent only in the accounting
        sense — each call MUST be paired with exactly one finish (use
        ``ExtensionAPI.track_background`` so the pairing is structural).
        """

        self._pending_work += 1
        self._no_pending_work.clear()

    def note_work_finished(self) -> None:
        """Record one detached background unit as terminal (#179).

        Sets the gate once the count returns to zero. Never drops below zero:
        an over-finish (which would indicate a producer bug) is clamped so the
        gate cannot be falsely set while real work is still live.
        """

        if self._pending_work > 0:
            self._pending_work -= 1
        if self._pending_work == 0:
            self._no_pending_work.set()

    @property
    def has_pending_work(self) -> bool:
        """True while at least one detached background unit is live (#179)."""

        return self._pending_work > 0

    async def wait_no_pending_work(self, timeout: float | None = None) -> bool:
        """Block until every tracked background unit has finished (#179).

        With ``timeout=None`` (default) this waits unbounded — the original
        #179 semantics. With a positive ``timeout`` it waits at most that long
        and returns ``False`` if the bound tripped while work was still live;
        ``True`` means the no-pending-work gate is (now) open. The bound is the
        #201 defense-in-depth: a leaked ``note_work_started`` (never matched by
        a finish) or a genuinely stuck background unit can no longer hang the
        wait forever.

        Cancelling the inner ``Event.wait`` on timeout is safe: ``Event.wait``
        removes its own future from the event's waiter set on cancellation, so
        the gate's state and the ``_pending_work`` counter are untouched — a
        later ``note_work_finished`` still flips the gate correctly.
        """

        if timeout is None:
            await self._no_pending_work.wait()
            return True
        try:
            await asyncio.wait_for(self._no_pending_work.wait(), timeout)
        except TimeoutError:
            # Return the ACTUAL gate state, not a flat False: if the last unit
            # finished exactly at the deadline boundary the gate is already set,
            # and reporting True avoids a boundary race. Do not "simplify" this
            # to ``return False``.
            return self._no_pending_work.is_set()
        return True

    def kick(self) -> None:
        """Wake :meth:`wait_nonempty` without enqueuing an item.

        **Contract**: the caller MUST guarantee a downstream :meth:`drain`
        (the kernel ``context`` event will call it) OR a driver exit before
        the next ``wait_nonempty`` iteration. Otherwise ``wait_nonempty``
        returns immediately on the next iteration (the event is still set)
        AND the driver tight-loops with no work to do. The two existing
        callsites both honour this:

        * :meth:`AgentSession.tick` (inject path): the kick is paired with
          messages appended directly to the session log; the next
          ``loop.run`` runs a real turn whose ``context`` handler drains
          the (empty) inbox and clears the event.
        * :meth:`AgentSession.shutdown`: flips ``_closed`` BEFORE the kick;
          the driver's ``if self._closed: return`` check after
          ``wait_nonempty`` exits the loop.
        """

        self._nonempty.set()


def render_item(item: InboxItem) -> AgentMessage:
    """Render an :class:`InboxItem` into an :class:`AgentMessage`.

    Handles ``source="user"`` → :class:`UserMessage`,
    ``source="background"`` → a
    ``<system-reminder source="background">``-wrapped :class:`UserMessage`
    for auto-backgrounding completion/status,
    ``source="monitor"`` → ``<system-reminder source="monitor">`` for
    agent-defined wakeups + channel subscriptions, and
    ``source="subagent"`` → ``<system-reminder source="subagent">`` for
    child-task findings posted by ``sub_agent._finalize_state``. All four land
    as new ``user`` messages so the prefix stays stable and the KV/prefix cache
    survives (no synthetic ``tool_result``, which would need a live
    ``tool_call_id`` the inbox does not carry at drain time).

    The ``source="..."`` attribute on the wrapper tag exists so the agent
    can distinguish producer classes textually — first surfaced as an E2E
    finding (#176 post-merge validation): without the tag the agent has to
    guess from payload shape whether a reminder is a bg completion vs a
    monitor wakeup vs a sub_agent finding. Keep this attribute stable —
    downstream prompts / extractors may key off it.

    Any other source raises so a mis-routed item fails loudly rather than
    landing wrong.
    """

    if item.source == "user":
        content: list[Any]
        if isinstance(item.payload, str):
            content = [TextContent(type="text", text=item.payload)]
        else:
            content = list(item.payload)
        return UserMessage(role="user", content=content, timestamp=time.time())

    if item.source in ("background", "monitor", "subagent"):
        text = item.payload if isinstance(item.payload, str) else str(item.payload)
        wrapped = (
            f'<system-reminder source="{item.source}">\n{text}\n</system-reminder>'
        )
        return UserMessage(
            role="user",
            content=[TextContent(type="text", text=wrapped)],
            timestamp=time.time(),
        )

    raise NotImplementedError(
        f"SessionInbox.render_item: source {item.source!r} is not handled "
        f"(only 'user' / 'background' / 'monitor' / 'subagent')."
    )


__all__ = ["InboxItem", "InboxSource", "SessionInbox", "render_item"]
