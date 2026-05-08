"""Fail-stop for ``TurnEndEvent.messages`` contract.

Locks down the snapshot every per-turn extension uses to inspect the live
trajectory mid-loop. The kernel persists messages to the SessionManager
in one batch *after* ``prompt()`` returns, so ``api.session.get_messages()``
inside a turn_end handler reflects only the initial user prompt — which
silently breaks any extension that slices "what was new this turn" from
the session view (cognitive-audit extractor cursor, drift heuristics,
mid-loop reminders). The fix is to carry the live trajectory on the event
itself; this test enforces that contract end-to-end through one
``AgentLoop.run`` invocation so a future refactor that re-orders emit /
append in loop.py cannot silently regress every consumer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AgentLoop,
    AssistantMessage,
    EventBus,
    LoopConfig,
    MessageEnd,
    Model,
    TextContent,
    TurnEndEvent,
    text_message,
)


class _Stream:
    def __init__(self, scripted: list[AssistantMessage]) -> None:
        self._scripted = scripted
        self.calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[MessageEnd]:
        del messages, model, tools, system, signal, thinking
        idx = self.calls
        self.calls += 1
        return self._iter(self._scripted[idx])

    async def _iter(self, message: AssistantMessage) -> AsyncIterator[MessageEnd]:
        yield MessageEnd(message=message)


def _model() -> Model:
    return Model(
        id="test", provider="fake", context_window=4096, max_output_tokens=512
    )


@pytest.mark.asyncio
async def test_turn_end_messages_carries_live_trajectory_including_new_assistant() -> None:
    bus = EventBus()
    snapshots: list[tuple[int, ...]] = []

    async def _capture(event: TurnEndEvent) -> None:
        # Record (turn_index, len(messages), id of last message) so we can
        # verify both growth and that the just-emitted assistant_msg is
        # included in the snapshot — the bug the field exists to prevent
        # is "snapshot lags behind by one turn."
        snapshots.append((event.turn_index, len(event.messages)))
        assert event.messages, "TurnEndEvent.messages must not be empty"
        assert event.messages[-1] is event.message, (
            "messages snapshot must include the just-emitted assistant_msg"
        )

    bus.on(TurnEndEvent.CHANNEL, _capture)

    scripted = [
        AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            timestamp=1.0,
            stop_reason="end_turn",
        )
    ]
    loop = AgentLoop(stream_fn=_Stream(scripted), bus=bus, config=LoopConfig(max_turns=1))
    result_messages = await loop.run(
        messages=[text_message("hi")],
        model=_model(),
        tools=[],
    )

    # turn_end fires after the assistant message is assembled. The
    # snapshot at that point must be the live trajectory (input user_msg
    # + the just-emitted assistant_msg), not a stale view containing only
    # the initial user message — that "stale view" was the actual prior
    # bug: extensions that read api.session.get_messages() inside this
    # handler saw len=1 because the kernel batches SessionManager
    # persistence to after prompt() returns.
    assert len(snapshots) == 1
    turn_index, snap_len = snapshots[0]
    assert turn_index == 0
    assert snap_len == 2  # user_msg + assistant_msg
    assert snap_len == len(result_messages)
