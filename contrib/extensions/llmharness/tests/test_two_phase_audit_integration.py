"""V3 fail-stop integration test for the two-phase cognitive audit.

This pins Phase 1 (extractor) under the v3 witness pipeline. The four
scenarios cover the load-bearing transitions:

1. **Happy** — extractor calls ``register_event`` once, ``add_edge``
   once with a witnessable quote, then ``submit_extraction``. The
   adapter MUST persist exactly one ``audit_event``, one ``audit_edge``
   and one ``extractor_cursor`` (no ``extractor_partial``). Cursor
   advances to the last absolute trajectory index in the window.

2. **Partial** — extractor calls ``register_event`` once, then
   ``add_edge`` THREE times with an unwitnessable quote on the SAME
   ``(src, dst, kind)`` tuple, then ``submit_extraction``. The third
   failure trips the retry budget, the tuple is dropped, and the
   adapter MUST persist one ``audit_event``, ZERO ``audit_edge``, one
   ``extractor_partial`` (with the dropped tuple in
   ``dropped_edges``), and one ``extractor_cursor`` — the cursor
   advances per design §6 because the firing wrote events.

3. **No-call** — child returns without ever calling
   ``submit_extraction``. The adapter MUST persist one
   ``extractor_no_call`` entry; cursor must NOT advance, so the next
   firing re-attempts the same window.

4. **Empty** — child calls ``submit_extraction`` immediately on a
   non-trivial window without registering any event. The adapter MUST
   persist one ``extractor_empty`` entry; cursor unchanged.

Phase 2 (auditor) is out of scope for this commit — commit 4 rewires
the auditor prompt to consume v3 entries. We use ``audit_interval_turns``
large enough that the auditor never fires here.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig

from llmharness.audit.entry_types import (
    AUDIT_EDGE,
    AUDIT_EVENT,
    EXTRACTOR_CURSOR,
    EXTRACTOR_EMPTY,
    EXTRACTOR_NO_CALL,
    EXTRACTOR_PARTIAL,
)
from llmharness.audit.extractor import (
    ADD_EDGE_TOOL_NAME,
    REGISTER_EVENT_TOOL_NAME,
    SUBMIT_EXTRACTION_TOOL_NAME,
)

# --- shared constants -------------------------------------------------------

_EXTRACTOR_PROMPT_NEEDLE = "cognitive-audit **extractor**"
_AUDITOR_PROMPT_NEEDLE = "cognitive-audit *auditor*"

# Witnessable quote — main agent's parent reply is a fixed string that
# we reuse as the cited_quote so the witness layer accepts the edge.
_PARENT_REPLY_TEMPLATE = "main turn {n} says alpha bravo charlie"
_GOOD_QUOTE = "alpha bravo charlie"
_BAD_QUOTE = "this phrase will never appear in any turn xyzzy"


# --- stub provider ----------------------------------------------------------


class _V3StubProvider:
    """Stub StreamFn that branches on system prompt + extractor mode.

    The audit adapter spawns child sessions that inherit the parent's
    provider, so this single ``__call__`` services the parent agent
    AND every extractor child. Disambiguation is by system-prompt
    needle. Per-firing extractor steps are tracked via
    ``extractor_step_index`` (reset implicitly because every child
    session creates a fresh sequence in the order: register_event →
    add_edge(s) → submit_extraction).

    Note: each child session can stream multiple assistant messages —
    one per (tool_call, tool_result) round trip. The stub returns ONE
    MessageEnd per call; the AgentM kernel runs the tool, then calls
    the StreamFn again with the tool_result appended to ``messages``.
    """

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.parent_calls = 0
        # Per-extractor-firing step index. Reset by detecting "new"
        # child sessions via the absence of any extractor tool_result
        # in the messages history fed to the stream.
        self.extractor_calls = 0
        self.auditor_calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, signal, thinking
        sys_text = system or ""
        if _EXTRACTOR_PROMPT_NEEDLE in sys_text:
            step = _count_extractor_tool_results(messages)
            self.extractor_calls += 1
            return self._extractor_iter(step=step)
        if _AUDITOR_PROMPT_NEEDLE in sys_text:
            self.auditor_calls += 1
            return self._auditor_iter()
        self.parent_calls += 1
        return self._parent_iter(self.parent_calls)

    async def _parent_iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=_PARENT_REPLY_TEMPLATE.format(n=n))],
            timestamp=float(n),
            stop_reason="end_turn",
        )
        yield MessageEnd(message=msg)

    async def _extractor_iter(self, *, step: int) -> AsyncIterator[AssistantStreamEvent]:
        """Drive the v3 extractor child loop step-by-step.

        ``step`` is the count of extractor tool-results already in the
        messages history — i.e. how many tools we've already driven
        through.
        """
        if self.mode == "no_call":
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="declining to submit")],
                timestamp=300.0 + step,
                stop_reason="end_turn",
            )
            yield MessageEnd(message=msg)
            return

        if self.mode == "empty":
            msg = AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=f"submit-{step}",
                        name=SUBMIT_EXTRACTION_TOOL_NAME,
                        arguments={},
                    )
                ],
                timestamp=300.0 + step,
                stop_reason="tool_use",
            )
            yield MessageEnd(message=msg)
            return

        if self.mode == "happy":
            # Steps:
            #   0 -> register_event #1
            #   1 -> register_event #2
            #   2 -> add_edge (witness passes)
            #   3 -> submit_extraction
            if step == 0:
                yield MessageEnd(message=_register_event_call(step, summary="event 1"))
                return
            if step == 1:
                yield MessageEnd(message=_register_event_call(step, summary="event 2"))
                return
            if step == 2:
                yield MessageEnd(
                    message=_add_edge_call(
                        step,
                        src=1,
                        dst=2,
                        cited_quote=_GOOD_QUOTE,
                    )
                )
                return
            yield MessageEnd(message=_submit_call(step))
            return

        if self.mode == "partial":
            # Steps:
            #   0 -> register_event #1
            #   1 -> register_event #2
            #   2 -> add_edge (bad witness, attempt 1)
            #   3 -> add_edge (bad witness, attempt 2)
            #   4 -> add_edge (bad witness, attempt 3 - dropped)
            #   5 -> submit_extraction
            if step == 0:
                yield MessageEnd(message=_register_event_call(step, summary="event 1"))
                return
            if step == 1:
                yield MessageEnd(message=_register_event_call(step, summary="event 2"))
                return
            if step in (2, 3, 4):
                yield MessageEnd(
                    message=_add_edge_call(
                        step,
                        src=1,
                        dst=2,
                        cited_quote=_BAD_QUOTE,
                    )
                )
                return
            yield MessageEnd(message=_submit_call(step))
            return

        raise AssertionError(f"unknown stub mode {self.mode!r}")

    async def _auditor_iter(self) -> AsyncIterator[AssistantStreamEvent]:
        # Auditor should never fire in these tests (k is set high enough
        # that no firing happens). If it does, terminate cleanly so the
        # session shuts down without hanging.
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="(auditor stub — not exercised)")],
            timestamp=999.0,
            stop_reason="end_turn",
        )
        yield MessageEnd(message=msg)


def _count_extractor_tool_results(messages: list[Any]) -> int:
    """Count tool_result messages in the child session's history.

    The kernel hands the StreamFn the running message list; counting
    tool_result entries gives a robust per-firing step index without
    needing to thread state through the stub.
    """
    count = 0
    for msg in messages:
        # ``ToolResultMessage`` is the canonical type, but duck-typing
        # on tool_call_id is sufficient and avoids importing the symbol
        # in case AgentM evolves the class layout.
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if getattr(block, "tool_call_id", None) is not None:
                    count += 1
                    break
        else:
            # ToolResultMessage may carry tool_call_id at the message level.
            if getattr(msg, "tool_call_id", None) is not None:
                count += 1
    return count


def _register_event_call(step: int, *, summary: str = "synthetic event") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id=f"reg-{step}",
                name=REGISTER_EVENT_TOOL_NAME,
                arguments={
                    "turn_indices": [0, 1],
                    "kind": "evid",
                    "summary": summary,
                },
            )
        ],
        timestamp=400.0 + step,
        stop_reason="tool_use",
    )


def _add_edge_call(
    step: int,
    *,
    src: int,
    dst: int,
    cited_quote: str,
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id=f"edge-{step}",
                name=ADD_EDGE_TOOL_NAME,
                arguments={
                    "src_event_id": src,
                    "dst_event_id": dst,
                    "kind": "ref",
                    "reason": "synthetic edge",
                    "src_turns": [1],
                    "dst_turns": [1],
                    "cited_quote": cited_quote,
                },
            )
        ],
        timestamp=500.0 + step,
        stop_reason="tool_use",
    )


def _submit_call(step: int) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id=f"submit-{step}",
                name=SUBMIT_EXTRACTION_TOOL_NAME,
                arguments={},
            )
        ],
        timestamp=600.0 + step,
        stop_reason="tool_use",
    )


def _install_provider_module(name: str, provider: _V3StubProvider) -> str:
    """Register the stub as an AgentM provider extension module."""
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-v3",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-v3",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-v3",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _build_session_config(*, cwd: str, provider_module: str) -> AgentSessionConfig:
    """Build an AgentSessionConfig wired with the v3 audit adapter (sync mode).

    Sync mode keeps assertions deterministic without waiting for a
    background worker drain. ``audit_interval_turns`` is large enough
    that the auditor never fires inside a single-turn test.
    """
    return AgentSessionConfig(
        cwd=cwd,
        provider=(provider_module, {}),
        extensions=[
            (
                "llmharness.adapters.agentm",
                {
                    "mode": "sync",
                    "audit_interval_turns": 100,  # auditor never fires
                    "cards_tools_config": None,
                    "observability_config": None,
                },
            ),
        ],
    )


def _entries(session: AgentSession, entry_type: str) -> list[Any]:
    return [e for e in session.session_manager.get_active_branch() if e.type == entry_type]


# --- Scenario 1: happy path ------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_writes_event_edge_and_cursor(tmp_path: Path) -> None:
    provider = _V3StubProvider(mode="happy")
    provider_module = _install_provider_module("tests._fake_v3_happy_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )
    await session.prompt("user turn 1")
    await session.shutdown()

    events = _entries(session, AUDIT_EVENT)
    edges = _entries(session, AUDIT_EDGE)
    partial = _entries(session, EXTRACTOR_PARTIAL)
    cursors = _entries(session, EXTRACTOR_CURSOR)

    assert len(events) == 2, f"expected 2 audit_event, got {len(events)}: {events}"
    assert len(edges) == 1, f"expected 1 audit_edge, got {len(edges)}: {edges}"
    assert len(partial) == 0, f"expected 0 extractor_partial, got {len(partial)}"
    assert len(cursors) == 1, f"expected 1 extractor_cursor, got {len(cursors)}"

    # Cursor advanced to the last absolute trajectory index of the window.
    cursor_payload = cursors[0].payload
    assert isinstance(cursor_payload, dict)
    assert cursor_payload["last_turn_index"] >= 1, (
        f"cursor must cover at least the assistant turn (>=1), got {cursor_payload}"
    )

    # No failure entries.
    assert _entries(session, EXTRACTOR_NO_CALL) == []
    assert _entries(session, EXTRACTOR_EMPTY) == []


# --- Scenario 2: partial (witness retry exhausted) -------------------------


@pytest.mark.asyncio
async def test_partial_path_drops_edge_and_writes_extractor_partial(
    tmp_path: Path,
) -> None:
    provider = _V3StubProvider(mode="partial")
    provider_module = _install_provider_module("tests._fake_v3_partial_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )
    await session.prompt("user turn 1")
    await session.shutdown()

    events = _entries(session, AUDIT_EVENT)
    edges = _entries(session, AUDIT_EDGE)
    partial = _entries(session, EXTRACTOR_PARTIAL)
    cursors = _entries(session, EXTRACTOR_CURSOR)

    assert len(events) == 2, (
        f"expected 2 audit_event (registered before retries), got {len(events)}"
    )
    assert len(edges) == 0, f"expected 0 audit_edge after dropped tuple, got {len(edges)}"
    assert len(partial) == 1, f"expected exactly 1 extractor_partial, got {len(partial)}"
    assert len(cursors) == 1, (
        "cursor must advance on partial firings (design §6) so we don't loop on the same window"
    )

    # The partial entry must list the dropped (src, dst, kind) tuple
    # and carry the turn_window.
    partial_payload = partial[0].payload
    assert isinstance(partial_payload, dict)
    dropped = partial_payload.get("dropped_edges")
    assert isinstance(dropped, list) and len(dropped) == 1, (
        f"expected one dropped edge in extractor_partial payload, got {dropped}"
    )
    assert dropped[0]["src"] == 1
    assert dropped[0]["dst"] == 2
    assert dropped[0]["kind"] == "ref"
    assert "turn_window" in partial_payload


# --- Scenario 3: no-call ---------------------------------------------------


@pytest.mark.asyncio
async def test_no_call_path_records_extractor_no_call_and_holds_cursor(
    tmp_path: Path,
) -> None:
    provider = _V3StubProvider(mode="no_call")
    provider_module = _install_provider_module("tests._fake_v3_no_call_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )
    await session.prompt("user turn 1")
    await session.shutdown()

    no_call = _entries(session, EXTRACTOR_NO_CALL)
    cursors = _entries(session, EXTRACTOR_CURSOR)
    events = _entries(session, AUDIT_EVENT)
    edges = _entries(session, AUDIT_EDGE)

    assert len(no_call) == 1, f"expected exactly 1 extractor_no_call, got {len(no_call)}"
    assert len(cursors) == 0, "cursor must NOT advance when submit_extraction was never called"
    assert events == []
    assert edges == []

    payload = no_call[0].payload
    assert isinstance(payload, dict)
    assert "turn_window" in payload


# --- Scenario 4: empty (terminator called without registering) -------------


@pytest.mark.asyncio
async def test_empty_path_records_extractor_empty_on_non_trivial_window(
    tmp_path: Path,
) -> None:
    provider = _V3StubProvider(mode="empty")
    provider_module = _install_provider_module("tests._fake_v3_empty_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )
    await session.prompt("user turn 1")
    await session.shutdown()

    empty = _entries(session, EXTRACTOR_EMPTY)
    cursors = _entries(session, EXTRACTOR_CURSOR)
    events = _entries(session, AUDIT_EVENT)
    edges = _entries(session, AUDIT_EDGE)

    assert len(empty) == 1, f"expected exactly 1 extractor_empty, got {len(empty)}"
    assert len(cursors) == 0, "cursor must NOT advance on extractor_empty"
    assert events == []
    assert edges == []
