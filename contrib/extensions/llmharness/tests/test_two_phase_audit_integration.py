"""V3.1 fail-stop integration test for the two-phase cognitive audit.

Pins Phase 1 (extractor) under the v3.1 single-tool flow. The four
scenarios cover the load-bearing transitions:

1. **Happy** — extractor calls ``submit_events`` once with a witnessable
   ref. Adapter MUST persist exactly one ``audit_event`` per submitted
   event, one ``audit_edge`` per accepted ref, and one
   ``extractor_cursor``. Cursor advances to the last absolute trajectory
   index in the window.

2. **Partial** — extractor calls ``submit_events`` once with a bad
   witness on the only ref. The ref is dropped (witness fails), but
   the events stay. Adapter MUST persist the events as ``audit_event``
   entries, ZERO ``audit_edge``, one ``extractor_partial`` (with the
   dropped tuple), and one ``extractor_cursor`` — cursor advances per
   design §6 because the firing wrote events.

3. **No-call** — child returns without ever calling ``submit_events``.
   Adapter MUST persist one ``extractor_no_call`` entry; cursor must
   NOT advance.

4. **Empty** — child calls ``submit_events`` immediately with an empty
   events list on a non-trivial window. Adapter MUST persist one
   ``extractor_empty`` entry; cursor unchanged.

Phase 2 (auditor) is out of scope here — ``audit_interval_turns`` is
set high enough that the auditor never fires.
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
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession

from llmharness.audit.auditor import SUBMIT_VERDICT_TOOL_NAME
from llmharness.audit.entry_types import (
    AUDIT_EDGE,
    AUDIT_EVENT,
    EXTRACTOR_CURSOR,
    EXTRACTOR_EMPTY,
    EXTRACTOR_NO_CALL,
    EXTRACTOR_PARTIAL,
    VERDICT,
)
from llmharness.audit.extractor import FINALIZE_EXTRACTION_TOOL_NAME

# --- shared constants -------------------------------------------------------

_EXTRACTOR_PROMPT_NEEDLE = "cognitive-audit **extractor**"
_AUDITOR_PROMPT_NEEDLE = "cognitive-audit *auditor*"

# Witnessable quote — the parent's reply embeds this fixed phrase so a
# ``ref`` ref using it as ``cited_quote`` will pass the witness check.
_PARENT_REPLY_TEMPLATE = "main turn {n} says alpha bravo charlie"
_GOOD_QUOTE = "alpha bravo charlie"
_BAD_QUOTE = "this phrase will never appear in any turn xyzzy"


# --- stub provider ----------------------------------------------------------


class _V31StubProvider:
    """Stub StreamFn that branches on system prompt + extractor mode.

    The audit adapter spawns child sessions that inherit the parent's
    provider, so this single ``__call__`` services the parent agent
    AND every extractor / auditor child. Disambiguation is by
    system-prompt needle.
    """

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.parent_calls = 0
        self.extractor_calls = 0
        self.auditor_calls = 0
        # Cursor into the scripted extractor message sequence — bumps once
        # per extractor child turn so a multi-tool firing yields the
        # planned tool_calls in order (upsert_node -> upsert_edge ->
        # ... -> finalize_extraction). Reset per child firing because
        # each spawn rebuilds the script from scratch.
        self._extractor_step = 0
        self._last_extractor_mode: str | None = None

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
        del messages, model, tools, signal, thinking
        sys_text = system or ""
        if _EXTRACTOR_PROMPT_NEEDLE in sys_text:
            self.extractor_calls += 1
            return self._extractor_iter()
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

    async def _extractor_iter(self) -> AsyncIterator[AssistantStreamEvent]:
        # Reset the cursor whenever we cross into a fresh extractor child
        # firing. The provider has no native "child started" hook so we
        # detect it by mode change OR by stepping past the script length.
        if self._last_extractor_mode != self.mode:
            self._extractor_step = 0
            self._last_extractor_mode = self.mode

        script = self._extractor_script()
        if self._extractor_step >= len(script):
            # Out of scripted turns — surface a no-tool message so the
            # child loop terminates without further tool calls. Used by
            # the ``no_call`` mode.
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="declining to submit")],
                timestamp=300.0,
                stop_reason="end_turn",
            )
            self._extractor_step = 0  # reset for next firing
            yield MessageEnd(message=msg)
            return

        message = script[self._extractor_step]
        self._extractor_step += 1
        # If we just emitted the terminator, reset so the next firing
        # starts at step 0.
        for blk in message.content:
            if isinstance(blk, ToolCallBlock) and blk.name == FINALIZE_EXTRACTION_TOOL_NAME:
                self._extractor_step = 0
                self._last_extractor_mode = None
                break
        yield MessageEnd(message=message)

    def _extractor_script(self) -> list[AssistantMessage]:
        """The scripted assistant-message sequence for one extractor firing."""
        if self.mode == "no_call":
            return []

        if self.mode == "empty":
            # Empty firing under v19 = jump straight to finalize_extraction.
            return [_tool_call_message("finalize-empty", FINALIZE_EXTRACTION_TOOL_NAME, {})]

        if self.mode == "happy":
            # Walks: upsert two evid nodes, link them with a ref edge
            # carrying a witnessable quote, finalize. The two events form
            # a single (in_deg, out_deg) = (0,1) / (1,0) pair so the
            # degree check passes.
            return [
                _tool_call_message(
                    "node-1",
                    "upsert_node",
                    {
                        "id": 1,
                        "kind": "evid",
                        "summary": "event 1",
                        "source_turns": [0, 1],
                    },
                ),
                _tool_call_message(
                    "node-2",
                    "upsert_node",
                    {
                        "id": 2,
                        "kind": "evid",
                        "summary": "event 2",
                        "source_turns": [0, 1],
                    },
                ),
                _tool_call_message(
                    "edge-1",
                    "upsert_edge",
                    {
                        "src": 1,
                        "dst": 2,
                        "kind": "ref",
                        "reason": "synthetic ref",
                        "cited_entities": [],
                        "cited_quote": _GOOD_QUOTE,
                    },
                ),
                _tool_call_message(
                    "finalize-1", FINALIZE_EXTRACTION_TOOL_NAME, {}
                ),
            ]

        if self.mode == "partial":
            # Two upserts, an edge with a bad witness (rejected by
            # apply_edge_upsert), then finalize. The edge rejection
            # surfaces as a tool_result error; the events still land via
            # the op log. We don't retry — finalize follows immediately
            # so the firing's ``dropped_edges`` snapshot stays empty
            # and the surviving event-only firing routes through the
            # adapter's "no edges" branch.
            return [
                _tool_call_message(
                    "node-1",
                    "upsert_node",
                    {
                        "id": 1,
                        "kind": "evid",
                        "summary": "event 1",
                        "source_turns": [0, 1],
                    },
                ),
                _tool_call_message(
                    "node-2",
                    "upsert_node",
                    {
                        "id": 2,
                        "kind": "evid",
                        "summary": "event 2",
                        "source_turns": [0, 1],
                    },
                ),
                _tool_call_message(
                    "edge-bad",
                    "upsert_edge",
                    {
                        "src": 1,
                        "dst": 2,
                        "kind": "ref",
                        "reason": "synthetic ref with bad witness",
                        "cited_entities": [],
                        "cited_quote": _BAD_QUOTE,
                    },
                ),
                _tool_call_message(
                    "finalize-2", FINALIZE_EXTRACTION_TOOL_NAME, {}
                ),
            ]

        raise AssertionError(f"unknown stub mode {self.mode!r}")

    async def _auditor_iter(self) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=_submit_verdict_call())


def _tool_call_message(
    call_id: str, name: str, arguments: dict[str, Any]
) -> AssistantMessage:
    """One assistant turn whose content is a single tool_call block."""
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id=call_id,
                name=name,
                arguments=arguments,
            )
        ],
        timestamp=600.0,
        stop_reason="tool_use",
    )


def _submit_verdict_call() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id="verdict-1",
                name=SUBMIT_VERDICT_TOOL_NAME,
                arguments={
                    "verdict": {
                        "surface_reminder": False,
                        "reminder_text": "",
                        "continuation_notes": ["stub auditor saw the graph"],
                        "matched_event_ids": [],
                        "cited_cards": [],
                    }
                },
            )
        ],
        timestamp=999.0,
        stop_reason="tool_use",
    )


def _install_provider_module(name: str, provider: _V31StubProvider) -> str:
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
    """Wire the v3.1 audit adapter in sync mode for deterministic asserts."""
    return AgentSessionConfig(
        cwd=cwd,
        provider=(provider_module, {}),
        extensions=[
            # Satisfy the adapter's MANIFEST.requires (observability,
            # operations_local, system_prompt) so the session-factory's
            # requires-ordering check passes. The audit child still
            # mounts its own copies via ``compose_audit_extensions``;
            # these are for the host session only.
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations_local", {}),
            ("agentm.extensions.builtin.system_prompt", {"prompt": ""}),
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


def _build_interval_session_config(
    *,
    cwd: str,
    provider_module: str,
    interval_turns: int,
) -> AgentSessionConfig:
    config = _build_session_config(cwd=cwd, provider_module=provider_module)
    extensions = list(config.extensions)
    extensions[-1] = (
        "llmharness.adapters.agentm",
        {
            "mode": "sync",
            "extractor_interval_turns": interval_turns,
            "audit_interval_turns": interval_turns,
            "cards_tools_config": None,
            "observability_config": None,
        },
    )
    config.extensions = extensions
    return config


def _entries(session: AgentSession, entry_type: str) -> list[Any]:
    return [e for e in session.session_manager.get_active_branch() if e.type == entry_type]


@pytest.mark.asyncio
async def test_extractor_and_auditor_fire_together_on_configured_interval(
    tmp_path: Path,
) -> None:
    provider = _V31StubProvider(mode="happy")
    provider_module = _install_provider_module(
        "tests._fake_v31_interval_provider", provider
    )

    session = await AgentSession.create(
        _build_interval_session_config(
            cwd=str(tmp_path),
            provider_module=provider_module,
            interval_turns=3,
        )
    )
    await session.prompt("user turn 1")
    await session.prompt("user turn 2")
    await session.prompt("user turn 3")
    await session.shutdown()

    assert provider.parent_calls == 3
    # v19: each extractor firing spans multiple LLM calls (the four-step
    # happy script: 2 upserts + 1 edge + finalize). The auditor is still
    # single-shot — one call per firing.
    assert provider.extractor_calls == 4
    assert provider.auditor_calls == 1
    assert len(_entries(session, AUDIT_EVENT)) == 2
    assert len(_entries(session, VERDICT)) == 1


# --- Scenario 1: happy path ------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_writes_event_edge_and_cursor(tmp_path: Path) -> None:
    provider = _V31StubProvider(mode="happy")
    provider_module = _install_provider_module("tests._fake_v31_happy_provider", provider)

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

    cursor_payload = cursors[0].payload
    assert isinstance(cursor_payload, dict)
    assert cursor_payload["last_turn_index"] >= 1, (
        f"cursor must cover at least the assistant turn (>=1), got {cursor_payload}"
    )

    assert _entries(session, EXTRACTOR_NO_CALL) == []
    assert _entries(session, EXTRACTOR_EMPTY) == []

    # Replay sidecar contract: the live adapter must write at least one
    # extractor record under .agentm/audit_replay/<root>.jsonl with the
    # parsed submit_events output preserved verbatim. Without this the
    # offline replay CLI would silently have no inputs.
    from llmharness.replay.record import iter_records

    replay_dir = tmp_path / ".agentm" / "audit_replay"
    assert replay_dir.exists(), "audit_replay/ directory was not created"
    sidecars = list(replay_dir.glob("*.jsonl"))
    assert len(sidecars) == 1, f"expected exactly one sidecar, got {sidecars}"
    records = list(iter_records(sidecars[0]))
    assert records, "sidecar file is empty"
    extractor_records = [r for r in records if r.phase == "extractor"]
    assert extractor_records, "no extractor record written to sidecar"
    rec = extractor_records[-1]
    assert rec.status == "ok"
    assert rec.output is not None
    assert "events" in rec.output and len(rec.output["events"]) == 2
    assert "edges" in rec.output and len(rec.output["edges"]) == 1


# --- Scenario 2: partial (witness retry exhausted) -------------------------


@pytest.mark.asyncio
async def test_partial_path_drops_edge_and_writes_extractor_partial(
    tmp_path: Path,
) -> None:
    provider = _V31StubProvider(mode="partial")
    provider_module = _install_provider_module(
        "tests._fake_v31_partial_provider", provider
    )

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )
    await session.prompt("user turn 1")
    await session.shutdown()

    events = _entries(session, AUDIT_EVENT)
    edges = _entries(session, AUDIT_EDGE)
    partial = _entries(session, EXTRACTOR_PARTIAL)
    cursors = _entries(session, EXTRACTOR_CURSOR)

    # v19: a bad-witness upsert_edge is rejected at the per-edit
    # boundary (the upsert_edge tool returns is_error=True) so no edge
    # lands in the op log AND no ``dropped_edges`` entry accumulates —
    # the v18 partial-accept path is gone. The two upsert_node calls
    # still succeed, so the firing finalizes with 2 events / 0 edges.
    # ``extractor_partial`` therefore stays empty under v19.
    assert len(events) == 2, (
        f"expected 2 audit_event (the upsert_node calls), got {len(events)}"
    )
    assert len(edges) == 0, f"expected 0 audit_edge after rejected ref, got {len(edges)}"
    assert len(partial) == 0, (
        "v19 has no partial-accept path; bad witnesses become tool "
        "rejections, not extractor_partial entries"
    )
    assert len(cursors) == 1, (
        "cursor must advance once the firing finalizes, even with no edges"
    )


# --- Scenario 3: no-call ---------------------------------------------------


@pytest.mark.asyncio
async def test_no_call_path_records_extractor_no_call_and_holds_cursor(
    tmp_path: Path,
) -> None:
    provider = _V31StubProvider(mode="no_call")
    provider_module = _install_provider_module(
        "tests._fake_v31_no_call_provider", provider
    )

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
    assert len(cursors) == 0, "cursor must NOT advance when submit_events was never called"
    assert events == []
    assert edges == []

    payload = no_call[0].payload
    assert isinstance(payload, dict)
    assert "turn_window" in payload


# --- Scenario 4: empty (terminator called with empty events) ---------------


@pytest.mark.asyncio
async def test_empty_path_records_extractor_empty_on_non_trivial_window(
    tmp_path: Path,
) -> None:
    provider = _V31StubProvider(mode="empty")
    provider_module = _install_provider_module(
        "tests._fake_v31_empty_provider", provider
    )

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
