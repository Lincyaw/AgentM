"""Offline driver smoke tests.

The closest deterministic proxy for the design's acceptance invariant
``#1`` (live ≡ offline equivalence) without burning LLM quota: drive
:func:`replay_pipeline_over_trajectory` over a fixture trajectory with
stub seams and assert the cadence + cumulative-state threading + sidecar
shape are what the design specifies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from agentm.core.abi import (
    AssistantMessage,
    TextContent,
)
from agentm.core.abi.messages import AgentMessage

from llmharness.agents.extractor.state import ExtractionState
from llmharness.graph.ops import NodeUpsert
from llmharness.replay.offline_driver import replay_pipeline_over_trajectory
from llmharness.runtime.offline import InMemorySink
from llmharness.runtime.runner import AuditorSettings, ExtractorSettings
from llmharness.schema import Event

_REPLAY_RECORD_KEYS = {
    "phase",
    "turn_index",
    "session_id",
    "trace_id",
    "ts_ns",
    "compose_kwargs",
    "payload",
    "provider",
    "output",
    "status",
    "error",
    "latency_ms",
    "extras",
    "raw_assistant_messages",
}


def _make_trajectory(n_assistant_turns: int) -> list[AgentMessage]:
    """Hand-craft ``n`` assistant messages.

    Every turn is a non-trivial assistant message so the extractor
    won't short-circuit on the ``trivial window`` branch. ``turn_count``
    in the runner is wall-clock turns (not just assistant); for this
    smoke test we feed only assistant messages so ``len(messages) ==
    turn_count`` and the cadence math is unambiguous.
    """
    msgs: list[AgentMessage] = []
    for i in range(n_assistant_turns):
        msgs.append(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"turn {i} narrative")],
                timestamp=float(i),
                stop_reason="end_turn",
            )
        )
    return msgs


class _StubChildRunner:
    """Canned :class:`ChildRunner` stub.

    Extractor: receives the bound ``ExtractionState`` directly via the
    ``state`` parameter (domain-level protocol), appends one fresh
    ``NodeUpsert`` to its ``pending_ops`` (id = ``state.next_event_id``),
    and returns ``(True, [<one tool_call block>])``. The runner's own
    ``RawExtractorOutput.from_state`` will read the state right after.

    Auditor: returns a silent verdict (``surface_reminder=False``).
    """

    def __init__(self) -> None:
        self.extractor_calls = 0
        self.auditor_calls = 0
        self.extractor_payloads: list[dict[str, Any]] = []
        self.auditor_graphs: list[list[Any]] = []

    async def run_extractor(
        self,
        *,
        state: ExtractionState,
        prompt_text: str,
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
        tool_call_budget: int | None = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        del provider, prompt_text, tool_call_budget
        self.extractor_calls += 1
        self.extractor_payloads.append(payload)

        node_id = state.next_event_id
        state.pending_ops.append(
            NodeUpsert(
                id=node_id,
                kind="task",
                summary=f"stub-node-{node_id}",
                source_turns=tuple(range(turn_window[0], turn_window[1] + 1)),
            )
        )

        raw_block = {
            "type": "tool_call",
            "id": f"call-{self.extractor_calls}",
            "name": "finalize_extraction",
            "arguments": {},
        }
        return True, [raw_block]

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Event],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> Any:
        del prompt_text, tools_config, provider
        del recent_verdicts, continuation_notes_from_prior_firing
        self.auditor_calls += 1
        self.auditor_graphs.append(list(graph_events))

        from llmharness.runtime.runner import AuditorChildResult
        from llmharness.schema import Verdict

        verdict = Verdict(
            surface_reminder=False,
            reminder_text="",
            continuation_notes=[f"audit-call-{self.auditor_calls}"],
            matched_event_ids=[],
        )
        raw_block = {
            "type": "tool_call",
            "id": f"verdict-{self.auditor_calls}",
            "name": "submit_verdict",
            "arguments": {"verdict": verdict.to_dict()},
        }
        return AuditorChildResult(verdict=verdict, raw_blocks=[raw_block])


class _SurfacingStubChildRunner(_StubChildRunner):
    """:class:`_StubChildRunner` variant whose auditor surfaces on the
    first firing.

    Extractor behaviour is inherited verbatim; only the auditor differs.
    Used to exercise ``stop_on_first_surface=True``'s early-break path.
    """

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Event],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> Any:
        del prompt_text, tools_config, provider
        del recent_verdicts, continuation_notes_from_prior_firing
        self.auditor_calls += 1
        self.auditor_graphs.append(list(graph_events))

        from llmharness.runtime.runner import AuditorChildResult
        from llmharness.schema import Verdict

        verdict = Verdict(
            surface_reminder=True,
            reminder_text="halt",
            continuation_notes=[],
            matched_event_ids=[],
        )
        raw_block = {
            "type": "tool_call",
            "id": f"verdict-{self.auditor_calls}",
            "name": "submit_verdict",
            "arguments": {"verdict": verdict.to_dict()},
        }
        return AuditorChildResult(verdict=verdict, raw_blocks=[raw_block])


@pytest.mark.asyncio
async def test_replay_pipeline_breaks_on_first_surface(tmp_path: Path) -> None:
    """``stop_on_first_surface=True`` must short-circuit on the first
    surfaced reminder.

    With ``extractor_interval=5`` and ``audit_interval=5``, the auditor
    fires first at turn 5. The surfacing stub returns
    ``surface_reminder=True`` on that first firing, so the run must
    halt at turn 5 — well before the 30-message trajectory ends.
    """
    messages = _make_trajectory(30)
    sink = InMemorySink()
    stub = _SurfacingStubChildRunner()

    extractor_settings = ExtractorSettings(

        compose_kwargs={"base_prompt": "stub-extractor-prompt"},
        base_prompt="stub-extractor-prompt",
    )
    auditor_settings = AuditorSettings(
        base_prompt="stub-auditor-prompt",
        observability_config=None,
        summary_threshold=30,
        tools=(),
    )

    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        session_id="test-session",
        provider=None,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=True,
        sidecar_path=None,
        sink=sink,
        child=stub,
    )

    assert result.reminder is not None
    assert result.reminder.text == "halt"
    # First auditor firing happens at turn 5 (cadence), so the loop
    # records steps for turns 1..5 inclusive and then breaks.
    assert len(result.all_step_results) == 5
    # The trajectory was 30 messages long — confirm we did break early.
    assert len(result.all_step_results) < len(messages)
    # Only one auditor firing happened before the break.
    assert stub.auditor_calls == 1
    # The surfacing step is the last one recorded.
    assert result.all_step_results[-1].surfaced_reminder is not None
    assert result.all_step_results[-1].surfaced_reminder.text == "halt"


@pytest.mark.asyncio
async def test_historical_source_turn_texts_elided_from_payload(tmp_path: Path) -> None:
    """Historical (prior-firing) node source turns carry no raw text in the
    extractor payload; the current window's raw text still ships via
    ``new_turns``.

    Fail-stop position: the extractor payload must not grow with run length
    by re-shipping every cumulative node's full source-turn text on every
    firing. A node folded into the graph in an earlier firing is represented
    by its ``summary`` + ``external_refs``; re-sending its raw turn text was
    the dominant, unbounded extractor cost.
    """
    import json

    messages = _make_trajectory(15)  # extractor_interval=5 → firings at 5/10/15
    sink = InMemorySink()
    stub = _StubChildRunner()

    extractor_settings = ExtractorSettings(

        compose_kwargs={"base_prompt": "stub-extractor-prompt"},
        base_prompt="stub-extractor-prompt",
    )
    auditor_settings = AuditorSettings(
        base_prompt="stub-auditor-prompt",
        observability_config=None,
        summary_threshold=30,
        tools=(),
    )

    await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        session_id="test-session",
        provider=None,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=False,
        sidecar_path=None,
        sink=sink,
        child=stub,
    )

    # Need at least two firings so a later payload carries prior-firing nodes.
    assert stub.extractor_calls >= 3
    last_payload = stub.extractor_payloads[-1]
    recent = last_payload.get("recent_graph") or []
    assert recent, "expected prior-firing nodes in recent_graph"

    # Every historical node's source-turn texts are elided to empty strings.
    for node in recent:
        stt = node.get("source_turn_texts")
        assert stt is not None
        assert all(t == "" for t in stt), f"historical node kept raw text: {node}"

    # The current window's raw text is NOT lost — it still ships via new_turns.
    new_turns_blob = json.dumps(last_payload.get("new_turns") or [])
    assert "narrative" in new_turns_blob
