"""Offline driver smoke tests.

The closest deterministic proxy for the design's acceptance invariant
``#1`` (live ≡ offline equivalence) without burning LLM quota: drive
:func:`replay_pipeline_over_trajectory` over a fixture trajectory with
stub seams and assert the cadence + cumulative-state threading + sidecar
shape are what the design specifies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agentm.core.abi import (
    AssistantMessage,
    TextContent,
)
from agentm.core.abi.messages import AgentMessage

from llmharness.audit._atom_constants import EXTRACTOR_TOOLS_MODULE
from llmharness.audit._offline_seams import InMemorySink
from llmharness.audit._runner import AuditorSettings, ExtractorSettings
from llmharness.audit.extractor.state import ExtractionState
from llmharness.audit.graph_ops import NodeUpsert
from llmharness.replay.offline_driver import replay_pipeline_over_trajectory
from llmharness.replay.record import iter_records

_REPLAY_RECORD_KEYS = {
    "phase",
    "turn_index",
    "root_session_id",
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

    Extractor: locates the bound ``ExtractionState`` in the per-firing
    extensions list, appends one fresh ``NodeUpsert`` to its
    ``pending_ops`` (id = ``state.next_event_id``), and returns
    ``(True, [<one tool_call block>])``. The runner's own
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
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
    ) -> tuple[bool, list[dict[str, Any]]]:
        del provider
        self.extractor_calls += 1
        self.extractor_payloads.append(payload)

        state: ExtractionState | None = None
        for module, cfg in extensions:
            if module == EXTRACTOR_TOOLS_MODULE:
                candidate = cfg.get("state")
                if isinstance(candidate, ExtractionState):
                    state = candidate
                    break
        assert state is not None, "stub could not find bound ExtractionState"

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
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Any],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> tuple[Any, list[dict[str, Any]]]:
        del extensions, provider, recent_verdicts, continuation_notes_from_prior_firing
        self.auditor_calls += 1
        self.auditor_graphs.append(list(graph_events))

        from llmharness.schema import Verdict

        verdict = Verdict(
            surface_reminder=False,
            reminder_text="",
            continuation_notes=[f"audit-call-{self.auditor_calls}"],
            matched_event_ids=[],
            cited_cards=[],
        )
        raw_block = {
            "type": "tool_call",
            "id": f"verdict-{self.auditor_calls}",
            "name": "submit_verdict",
            "arguments": {"verdict": verdict.to_dict()},
        }
        return verdict, [raw_block]


@pytest.mark.asyncio
async def test_cadence_and_cumulative_threading(tmp_path: Path) -> None:
    messages = _make_trajectory(20)
    sidecar_path = tmp_path / "audit_replay.jsonl"
    sink = InMemorySink()
    stub = _StubChildRunner()

    extractor_settings = ExtractorSettings(
        extensions=[(EXTRACTOR_TOOLS_MODULE, {})],
        compose_kwargs={"base_prompt": "stub-extractor-prompt"},
    )
    auditor_settings = AuditorSettings(
        base_prompt="stub-auditor-prompt",
        cards_tools_config=None,
        observability_config=None,
        summary_threshold=30,
        tools=(),
    )

    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        root_session_id="test-session",
        provider=None,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=True,
        sidecar_path=sidecar_path,
        sink=sink,
        child=stub,
    )

    # No reminder ever surfaced (stub auditor always silent) so the run
    # walked all 20 turns.
    assert result.reminder is None
    assert len(result.all_step_results) == 20

    # Cadence: extractor + auditor fire at turns 5, 10, 15, 20.
    fired_turns = [
        i + 1 for i, step in enumerate(result.all_step_results) if step.fired_extractor
    ]
    assert fired_turns == [5, 10, 15, 20]

    assert stub.extractor_calls == 4
    assert stub.auditor_calls == 4

    # Sink got one op per firing, four cursor advances, four verdicts.
    assert len(sink.ops) == 4
    assert sink.cursors == [4, 9, 14, 19]
    assert len(sink.verdicts) == 4

    # Cumulative threading: the runner-held state should expose all four
    # stub-emitted nodes in graph_view, and each subsequent extractor
    # firing should have seen the prior firings' nodes in its
    # ``recent_graph`` payload.
    events, _edges, _phases = result.state.graph_view()
    assert len(events) == 4
    assert [e.summary for e in events] == [
        "stub-node-1",
        "stub-node-2",
        "stub-node-3",
        "stub-node-4",
    ]
    # First firing has empty recent_graph; subsequent ones see prior
    # nodes (re-threading invariant — the design defect chain_replay
    # called out is *not* present here).
    recent_lens = [
        len(payload.get("recent_graph") or []) for payload in stub.extractor_payloads
    ]
    assert recent_lens == [0, 1, 2, 3]

    # Sidecar: one extractor + one auditor record per firing → 8 lines,
    # each carrying all 13 ReplayRecord keys.
    raw_lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 8
    for line in raw_lines:
        d = json.loads(line)
        # to_jsonl strips empty extras / raw_assistant_messages; widen
        # back the minimum guaranteed key set.
        always_present = _REPLAY_RECORD_KEYS - {"extras", "raw_assistant_messages"}
        assert always_present.issubset(d.keys()), (
            f"missing keys: {always_present - d.keys()}"
        )

    # Round-trip via iter_records should yield 8 records with all 13
    # logical fields populated on the dataclass.
    records = list(iter_records(sidecar_path))
    assert len(records) == 8
    phases_seen = [r.phase for r in records]
    # extractor and auditor records interleave: e, a, e, a, e, a, e, a.
    assert phases_seen == ["extractor", "auditor"] * 4
    for r in records:
        # All 13 keys present on the dataclass form.
        as_dict = r.to_dict()
        assert set(as_dict.keys()) == _REPLAY_RECORD_KEYS
