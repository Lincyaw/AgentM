"""Fail-stop tests for the chained-fork driver + sidecar writer.

Why these are load-bearing: chained_fork is the host-side driver for
multi-segment intervention experiments. Single-segment behaviour
(cadence, sidecar shape, stop_on_first_surface) is already covered by
``test_offline_driver.py``; this file pins the *inter-segment* contract
that is unique to chained-fork:

* ``replay_pipeline_over_trajectory`` honors ``seed_cumulative`` and
  ``start_turn`` — segment k+1 must skip turns already replayed under
  segment k and must see segment k's verdicts in the new firing's
  payload. Wrong → segment k+1 compares to a blank state and the
  experiment compares apples to oranges.
* ``run_chained_fork_experiment`` terminates correctly: 0 reminders →
  control only; N reminders → at most ``max_interventions`` branches
  appended.
* ``write_chained_replay`` filters each segment's records to its
  ``[turn_lo, turn_hi]`` window and rebinds every record to the final
  segment's session_log_id. Wrong → the case viewer joins records to
  the wrong trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from agentm.core.abi import AssistantMessage, TextContent
from agentm.core.abi.messages import (
    AgentMessage,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)

from llmharness.audit.extractor.state import ExtractionState
from llmharness.audit.graph.ops import NodeUpsert
from llmharness.audit.runner import (
    AuditorChildResult,
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    StepResult,
)
from llmharness.audit.seams.offline import InMemorySink
from llmharness.audit.toolkit.atom_constants import EXTRACTOR_TOOLS_MODULE
from llmharness.replay import chained_fork as _chained_fork_mod
from llmharness.replay.chained_fork import (
    CHAIN_HEADER_KEY,
    ChainSegment,
    _fork_prefix,
    read_chain_header,
    run_chained_fork_experiment,
    write_chained_replay,
)
from llmharness.replay.offline_driver import (
    OfflineRunResult,
    replay_pipeline_over_trajectory,
)
from llmharness.replay.record import ReplayRecord, iter_records
from llmharness.schema import Reminder, Verdict

# ---------------------------------------------------------------------------
# Test 1: replay_pipeline_over_trajectory honors seed_cumulative + start_turn


def _make_trajectory(n_assistant_turns: int) -> list[AgentMessage]:
    msgs: list[AgentMessage] = []
    for i in range(n_assistant_turns):
        msgs.append(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"turn {i}")],
                timestamp=float(i),
                stop_reason="end_turn",
            )
        )
    return msgs


class _NodeAddingStubChildRunner:
    """Stub that adds one NodeUpsert per extractor firing + silent auditor."""

    def __init__(self) -> None:
        self.extractor_calls = 0
        self.auditor_calls = 0
        self.extractor_payloads: list[dict[str, Any]] = []

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
        assert state is not None

        node_id = state.next_event_id
        state.pending_ops.append(
            NodeUpsert(
                id=node_id,
                kind="task",
                summary=f"seed-node-{node_id}",
                source_turns=tuple(range(turn_window[0], turn_window[1] + 1)),
            )
        )
        return True, [{"type": "tool_call", "id": "x", "name": "finalize_extraction", "arguments": {}}]

    async def run_auditor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Any],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> Any:
        del extensions, provider, recent_verdicts, continuation_notes_from_prior_firing
        del graph_events
        self.auditor_calls += 1
        verdict = Verdict(
            surface_reminder=False,
            reminder_text="",
            continuation_notes=[],
            matched_event_ids=[],
        )
        return AuditorChildResult(verdict=verdict, raw_blocks=[])


@pytest.mark.asyncio
async def test_replay_pipeline_honors_seed_cumulative_and_start_turn(tmp_path: Path) -> None:
    """The fail-stop position: when chained_fork hands segment k+1 a
    pre-populated CumulativeAuditState plus ``start_turn=T+1``, the
    runner must (a) skip every cadence boundary at turns < T+1 and
    (b) include the seeded ops in the next extractor firing's
    ``recent_graph`` payload. Without (a) we double-replay turns
    already counted in segment k; without (b) the segment-boundary
    cumulative-state thread is broken.
    """
    messages = _make_trajectory(20)

    # Pre-seed the cumulative state with one Event so the next firing
    # sees recent_graph of length 1.
    seeded = CumulativeAuditState.fresh()
    seeded.absorb_extractor_firing(
        firing_ops=[
            NodeUpsert(
                id=1,
                kind="task",
                summary="prior-segment-node",
                source_turns=(0, 1, 2),
            )
        ],
        firing_cursor=5,
        firing_id=0,
    )

    sink = InMemorySink()
    stub = _NodeAddingStubChildRunner()

    extractor_settings = ExtractorSettings(
        extensions=[(EXTRACTOR_TOOLS_MODULE, {})],
        compose_kwargs={"base_prompt": "stub"},
    )
    auditor_settings = AuditorSettings(
        base_prompt="stub",
        observability_config=None,
        summary_threshold=30,
        tools=(),
    )

    # start_turn=11 → with cadence 5 the loop fires at turns 15 and 20
    # only (turns < 11 are skipped; turn 10 < start_turn).
    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        root_session_id="seg-2",
        provider=None,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=False,
        sidecar_path=None,
        sink=sink,
        child=stub,  # type: ignore[arg-type]  # structural ChildRunner
        seed_cumulative=seeded,
        start_turn=11,
    )

    # (a) start_turn honoured: the loop produced exactly len(11..20)=10
    # StepResults, and the auditor fired only at turns 15 and 20.
    assert len(result.all_step_results) == 10
    fired_extractor_turns = [
        i + 11
        for i, step in enumerate(result.all_step_results)
        if step.fired_extractor
    ]
    assert fired_extractor_turns == [15, 20]
    assert stub.extractor_calls == 2

    # (b) seed_cumulative honoured: the first firing's recent_graph
    # payload includes the seeded "prior-segment-node" event. (Without
    # the seed, recent_graph would start empty and only the fresh
    # firings would populate it.)
    first_payload = stub.extractor_payloads[0]
    recent = first_payload.get("recent_graph") or []
    assert len(recent) >= 1
    summaries = [e.get("summary") for e in recent]
    assert "prior-segment-node" in summaries


# ---------------------------------------------------------------------------
# Test 2: run_chained_fork_experiment segment count + termination


@dataclass
class _FakePayload:
    session_log_id: str
    final_messages: list[AgentMessage] = field(default_factory=list)


class _ScriptedReplay:
    """Drop-in replacement for ``replay_pipeline_over_trajectory``.

    Each call returns one canned :class:`OfflineRunResult`. The chained-
    fork driver invokes this once per segment (control + N branches), so
    the script length determines the chain length.
    """

    def __init__(self, scripted_reminders: list[Reminder | None]) -> None:
        self._scripted = list(scripted_reminders)
        self._call_index = 0
        self.calls_seen: list[dict[str, Any]] = []
        self.return_values: list[OfflineRunResult] = []
        # Per-segment surfacing turn: deterministic so callers can
        # assert ``start_turn == surface_turn + 1`` on the next call.
        self.surface_turns: list[int] = []

    async def __call__(self, **kwargs: Any) -> OfflineRunResult:
        idx = self._call_index
        self._call_index += 1
        self.calls_seen.append(kwargs)
        reminder = self._scripted[idx] if idx < len(self._scripted) else None
        # Construct a synthetic auditor record so chained_fork can
        # recover the surfacing turn via StepResult.auditor_record.
        if reminder is not None:
            # Surfacing turn = a deterministic value based on segment idx;
            # use 4, 9, 14, ... so the prefix slicing changes per
            # segment and we can verify the call ordering downstream.
            surface_turn = 4 + 5 * idx
            auditor_rec = ReplayRecord(
                phase="auditor",
                turn_index=surface_turn,
                root_session_id="stub",
                ts_ns=1,
                compose_kwargs={},
                payload={},
                provider=None,
                output={"surface_reminder": True, "reminder_text": reminder.text},
                status="ok",
            )
            step = StepResult(
                fired_extractor=True,
                fired_auditor=True,
                surfaced_reminder=reminder,
                auditor_record=auditor_rec,
            )
            steps = [step]
            self.surface_turns.append(surface_turn)
        else:
            steps = []
            self.surface_turns.append(-1)  # sentinel; never read for None
        # Distinct CumulativeAuditState per call so the threading
        # assertion can verify identity (segment k+1 must receive THE
        # state object returned by segment k, not a fresh blank).
        result = OfflineRunResult(
            reminder=reminder,
            state=CumulativeAuditState.fresh(),
            sidecar_path=None,
            all_step_results=steps,
        )
        self.return_values.append(result)
        return result


def _factory_returning_n_msgs(
    expected_calls: list[dict[str, Any]],
    message_lengths: list[int],
):
    counter = {"i": 0}

    async def factory(*, initial_messages, seed_reminder_text):
        i = counter["i"]
        counter["i"] += 1
        expected_calls.append(
            {
                "initial_messages_len": len(initial_messages) if initial_messages else None,
                "seed_reminder_text": seed_reminder_text,
            }
        )
        n = message_lengths[i] if i < len(message_lengths) else message_lengths[-1]
        msgs = _make_trajectory(n)
        return _FakePayload(session_log_id=f"seg-{i}", final_messages=msgs)

    return factory






@pytest.mark.asyncio
async def test_chained_fork_max_interventions_one_runs_control_plus_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_interventions=1`` must produce exactly one branch on top of
    the control, even if more reminders would be available downstream.
    """
    scripted = _ScriptedReplay(
        [Reminder(text="r1"), Reminder(text="r2")]
    )
    monkeypatch.setattr(
        _chained_fork_mod, "replay_pipeline_over_trajectory", scripted
    )
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_returning_n_msgs(factory_calls, [10, 12])

    experiment = await run_chained_fork_experiment(
        session_factory=factory,
        cwd=str(tmp_path),
        provider=None,
        extractor_settings=ExtractorSettings.default(),
        auditor_settings=AuditorSettings.default(),
        max_interventions=1,
    )
    assert len(experiment.segments) == 2
    assert experiment.segments[0].seeded_reminder is None
    assert experiment.segments[1].seeded_reminder is not None
    assert experiment.segments[1].seeded_reminder.text == "r1"
    assert len(factory_calls) == 2


@pytest.mark.asyncio
async def test_chained_fork_three_reminders_then_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When three reminders surface (control + 2 branches), then the
    fourth segment stays silent, the chain stops naturally at 4 segments
    even though ``max_interventions=10`` would allow more.
    """
    scripted = _ScriptedReplay(
        [
            Reminder(text="r1"),
            Reminder(text="r2"),
            Reminder(text="r3"),
            None,
        ]
    )
    monkeypatch.setattr(
        _chained_fork_mod, "replay_pipeline_over_trajectory", scripted
    )
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_returning_n_msgs(factory_calls, [10, 12, 14, 16])

    experiment = await run_chained_fork_experiment(
        session_factory=factory,
        cwd=str(tmp_path),
        provider=None,
        extractor_settings=ExtractorSettings.default(),
        auditor_settings=AuditorSettings.default(),
        max_interventions=10,
    )
    assert len(experiment.segments) == 4
    # Final segment had silent auditor → no reminder.
    assert experiment.segments[-1].surfaced_reminder_turn is None
    # The three branches carry their seeded reminders in order.
    assert [
        s.seeded_reminder.text if s.seeded_reminder else None
        for s in experiment.segments
    ] == [None, "r1", "r2", "r3"]

    # Fail-stop: the cumulative state of segment k must be the SAME
    # object passed as ``seed_cumulative`` to segment k+1's call, and
    # the next ``start_turn`` must equal segment k's surfacing turn
    # plus one full ``audit_interval`` (NOT +1). Without the cadence
    # shift the first iteration of the branch replay would fire on an
    # empty extractor window (cursor still at surfacing turn from the
    # seeded cumulative state) and the auditor would re-surface a
    # phantom reminder at the same turn the parent surfaced at,
    # creating a fork-at-same-turn loop. Validated empirically against
    # a live RCA case before the shift landed.
    # Control segment (idx 0) is called with seed_cumulative=None,
    # start_turn=1 — verify that explicitly too.
    assert scripted.calls_seen[0]["seed_cumulative"] is None
    assert scripted.calls_seen[0]["start_turn"] == 1
    audit_interval = 5  # matches the fixture's cadence
    for k in range(len(scripted.calls_seen) - 1):
        assert (
            scripted.calls_seen[k + 1]["seed_cumulative"]
            is scripted.return_values[k].state
        ), f"segment {k + 1} did not receive segment {k}'s cumulative state"
        expected = scripted.surface_turns[k] + audit_interval
        assert (
            scripted.calls_seen[k + 1]["start_turn"] == expected
        ), (
            f"segment {k + 1} start_turn={scripted.calls_seen[k + 1]['start_turn']}, "
            f"expected surface_turn+audit_interval={expected}"
        )


# ---------------------------------------------------------------------------
# Test 3: write_chained_replay filters by [turn_lo, turn_hi] + rebinds sid


def _ext_step(turn: int, root_id: str) -> StepResult:
    rec = ReplayRecord(
        phase="extractor",
        turn_index=turn,
        root_session_id=root_id,
        ts_ns=1_000_000_000 + turn,
        compose_kwargs={},
        payload={"graph": []},
        provider=None,
        output={"events": [], "edges": []},
        status="ok",
    )
    return StepResult(
        fired_extractor=True,
        fired_auditor=False,
        surfaced_reminder=None,
        extractor_record=rec,
        auditor_record=None,
    )


def _aud_step(turn: int, root_id: str, *, surface: bool = False) -> StepResult:
    rec = ReplayRecord(
        phase="auditor",
        turn_index=turn,
        root_session_id=root_id,
        ts_ns=1_000_000_000 + turn + 5_000,
        compose_kwargs={},
        payload={"graph": []},
        provider=None,
        output={"surface_reminder": surface, "reminder_text": "x" if surface else ""},
        status="ok",
    )
    return StepResult(
        fired_extractor=False,
        fired_auditor=True,
        surfaced_reminder=Reminder(text="x") if surface else None,
        extractor_record=None,
        auditor_record=rec,
    )


def test_write_chained_replay_filters_window_and_rebinds(tmp_path: Path) -> None:
    """Three-segment fixture. The writer must:

    1. Emit only records inside each segment's [turn_lo, turn_hi].
    2. Rebind every record's root_session_id to the final segment's id.
    3. Drop records that fall outside the window (verify by including
       out-of-window records that should be filtered out).
    """
    # Segment 0 (control): fork_turn_index=None, surfaced_reminder_turn=4.
    # Window: [0, 4]. We include an out-of-window record at turn 5 to
    # verify the upper-bound filter.
    seg0 = ChainSegment(
        segment_index=0,
        payload=_FakePayload(session_log_id="seg-0"),
        seeded_reminder=None,
        fork_turn_index=None,
        surfaced_reminder_turn=4,
        step_results=[
            _ext_step(0, "seg-0"),
            _ext_step(2, "seg-0"),
            _aud_step(4, "seg-0", surface=True),
            _ext_step(5, "seg-0"),  # outside window → must be filtered.
        ],
    )
    # Segment 1: fork_turn_index=4, surfaced_reminder_turn=9.
    # Window: [5, 9]. Include an in-window-but-pre-cut record at turn 3
    # which should be filtered.
    seg1 = ChainSegment(
        segment_index=1,
        payload=_FakePayload(session_log_id="seg-1"),
        seeded_reminder=Reminder(text="r1"),
        fork_turn_index=4,
        surfaced_reminder_turn=9,
        step_results=[
            _ext_step(3, "seg-1"),  # below turn_lo → must be filtered.
            _ext_step(6, "seg-1"),
            _aud_step(9, "seg-1", surface=True),
        ],
    )
    # Segment 2 (terminator): fork_turn_index=9, no surfaced reminder.
    # Window: [10, +inf). All records pass.
    seg2 = ChainSegment(
        segment_index=2,
        payload=_FakePayload(session_log_id="seg-2"),
        seeded_reminder=Reminder(text="r2"),
        fork_turn_index=9,
        surfaced_reminder_turn=None,
        step_results=[
            _ext_step(10, "seg-2"),
            _aud_step(12, "seg-2"),
            _ext_step(14, "seg-2"),
        ],
    )

    out_path = tmp_path / "chained.jsonl"
    returned = write_chained_replay([seg0, seg1, seg2], out_path=out_path)
    assert returned == out_path
    written = list(iter_records(out_path))

    # (2) Every record rebound to the final segment id.
    assert all(r.root_session_id == "seg-2" for r in written)

    # (1)+(3) Exactly the in-window records, in segment order.
    pairs = [(r.phase, int(r.turn_index)) for r in written]
    assert pairs == [
        ("extractor", 0),
        ("extractor", 2),
        ("auditor", 4),
        ("extractor", 6),
        ("auditor", 9),
        ("extractor", 10),
        ("auditor", 12),
        ("extractor", 14),
    ]

    # Sanity: the JSONL is parseable round-trip.
    raw_lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == len(written)
    for line in raw_lines:
        json.loads(line)






# ---------------------------------------------------------------------------
# Test 4: _fork_prefix bumps the cut past a paired ToolResultMessage.
#
# The auditor cadence (`turn_count % 5 == 0`) is role-agnostic, so the
# surfacing turn regularly lands on an assistant message that carries
# ToolCallBlocks. If the fork prefix ends at exactly that index, the
# branch session starts with a dangling tool_call — the next provider
# call rejects the request (most providers fail-stop on an unmatched
# assistant-tool_call). The bump is the load-bearing invariant that
# keeps chained-fork branches viable.


def _user(text: str) -> UserMessage:
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
    )


def _assistant_text(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
        stop_reason="end_turn",
    )


def _assistant_tool_call(call_id: str, name: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(type="tool_call", id=call_id, name=name, arguments={}),
        ],
        timestamp=0.0,
        stop_reason="tool_use",
    )


def _tool_result(call_id: str, text: str) -> ToolResultMessage:
    return ToolResultMessage(
        role="tool_result",
        content=[
            ToolResultBlock(
                type="tool_result",
                tool_call_id=call_id,
                content=[TextContent(type="text", text=text)],
                is_error=False,
            )
        ],
        timestamp=0.0,
    )


def test_fork_prefix_bumps_past_paired_tool_result() -> None:
    """When the surfacing turn is an assistant message with tool_calls
    and the next message is its matching tool_result, the prefix must
    include the tool_result. Otherwise the seeded-reminder branch
    starts with a dangling tool_call → provider rejects.
    """
    messages: list[AgentMessage] = [
        _user("kick off"),
        _assistant_text("ok"),
        _assistant_tool_call("c-1", "search"),  # index 2 = surfacing turn
        _tool_result("c-1", "found"),  # index 3 = paired result
        _assistant_text("done"),
    ]

    prefix = _fork_prefix(messages, turn_index=2)
    assert len(prefix) == 4, (
        f"expected the cut to bump past the paired tool_result; "
        f"got prefix={[type(m).__name__ for m in prefix]}"
    )
    # The paired tool_result must be the last element.
    assert isinstance(prefix[-1], ToolResultMessage)








# ---------------------------------------------------------------------------
# Test 5: chain header — single-file topology bundle
#
# These tests cover Option D's wire shape: the chained sidecar's first
# line carries the chain topology under ``__chain_header__`` so consumers
# can recover segment lineage from one file (no sibling .meta.json).
# Fail-stop position: if the header drifts from the records below it,
# downstream joins (case viewer, distill) will silently misalign segments
# to firings.


def _seg_for_header(
    idx: int,
    sid: str,
    *,
    msg_count: int,
    seeded_reminder_text: str | None,
    fork_turn_index: int | None,
    surfaced_reminder_turn: int | None,
) -> ChainSegment:
    payload = _FakePayload(
        session_log_id=sid,
        final_messages=[_user("x")] * msg_count,
    )
    seeded = (
        Reminder(text=seeded_reminder_text)
        if seeded_reminder_text is not None
        else None
    )
    return ChainSegment(
        segment_index=idx,
        payload=payload,
        seeded_reminder=seeded,
        fork_turn_index=fork_turn_index,
        surfaced_reminder_turn=surfaced_reminder_turn,
        step_results=[],
    )


def test_chain_header_first_line_and_records_intact(tmp_path: Path) -> None:
    """write_chained_replay writes the header at file head + records below.

    A reader using iter_records skips the header silently and yields the
    records in order. A reader using read_chain_header gets the dict
    back.
    """
    segs = [
        _seg_for_header(0, "ctl", msg_count=10, seeded_reminder_text=None,
                        fork_turn_index=None, surfaced_reminder_turn=4),
        _seg_for_header(1, "b1", msg_count=15, seeded_reminder_text="r1",
                        fork_turn_index=4, surfaced_reminder_turn=9),
    ]
    segs[0].step_results.append(
        StepResult(
            fired_extractor=True, fired_auditor=True, surfaced_reminder=None,
            extractor_record=_ext_step(turn=4, root_id="ctl").extractor_record,
            auditor_record=_aud_step(turn=4, root_id="ctl", surface=False).auditor_record,
        )
    )
    segs[1].step_results.append(
        StepResult(
            fired_extractor=True, fired_auditor=True, surfaced_reminder=None,
            extractor_record=_ext_step(turn=9, root_id="b1").extractor_record,
            auditor_record=_aud_step(turn=9, root_id="b1", surface=True).auditor_record,
        )
    )
    header = {
        "schema_version": 1,
        "audit_interval": 5,
        "extractor_interval": 5,
        "max_interventions": 3,
        "segments": [
            {"segment_index": 0, "session_log_id": "ctl", "is_control": True,
             "fork_turn_index": None, "seeded_reminder_text": None,
             "surfaced_reminder_turn": 4, "msg_count": 10},
            {"segment_index": 1, "session_log_id": "b1", "is_control": False,
             "fork_turn_index": 4, "seeded_reminder_text": "r1",
             "surfaced_reminder_turn": 9, "msg_count": 15},
        ],
    }
    out = tmp_path / "sample.chained.jsonl"
    write_chained_replay(segs, out_path=out, header=header)

    raw_lines = out.read_text(encoding="utf-8").splitlines()
    assert raw_lines, "sidecar should not be empty"
    head_obj = json.loads(raw_lines[0])
    assert CHAIN_HEADER_KEY in head_obj
    assert head_obj[CHAIN_HEADER_KEY] == header

    # read_chain_header pulls the same dict back
    assert read_chain_header(out) == header

    # iter_records skips the header and yields only ReplayRecord rows
    recs = list(iter_records(out))
    assert all(r.root_session_id == "b1" for r in recs), "rebind to final sid"
    assert sorted((r.phase, r.turn_index) for r in recs) == [
        ("auditor", 4), ("auditor", 9), ("extractor", 4), ("extractor", 9),
    ]






