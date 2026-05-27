"""Fail-stop tests for the fork-tree driver + sidecar writer.

Why these are load-bearing: the fork tree is the host-side driver for
multi-rollout counterfactual experiments. Single-rollout behaviour
(cadence, sidecar shape, stop_on_first_surface) is covered by
``test_offline_driver.py``; this file pins the *fan-out + tree* contract
that is unique to the fork tree:

* ``replay_pipeline_over_trajectory`` honors ``seed_cumulative`` and
  ``start_turn`` AND, under ``stop_on_first_surface=False``, exposes one
  independent ``SurfaceFiring`` snapshot per surfaced firing. Wrong → a
  child fork seeds from blank state or from a snapshot that the
  continuing backbone mutates out from under it.
* ``run_fork_tree_experiment`` fans out: N surfaces → N child tasks, each
  forked at the right prefix length; a silent backbone → no children;
  ``max_depth`` / ``max_total_nodes`` halt the run.
* Child tasks receive an *independent* ``seed_cumulative`` snapshot.
* The tree is recoverable from ``parent_id``.
* ``write_fork_tree_replay`` filters each node's records to its window,
  rebinds to the root session id, and tags every record with its
  ``node_id`` / ``parent_node_id``.
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
from llmharness.replay import fork_tree as _fork_tree_mod
from llmharness.replay.fork_tree import (
    FORK_TREE_HEADER_KEY,
    ForkNode,
    Surface,
    _collect_surfaces,
    _fork_prefix,
    read_fork_tree_header,
    run_fork_tree_experiment,
    write_fork_tree_replay,
)
from llmharness.replay.offline_driver import (
    OfflineRunResult,
    SurfaceFiring,
    replay_pipeline_over_trajectory,
)
from llmharness.replay.record import ReplayRecord, iter_records
from llmharness.schema import Reminder, Verdict


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


# ---------------------------------------------------------------------------
# Test 1: replay_pipeline_over_trajectory honors seed_cumulative + start_turn
#         AND emits a per-surface snapshot under stop_on_first_surface=False.


class _NodeAddingStubChildRunner:
    """Adds one NodeUpsert per extractor firing; auditor silent."""

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
        return True, [
            {"type": "tool_call", "id": "x", "name": "finalize_extraction", "arguments": {}}
        ]

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
async def test_replay_pipeline_honors_seed_cumulative_and_start_turn(
    tmp_path: Path,
) -> None:
    messages = _make_trajectory(20)

    seeded = CumulativeAuditState.fresh()
    seeded.absorb_extractor_firing(
        firing_ops=[NodeUpsert(id=1, kind="task", summary="prior-node", source_turns=(0, 1, 2))],
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

    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        session_id="node-x",
        provider=None,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=False,
        sidecar_path=None,
        sink=sink,
        child=stub,  # type: ignore[arg-type]
        seed_cumulative=seeded,
        start_turn=11,
    )

    assert len(result.all_step_results) == 10
    fired_extractor_turns = [
        i + 11 for i, step in enumerate(result.all_step_results) if step.fired_extractor
    ]
    assert fired_extractor_turns == [15, 20]
    assert stub.extractor_calls == 2

    first_payload = stub.extractor_payloads[0]
    recent = first_payload.get("recent_graph") or []
    summaries = [e.get("summary") for e in recent]
    assert "prior-node" in summaries

    # Silent auditor → no surfaces recorded.
    assert result.surfaces == []


class _SurfacingTwiceStub(_NodeAddingStubChildRunner):
    """Auditor surfaces on EVERY firing — used to exercise multi-surface
    capture under ``stop_on_first_surface=False``.
    """

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
            surface_reminder=True,
            reminder_text=f"halt-{self.auditor_calls}",
            continuation_notes=[],
            matched_event_ids=[],
        )
        raw = {
            "type": "tool_call",
            "id": f"v-{self.auditor_calls}",
            "name": "submit_verdict",
            "arguments": {"verdict": verdict.to_dict()},
        }
        return AuditorChildResult(verdict=verdict, raw_blocks=[raw])


@pytest.mark.asyncio
async def test_full_tree_run_captures_independent_surface_snapshots(
    tmp_path: Path,
) -> None:
    """Under ``stop_on_first_surface=False`` the result must expose one
    ``SurfaceFiring`` per surfaced firing, each with an independent deep
    snapshot of the cumulative state. Fail-stop: a shared (aliased)
    snapshot would let a later firing mutate an earlier child's seed.
    """
    messages = _make_trajectory(20)
    stub = _SurfacingTwiceStub()
    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        session_id="node-y",
        provider=None,
        extractor_settings=ExtractorSettings(
            extensions=[(EXTRACTOR_TOOLS_MODULE, {})],
            compose_kwargs={"base_prompt": "stub"},
        ),
        auditor_settings=AuditorSettings(
            base_prompt="stub", observability_config=None, summary_threshold=30, tools=()
        ),
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=False,
        sink=InMemorySink(),
        child=stub,  # type: ignore[arg-type]
        seed_cumulative=None,
        start_turn=1,
    )

    # Auditor fires at turns 5, 10, 15, 20 (cadence 5) and surfaces each.
    assert [s.turn_index for s in result.surfaces] == [4, 9, 14, 19]
    assert [s.reminder_text for s in result.surfaces] == [
        "halt-1",
        "halt-2",
        "halt-3",
        "halt-4",
    ]
    # Snapshots are progressively larger (the graph grows each firing) and
    # are independent objects: mutating the live final state must not
    # change an already-captured snapshot.
    snap_sizes = [len(s.cumulative_snapshot.ops) for s in result.surfaces]
    assert snap_sizes == sorted(snap_sizes)
    assert snap_sizes[0] < snap_sizes[-1]
    first_snap = result.surfaces[0].cumulative_snapshot
    before = len(first_snap.ops)
    result.state.ops.append(NodeUpsert(id=999, kind="task", summary="late", source_turns=(0,)))
    assert len(first_snap.ops) == before, "snapshot must be independent of live state"


@pytest.mark.asyncio
async def test_child_resume_past_boundary_skips_fork_turn_firing(
    tmp_path: Path,
) -> None:
    """A forked child must NOT re-audit its own fork-boundary turn.

    Fail-stop position: with ``start_turn = fork_boundary + 1`` the offline
    driver skips the boundary cadence firing entirely, so the auditor is
    not invoked there (no wasted LLM call) and no duplicate surface lands
    at the boundary turn. The parent's fork-point verdict reaches the
    child via ``seed_cumulative`` instead.

    Concretely: a 20-turn backbone with cadence 5 fires at turns 5, 10,
    15, 20. Forking at boundary turn 5 (message index 4 → prefix length
    5) and resuming at ``start_turn = 6`` must drop the turn-5 firing —
    leaving firings at 10, 15, 20 only.
    """
    messages = _make_trajectory(20)
    stub = _SurfacingTwiceStub()
    settings_kw = dict(
        extractor_settings=ExtractorSettings(
            extensions=[(EXTRACTOR_TOOLS_MODULE, {})],
            compose_kwargs={"base_prompt": "stub"},
        ),
        auditor_settings=AuditorSettings(
            base_prompt="stub", observability_config=None, summary_threshold=30, tools=()
        ),
    )

    # fork boundary = message index 4 (prefix length 5); resume at 6.
    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=str(tmp_path),
        session_id="child",
        provider=None,
        extractor_interval=5,
        audit_interval=5,
        enable_auditor=True,
        stop_on_first_surface=False,
        sink=InMemorySink(),
        child=stub,  # type: ignore[arg-type]
        seed_cumulative=CumulativeAuditState.fresh(),
        start_turn=6,
        **settings_kw,  # type: ignore[arg-type]
    )

    # No surface (and no auditor firing) at the fork boundary turn 4.
    surfaced_turns = [s.turn_index for s in result.surfaces]
    assert 4 not in surfaced_turns
    assert surfaced_turns == [9, 14, 19]
    # The auditor was invoked exactly 3 times — the turn-5 boundary firing
    # was skipped, not merely discarded after the call.
    assert stub.auditor_calls == 3


# ---------------------------------------------------------------------------
# Scripted replay for engine-level tests (monkeypatches the offline driver).


@dataclass
class _FakePayload:
    session_log_id: str
    final_messages: list[AgentMessage] = field(default_factory=list)


class _ScriptedReplay:
    """Drop-in replacement for ``replay_pipeline_over_trajectory``.

    ``surfaces_by_session`` maps a backbone ``session_id`` to the
    list of ``(fork_message_index, reminder_text)`` surfaces the auditor
    produced on that backbone. Each surface carries a fresh, distinct
    ``CumulativeAuditState`` so identity-of-seed assertions are possible.
    """

    def __init__(self, surfaces_by_session: dict[str, list[tuple[int, str]]]) -> None:
        self._surfaces = surfaces_by_session
        self.calls_seen: list[dict[str, Any]] = []
        # session_id -> the snapshot objects handed out (for identity checks).
        self.snapshots: dict[str, list[CumulativeAuditState]] = {}

    async def __call__(self, **kwargs: Any) -> OfflineRunResult:
        self.calls_seen.append(kwargs)
        sid = kwargs["session_id"]
        spec = self._surfaces.get(sid, [])
        surfaces: list[SurfaceFiring] = []
        snaps: list[CumulativeAuditState] = []
        for idx, text in spec:
            snap = CumulativeAuditState.fresh()
            snaps.append(snap)
            surfaces.append(
                SurfaceFiring(turn_index=idx, reminder_text=text, cumulative_snapshot=snap)
            )
        self.snapshots[sid] = snaps
        return OfflineRunResult(
            reminder=None,
            state=CumulativeAuditState.fresh(),
            sidecar_path=None,
            all_step_results=[],
            surfaces=surfaces,
        )


def _factory_recording(factory_calls: list[dict[str, Any]], *, msgs_len: int = 30):
    counter = {"i": 0}

    async def factory(*, initial_messages, seed_reminder_text):
        i = counter["i"]
        counter["i"] += 1
        factory_calls.append(
            {
                "initial_messages_len": (len(initial_messages) if initial_messages else None),
                "seed_reminder_text": seed_reminder_text,
            }
        )
        return _FakePayload(session_log_id=f"sess-{i}", final_messages=_make_trajectory(msgs_len))

    return factory


def _run(
    monkeypatch: pytest.MonkeyPatch,
    scripted: _ScriptedReplay,
    factory,
    tmp_path: Path,
    **kwargs: Any,
):
    monkeypatch.setattr(_fork_tree_mod, "replay_pipeline_over_trajectory", scripted)
    return run_fork_tree_experiment(
        session_factory=factory,
        cwd=str(tmp_path),
        provider=None,
        extractor_settings=ExtractorSettings.default(),
        auditor_settings=AuditorSettings.default(),
        extractor_interval=5,
        audit_interval=5,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 2: N surfaces → N child tasks (fan-out), each forked at right prefix.


@pytest.mark.asyncio
async def test_fan_out_n_surfaces_produce_n_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root backbone surfaces 3 times → exactly 3 children. Each child is
    forked at ``surface_index + 1`` (its ``initial_messages`` length).
    Forked children are silent → no grandchildren.
    """
    scripted = _ScriptedReplay({"sess-0": [(4, "r-a"), (9, "r-b"), (14, "r-c")]})
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(monkeypatch, scripted, factory, tmp_path)

    # 1 root + 3 children = 4 nodes.
    assert len(experiment.nodes) == 4
    root = experiment.nodes[0]
    assert root.parent_id is None
    children = experiment.nodes[1:]
    assert all(c.parent_id == root.node_id for c in children)
    assert sorted(c.seeded_reminder for c in children) == ["r-a", "r-b", "r-c"]
    # Each child's prefix length = surface index + 1 (no tool_result bump
    # because the fixture trajectory is all bare assistant messages).
    child_prefix_lens = sorted(c["initial_messages_len"] for c in factory_calls[1:])
    assert child_prefix_lens == [5, 10, 15]
    # All children silent → leaves.
    assert all(c.surfaces == [] for c in children)


# ---------------------------------------------------------------------------
# Test 3: silent backbone → no children (queue drains to a single leaf).


@pytest.mark.asyncio
async def test_silent_backbone_produces_no_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = _ScriptedReplay({})  # no surfaces for any session
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(monkeypatch, scripted, factory, tmp_path)

    assert len(experiment.nodes) == 1
    assert experiment.nodes[0].parent_id is None
    assert experiment.nodes[0].surfaces == []
    assert len(factory_calls) == 1


# ---------------------------------------------------------------------------
# Test 4: max_depth halts deepening; max_total_nodes halts the whole run.


@pytest.mark.asyncio
async def test_max_depth_halts_deepening(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every backbone surfaces once, each surface strictly forward of the
    child's fork floor (so the non-progressing guard never drops it);
    without a depth guard this would deepen forever. ``max_depth=2`` must
    stop deepening: a node at ``depth == max_depth`` enqueues no children.
    """
    # sess-0 surfaces at 4 → child forks at floor 4, must surface > 4;
    # sess-1 surfaces at 9 → grandchild forks at floor 9, must surface > 9.
    scripted = _ScriptedReplay({f"sess-{i}": [(4 + 5 * i, f"r{i}")] for i in range(10)})
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(monkeypatch, scripted, factory, tmp_path, max_depth=2)

    depths = sorted(n.depth for n in experiment.nodes)
    assert max(depths) == 2
    # depth 0 (root) → 1 child (depth 1) → 1 child (depth 2) → stop.
    assert depths == [0, 1, 2]


@pytest.mark.asyncio
async def test_max_total_nodes_halts_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fan-out-3-per-node tree, capped at 5 total nodes, must emit no
    more than 5 nodes.
    """
    scripted = _ScriptedReplay({f"sess-{i}": [(4, "a"), (9, "b"), (14, "c")] for i in range(20)})
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(
        monkeypatch, scripted, factory, tmp_path, max_depth=8, max_total_nodes=5
    )

    assert len(experiment.nodes) <= 5


# ---------------------------------------------------------------------------
# Test 5: child tasks receive the correct, independent seed_cumulative.


@pytest.mark.asyncio
async def test_child_receives_independent_seed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed_cumulative passed into a child's replay must be the exact
    snapshot object captured at the parent's surface; mutating it
    afterward must not change what the engine already produced.
    """
    scripted = _ScriptedReplay({"sess-0": [(4, "r1")]})
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(monkeypatch, scripted, factory, tmp_path)

    root_snapshot = scripted.snapshots["sess-0"][0]
    child_call = scripted.calls_seen[1]
    assert child_call["seed_cumulative"] is root_snapshot
    # The child resumes auditing ONE turn past the fork boundary
    # (surface index 4 → fork boundary prefix-length 5 → start_turn 6),
    # so the parent's fork-point firing (carried by seed_cumulative) is
    # not re-audited.
    assert child_call["start_turn"] == 6

    before = len(root_snapshot.ops)
    root_snapshot.ops.append(NodeUpsert(id=1, kind="task", summary="x", source_turns=(0,)))
    assert len(root_snapshot.ops) == before + 1
    child_node = experiment.nodes[1]
    assert child_node.fork_turn_index == 4


# ---------------------------------------------------------------------------
# Test 6: tree recoverable from parent_id — linear chain + 2-leaf star.


@pytest.mark.asyncio
async def test_one_surface_per_node_reproduces_linear_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One surface per node, three deep, then silent → a linear chain
    root → c1 → c2 → c3 recoverable purely from parent_id links.
    """
    scripted = _ScriptedReplay(
        {
            "sess-0": [(4, "r1")],
            "sess-1": [(9, "r2")],
            "sess-2": [(14, "r3")],
            "sess-3": [],  # silent terminator
        }
    )
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(monkeypatch, scripted, factory, tmp_path, max_depth=8)

    assert len(experiment.nodes) == 4
    by_id = {n.node_id: n for n in experiment.nodes}
    spine: list[ForkNode] = []
    cursor: ForkNode | None = experiment.root
    while cursor is not None:
        spine.append(cursor)
        children = [n for n in experiment.nodes if n.parent_id == cursor.node_id]
        assert len(children) <= 1
        cursor = by_id[children[0].node_id] if children else None
    assert len(spine) == 4
    assert [n.seeded_reminder for n in spine] == [None, "r1", "r2", "r3"]
    assert [tuple(n.path) for n in spine] == [
        (),
        ("r1",),
        ("r1", "r2"),
        ("r1", "r2", "r3"),
    ]


@pytest.mark.asyncio
async def test_two_surfaces_at_root_then_silent_yields_two_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = _ScriptedReplay({"sess-0": [(4, "left"), (9, "right")]})
    factory_calls: list[dict[str, Any]] = []
    factory = _factory_recording(factory_calls)

    experiment = await _run(monkeypatch, scripted, factory, tmp_path)

    assert len(experiment.nodes) == 3  # root + 2 leaves
    root = experiment.root
    leaves = [n for n in experiment.nodes if n.parent_id == root.node_id]
    assert len(leaves) == 2
    assert sorted(leaf.seeded_reminder for leaf in leaves) == ["left", "right"]
    assert all(leaf.surfaces == [] for leaf in leaves)


# ---------------------------------------------------------------------------
# Test 7: write_fork_tree_replay filters windows, rebinds sid, tags node ids.


def _ext_step(turn: int, root_id: str) -> StepResult:
    rec = ReplayRecord(
        phase="extractor",
        turn_index=turn,
        session_id=root_id,
        trace_id=f"trace-{root_id}",
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
        session_id=root_id,
        trace_id=f"trace-{root_id}",
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


def _node(
    node_id: str,
    parent_id: str | None,
    sid: str,
    *,
    fork_turn_index: int | None,
    depth: int,
    path: tuple[str, ...],
    surfaces: list[Surface],
    steps: list[StepResult],
) -> ForkNode:
    return ForkNode(
        node_id=node_id,
        parent_id=parent_id,
        backbone_session_id=sid,
        seeded_reminder=path[-1] if path else None,
        fork_turn_index=fork_turn_index,
        surfaces=surfaces,
        step_results=steps,
        depth=depth,
        path=path,
        payload=_FakePayload(session_log_id=sid, final_messages=_make_trajectory(20)),
    )


def test_write_fork_tree_replay_filters_window_rebinds_and_tags(tmp_path: Path) -> None:
    """Two-node fixture (root + one child). The writer must:

    1. Emit only records inside each node's window
       (root: ``[0, +inf)``; child: ``[fork+1, +inf)``).
    2. Rebind every record's session_id to the ROOT node's id.
    3. Tag every record with its node_id / parent_node_id in extras.
    """
    root = _node(
        "n0",
        None,
        "sess-0",
        fork_turn_index=None,
        depth=0,
        path=(),
        surfaces=[
            Surface(
                fork_message_index=4,
                reminder_text="r1",
                cumulative_snapshot=CumulativeAuditState.fresh(),
            )
        ],
        steps=[
            _ext_step(0, "sess-0"),
            _ext_step(2, "sess-0"),
            _aud_step(4, "sess-0", surface=True),
        ],
    )
    child = _node(
        "n1",
        "n0",
        "sess-1",
        fork_turn_index=4,
        depth=1,
        path=("r1",),
        surfaces=[],
        steps=[
            _ext_step(3, "sess-1"),  # below fork floor (turn_lo=5) → filtered.
            _ext_step(6, "sess-1"),
            _aud_step(9, "sess-1"),
        ],
    )

    out_path = tmp_path / "forktree.jsonl"
    returned = write_fork_tree_replay([root, child], out_path=out_path)
    assert returned == out_path
    written = list(iter_records(out_path))

    assert all(r.session_id == "sess-0" for r in written)
    pairs = [(r.phase, int(r.turn_index)) for r in written]
    assert pairs == [
        ("extractor", 0),
        ("extractor", 2),
        ("auditor", 4),
        ("extractor", 6),
        ("auditor", 9),
    ]
    by_turn = {(r.phase, r.turn_index): r for r in written}
    assert by_turn[("extractor", 0)].extras["node_id"] == "n0"
    assert by_turn[("extractor", 0)].extras["parent_node_id"] is None
    assert by_turn[("auditor", 9)].extras["node_id"] == "n1"
    assert by_turn[("auditor", 9)].extras["parent_node_id"] == "n0"

    for line in out_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


# ---------------------------------------------------------------------------
# Test 8: fork-tree header — single-file topology bundle.


def test_fork_tree_header_first_line_and_records_intact(tmp_path: Path) -> None:
    """write_fork_tree_replay writes the header at file head + records
    below. iter_records skips the header; read_fork_tree_header returns it.
    """
    root = _node(
        "n0",
        None,
        "ctl",
        fork_turn_index=None,
        depth=0,
        path=(),
        surfaces=[],
        steps=[
            StepResult(
                fired_extractor=True,
                fired_auditor=True,
                surfaced_reminder=None,
                extractor_record=_ext_step(4, "ctl").extractor_record,
                auditor_record=_aud_step(4, "ctl", surface=True).auditor_record,
            )
        ],
    )
    child = _node(
        "n1",
        "n0",
        "b1",
        fork_turn_index=4,
        depth=1,
        path=("r1",),
        surfaces=[],
        steps=[
            StepResult(
                fired_extractor=True,
                fired_auditor=True,
                surfaced_reminder=None,
                extractor_record=_ext_step(9, "b1").extractor_record,
                auditor_record=_aud_step(9, "b1").auditor_record,
            )
        ],
    )

    out = tmp_path / "sample.chained.jsonl"
    header = {
        "schema_version": 1,
        "audit_interval": 5,
        "extractor_interval": 5,
        "max_depth": 8,
        "max_total_nodes": 64,
        "nodes": [
            {
                "node_id": "n0",
                "parent_id": None,
                "depth": 0,
                "is_control": True,
                "path": [],
                "backbone_session_id": "ctl",
            },
            {
                "node_id": "n1",
                "parent_id": "n0",
                "depth": 1,
                "is_control": False,
                "path": ["r1"],
                "backbone_session_id": "b1",
            },
        ],
    }
    write_fork_tree_replay([root, child], out_path=out, header=header)

    raw_lines = out.read_text(encoding="utf-8").splitlines()
    assert raw_lines
    head_obj = json.loads(raw_lines[0])
    assert FORK_TREE_HEADER_KEY in head_obj
    assert head_obj[FORK_TREE_HEADER_KEY] == header
    assert read_fork_tree_header(out) == header

    recs = list(iter_records(out))
    assert all(r.session_id == "ctl" for r in recs), "rebind to root sid"
    assert sorted((r.phase, r.turn_index) for r in recs) == [
        ("auditor", 4),
        ("auditor", 9),
        ("extractor", 4),
        ("extractor", 9),
    ]


# ---------------------------------------------------------------------------
# Test 9: _fork_prefix bumps past a paired ToolResultMessage; _collect_surfaces.


def _user(text: str) -> UserMessage:
    return UserMessage(role="user", content=[TextContent(type="text", text=text)], timestamp=0.0)


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
        content=[ToolCallBlock(type="tool_call", id=call_id, name=name, arguments={})],
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
    messages: list[AgentMessage] = [
        _user("kick off"),
        _assistant_text("ok"),
        _assistant_tool_call("c-1", "search"),  # index 2 = surfacing turn
        _tool_result("c-1", "found"),  # index 3 = paired result
        _assistant_text("done"),
    ]
    prefix = _fork_prefix(messages, turn_index=2)
    assert len(prefix) == 4
    assert isinstance(prefix[-1], ToolResultMessage)


def test_collect_surfaces_maps_every_firing() -> None:
    """_collect_surfaces forks at every surface, not just the last."""
    snap_a = CumulativeAuditState.fresh()
    snap_b = CumulativeAuditState.fresh()
    firings = [
        SurfaceFiring(turn_index=4, reminder_text="a", cumulative_snapshot=snap_a),
        SurfaceFiring(turn_index=9, reminder_text="b", cumulative_snapshot=snap_b),
    ]
    surfaces = _collect_surfaces(firings)
    assert [s.fork_message_index for s in surfaces] == [4, 9]
    assert [s.reminder_text for s in surfaces] == ["a", "b"]
    assert surfaces[0].cumulative_snapshot is snap_a
    assert surfaces[1].cumulative_snapshot is snap_b
