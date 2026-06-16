"""Fork-tree counterfactual experiment driver.

A *fork tree* generalises the old linear chained-fork experiment into a
producer-consumer work queue. Each **surface point** (an auditor firing
with ``surface_reminder=True``) on a backbone trajectory is a
parallel-universe split:

* "don't intervene" — the backbone simply continues past the firing
  (free; already in the trajectory); and
* "intervene" — a *new* child rollout forked at that point with the
  surfaced reminder seeded into the continuation.

Consuming one :class:`ForkTask` produces a :class:`ForkNode` plus a child
``ForkTask`` per surface, so the queue feeds itself and drains when every
branch's auditor goes silent.

Degenerate cases fall out of the same engine:

* take ``surfaces[:1]`` per node ⇒ the old linear chain (greedy spine);
* ``max_depth=1`` ⇒ a "star" (control + one fork per first-level surface);
* all surfaces, unbounded ⇒ the full tree (the default).

The default policy is the full tree with soft guards
(``max_depth=8``, ``max_total_nodes=64``) against a misbehaving auditor;
trees are expected small because the auditor is usually silent.

Per-surface cumulative-state plumbing lives in
:func:`replay_pipeline_over_trajectory`: when
``stop_on_first_surface=False`` the result exposes, for each surfaced
firing, the reminder text + a deep snapshot of
:class:`CumulativeAuditState`. The engine threads that snapshot into the
forked child's ``seed_cumulative`` so a parallel branch starts from the
auditor state its parent had at the fork point.

See ``.claude/designs/harness-runner.md`` §3 (P4+) for the rationale.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResultMessage,
)
from loguru import logger

from llmharness.replay.record import ReplayRecord, write_record
from llmharness.state import CumulativeAuditState

from .offline import InMemorySink, StandaloneChildRunner
from .offline_driver import SurfaceFiring, replay_pipeline_over_trajectory
from .runner import AuditorSettings, ExtractorSettings

__all__ = [
    "FORK_TREE_HEADER_KEY",
    "ForkNode",
    "ForkTask",
    "ForkTreeExperiment",
    "SessionFactory",
    "SessionPayload",
    "Surface",
    "forktree_replay_path",
    "read_fork_tree_header",
    "run_fork_tree_experiment",
    "write_fork_tree_replay",
]

class SessionPayload(Protocol):
    """The seam between the experiment driver and the host session.

    A :class:`SessionFactory` returns one of these per backbone rollout.
    Anything that exposes ``session_log_id: str`` and
    ``final_messages: list[AgentMessage]`` qualifies — notably the rca
    eval driver's ``_SessionRun`` dataclass matches structurally so the
    factory can return it directly.
    """

    session_log_id: str
    final_messages: list[AgentMessage]

SessionFactory = Callable[..., Awaitable[SessionPayload]]
"""Coroutine: ``(*, initial_messages, seed_reminder_text) -> SessionPayload``.

The root (control) backbone is invoked with ``initial_messages=None`` and
``seed_reminder_text=None``; each forked child receives the parent
trajectory prefix up to the fork point and the surfaced reminder text.
"""

@dataclass(frozen=True)
class Surface:
    """One auditor firing that surfaced a reminder on a backbone.

    ``fork_message_index`` is the **message** index the auditor fired at
    (== ``auditor_record.turn_index``); a child forked here inherits the
    prefix ``backbone.final_messages[: fork_message_index + 1]`` (bumped
    by :func:`_fork_prefix` past a paired tool_result). ``reminder_text``
    is what gets seeded into the child's continuation.
    ``cumulative_snapshot`` is the independent deep copy of the auditor
    state as of this firing — the child's ``seed_cumulative``.
    """

    fork_message_index: int
    reminder_text: str
    cumulative_snapshot: CumulativeAuditState

@dataclass(frozen=True)
class ForkTask:
    """A unit of work in the fork-tree queue.

    Consuming a task produces exactly one :class:`ForkNode` (a backbone
    rollout + its surfaces) and zero-or-more child tasks (one per surface,
    subject to depth / total-node guards).
    """

    node_id: str
    parent_id: str | None
    prefix: tuple[AgentMessage, ...]
    """Inherited trajectory prefix up to the fork point. Empty for the
    root/control task."""
    seed_reminder: str | None
    """Reminder injected at the head of the continuation. ``None`` for the
    root."""
    seed_cumulative: CumulativeAuditState | None
    """Parent's audit-state snapshot at the fork point. ``None`` for the
    root (a fresh state is used)."""
    fork_turn: int
    """Fork-boundary message index + 1 — i.e. the message-prefix length at
    the fork point. ``1`` for the root. Drives the recorded
    ``fork_turn_index`` (``fork_turn - 1``) and the non-progressing floor.
    The child's auditing resumes one turn *past* the boundary
    (``start_turn = fork_turn + 1``) so the parent's fork-point firing —
    already carried by ``seed_cumulative`` — is not re-audited; see
    :func:`run_fork_tree_experiment`."""
    depth: int
    """``0`` for the root; ``parent.depth + 1`` for a child."""
    path: tuple[str, ...]
    """Intervention provenance: the reminder texts applied from the root
    down to (and including) this task's seed. Empty for the root."""

@dataclass
class ForkNode:
    """Product of consuming one :class:`ForkTask`.

    Carries the backbone rollout's identity, the reminder it was seeded
    with (if any), every surface on its trajectory, the per-step audit
    results, and the tree-position metadata needed to reconstruct the
    tree from ``parent_id`` links and to report provenance (``path``).
    """

    node_id: str
    parent_id: str | None
    backbone_session_id: str
    seeded_reminder: str | None
    fork_turn_index: int | None
    surfaces: list[Surface]
    step_results: list[dict[str, Any]]
    depth: int
    path: tuple[str, ...]
    payload: SessionPayload
    """The backbone rollout payload (session id + final messages). The rca
    consumer reaches through this for the submission / AgentResult side
    table keyed by ``session_log_id``."""

@dataclass(frozen=True)
class ForkTreeExperiment:
    """Outcome of one :func:`run_fork_tree_experiment` invocation.

    ``nodes`` is in consumption order; ``nodes[0]`` is always the
    root/control. The tree is recoverable from each node's ``parent_id``.
    """

    nodes: list[ForkNode]
    forktree_replay_path: Path | None
    header: dict[str, Any] = field(default_factory=dict)
    """Tree-topology header — the same dict written to the sidecar's first
    line (under :data:`FORK_TREE_HEADER_KEY`) and shared with callers that
    want to build ``AgentResult.metadata`` without re-deriving it from
    :attr:`nodes`. Schema: see :func:`_build_fork_tree_header`."""

    @property
    def root(self) -> ForkNode:
        return self.nodes[0]

def forktree_replay_path(cwd: str | Path, root_session_log_id: str) -> Path:
    """Canonical sidecar path for a fork-tree replay.

    Keyed off the *root/control* backbone's session log id (the node the
    rca eval reports as its primary submission), with a ``.chained.jsonl``
    suffix so the existing case-viewer discovery (via the root run's
    ``audit_replay_path`` metadata) keeps finding it.
    """
    return Path(cwd) / ".agentm" / "audit_replay" / f"{root_session_log_id}.chained.jsonl"

def _fork_prefix(parent_messages: list[AgentMessage], turn_index: int) -> list[AgentMessage]:
    """Return the parent-trajectory prefix that ends *cleanly* at ``turn_index``.

    The auditor's cadence (``turn_count % audit_interval == 0``) is
    role-agnostic, so ``turn_index`` regularly lands on an
    :class:`AssistantMessage` that carries :class:`ToolCallBlock`s. The
    naive cut ``messages[: turn_index + 1]`` would leave the branch
    session's initial messages ending with an unanswered tool_call —
    the next provider request then has a dangling assistant-tool_call
    with no matching ToolResultMessage, and most providers fail-stop
    on that shape. Bump the cut to include the paired
    :class:`ToolResultMessage` so the branch starts from a well-formed
    transcript.

    Bump rule: include ``parent_messages[turn_index + 1]`` iff
    ``parent_messages[turn_index]`` is an :class:`AssistantMessage`
    with at least one :class:`ToolCallBlock`, and the next message is
    a :class:`ToolResultMessage` whose result-blocks reference at
    least one of those ``tool_call`` ids. Adjacency alone is not
    enough — pairing is by id.
    """
    if turn_index < 0:
        return []
    cut = min(turn_index + 1, len(parent_messages))
    if cut >= len(parent_messages):
        return list(parent_messages[:cut])

    current = parent_messages[turn_index]
    following = parent_messages[cut]
    if isinstance(current, AssistantMessage) and isinstance(following, ToolResultMessage):
        assistant_call_ids = {
            block.id for block in current.content if isinstance(block, ToolCallBlock)
        }
        result_call_ids = {
            block.tool_call_id for block in following.content if hasattr(block, "tool_call_id")
        }
        if assistant_call_ids and assistant_call_ids & result_call_ids:
            cut += 1
    return list(parent_messages[:cut])

def _collect_surfaces(run_surfaces: list[SurfaceFiring]) -> list[Surface]:
    """Map the offline driver's per-firing surfaces onto :class:`Surface`.

    Replaces the linear driver's last-only ``_last_audited_turn``: the
    fork tree forks at *every* surface, not just the final one.
    """
    return [
        Surface(
            fork_message_index=int(firing.turn_index),
            reminder_text=firing.reminder_text,
            cumulative_snapshot=firing.cumulative_snapshot,
        )
        for firing in run_surfaces
    ]

def _rebind_record(record: ReplayRecord, *, session_id: str) -> ReplayRecord:
    """Return a copy of ``record`` re-keyed under a new session_id.

    The fork-tree sidecar is keyed off the *root* backbone session id but
    composed from records produced under many different session ids (one
    per node). Every entry's ``session_id`` is rewritten while everything
    else (timing, payload, output) stays verbatim.
    """
    return replace(record, session_id=session_id)

async def run_fork_tree_experiment(
    *,
    session_factory: SessionFactory,
    cwd: str,
    provider: tuple[str, dict[str, Any]] | None,
    extractor_settings: ExtractorSettings,
    auditor_settings: AuditorSettings,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    max_depth: int = 8,
    max_total_nodes: int = 64,
    max_surfaces_per_node: int | None = None,
    max_workers: int = 1,
    out_path: Path | None = None,
    sink: InMemorySink | None = None,
    child: StandaloneChildRunner | None = None,
    skip_extractor: bool = False,
    trigger_registry: Any | None = None,
    trace_id: str | None = None,
) -> ForkTreeExperiment:
    """Drive a fork-tree counterfactual experiment to completion.

    Producer-consumer algorithm:

    1. Seed the queue with the root task (``prefix=()``, ``seed=None``,
       ``fork_turn=1``, ``depth=0``).
    2. Drain the queue with ``max_workers`` concurrent workers (default
       ``1`` — a sequential drain; ``>1`` is a safe config flip because
       tasks share no mutable state). Each worker:

       * pops a task, builds its backbone via ``session_factory``;
       * replays the audit pipeline offline over the backbone with
         ``stop_on_first_surface=False`` so *every* surface is captured,
         seeded with the parent's cumulative snapshot + ``start_turn``
         floor;
       * emits a :class:`ForkNode`;
       * for each surface (subject to ``max_depth`` /
         ``max_total_nodes`` / ``max_surfaces_per_node``) enqueues a child
         :class:`ForkTask` forked at that surface.
    3. Terminate when the queue empties or ``max_total_nodes`` nodes have
       been emitted. Return a :class:`ForkTreeExperiment` holding every
       node (tree recoverable via ``parent_id``).

    Degenerate policies (documented; only the default is wired):
    ``max_surfaces_per_node=1`` reproduces the greedy linear spine;
    ``max_depth=1`` produces a star; the default (all surfaces, soft
    guards) is the full tree.

    ``sink`` / ``child`` are exposed for tests that inject stubs while
    still exercising the real :class:`HarnessRunner`.
    """
    queue: asyncio.Queue[ForkTask] = asyncio.Queue()
    nodes: list[ForkNode] = []
    # ``_node_counter`` and ``_emitted`` are mutated only by the worker
    # coroutines, which run on a single event loop. The cap check + counter
    # bump happen without an ``await`` between them, so even at
    # ``max_workers > 1`` no two workers can both pass the guard for the
    # same slot — the cooperative scheduler never preempts mid-check.
    state = {"emitted": 0, "counter": 0}

    root_task = ForkTask(
        node_id="n0",
        parent_id=None,
        prefix=(),
        seed_reminder=None,
        seed_cumulative=None,
        fork_turn=1,
        depth=0,
        path=(),
    )
    state["counter"] = 1
    queue.put_nowait(root_task)

    def _next_node_id() -> str:
        nid = f"n{state['counter']}"
        state["counter"] += 1
        return nid

    async def _consume(task: ForkTask) -> None:
        backbone = await session_factory(
            initial_messages=list(task.prefix) if task.prefix else None,
            seed_reminder_text=task.seed_reminder,
        )
        logger.info(f'fork_tree: node={task.node_id} parent={task.parent_id} depth={task.depth} session={backbone.session_log_id} msgs={len(backbone.final_messages)} seed={"None" if task.seed_reminder is None else "set"}')
        run = await replay_pipeline_over_trajectory(
            messages=backbone.final_messages,
            cwd=cwd,
            session_id=backbone.session_log_id,
            provider=provider,
            extractor_settings=extractor_settings,
            auditor_settings=auditor_settings,
            extractor_interval=extractor_interval,
            audit_interval=audit_interval,
            enable_auditor=True,
            stop_on_first_surface=False,
            sidecar_path=None,
            sink=sink,
            child=child,
            seed_cumulative=task.seed_cumulative,
            # Resume the child's auditing ONE turn past the fork boundary.
            # The parent's fork-point firing is already folded into
            # ``seed_cumulative``; re-auditing the boundary turn would burn
            # an auditor LLM call, write a duplicate verdict into the
            # child's ``recent_verdicts``, and surface a duplicate reminder
            # that the non-progressing floor below then discards. The root
            # (no boundary to skip) starts at ``fork_turn`` == 1.
            start_turn=(task.fork_turn if task.parent_id is None else task.fork_turn + 1),
            skip_extractor=skip_extractor,
            trigger_registry=trigger_registry,
            trace_id=trace_id,
        )
        surfaces = _collect_surfaces(run.surfaces)
        # Defence-in-depth: ignore any surface at or before the fork floor.
        # A re-seeded child replay can, in pathological auditor behaviour,
        # surface on the boundary turn it just resumed from; forking there
        # would re-fork at the same point and never make progress.
        floor = task.fork_turn - 1
        forward_surfaces = [s for s in surfaces if s.fork_message_index > floor]
        if len(forward_surfaces) != len(surfaces):
            logger.warning(f"fork_tree: node={task.node_id} dropped {len(surfaces) - len(forward_surfaces)} non-progressing surface(s) at/below fork floor={floor}")

        node = ForkNode(
            node_id=task.node_id,
            parent_id=task.parent_id,
            backbone_session_id=backbone.session_log_id,
            seeded_reminder=task.seed_reminder,
            fork_turn_index=(task.fork_turn - 1) if task.parent_id is not None else None,
            surfaces=forward_surfaces,
            step_results=list(run.all_step_results),
            depth=task.depth,
            path=task.path,
            payload=backbone,
        )
        nodes.append(node)

        if task.depth >= max_depth:
            return
        selected = (
            forward_surfaces
            if max_surfaces_per_node is None
            else forward_surfaces[:max_surfaces_per_node]
        )
        for surface in selected:
            if state["emitted"] >= max_total_nodes:
                logger.warning(f"fork_tree: max_total_nodes={max_total_nodes} reached; not enqueuing further children")
                break
            state["emitted"] += 1
            child_prefix = _fork_prefix(backbone.final_messages, surface.fork_message_index)
            queue.put_nowait(
                ForkTask(
                    node_id=_next_node_id(),
                    parent_id=task.node_id,
                    prefix=tuple(child_prefix),
                    seed_reminder=surface.reminder_text,
                    seed_cumulative=surface.cumulative_snapshot,
                    fork_turn=surface.fork_message_index + 1,
                    depth=task.depth + 1,
                    path=(*task.path, surface.reminder_text),
                )
            )

    async def _worker() -> None:
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _consume(task)
            finally:
                queue.task_done()

    # The root counts toward the node budget before the first pop.
    state["emitted"] = 1
    workers = [asyncio.create_task(_worker()) for _ in range(max(1, max_workers))]
    await asyncio.gather(*workers)

    # Stable consumption order is already preserved by ``nodes.append``;
    # the root (n0) is always first because it is the only seeded task.
    resolved_out: Path = (
        out_path
        if out_path is not None
        else forktree_replay_path(cwd, nodes[0].backbone_session_id)
    )
    header = _build_fork_tree_header(
        nodes=nodes,
        audit_interval=audit_interval,
        extractor_interval=extractor_interval,
        max_depth=max_depth,
        max_total_nodes=max_total_nodes,
    )
    sidecar = write_fork_tree_replay(nodes, out_path=resolved_out, header=header)
    return ForkTreeExperiment(nodes=nodes, forktree_replay_path=sidecar, header=header)

FORK_TREE_HEADER_KEY = "__fork_tree_header__"
"""JSON key marking the fork-tree replay sidecar's header line.

The header is the first line of a ``<root_sid>.chained.jsonl`` sidecar.
It carries the experiment topology (per-node ids, parent links, depth,
fork points, seeded reminders, intervention paths) so downstream
consumers can reconstruct the tree from the per-firing
:class:`ReplayRecord` stream without re-running the driver. See
:func:`_build_fork_tree_header` for the field layout and
:func:`read_fork_tree_header` for the matching reader."""

def _node_to_header_dict(node: ForkNode) -> dict[str, Any]:
    """JSON-safe summary of one node, as it appears in the sidecar header."""
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "fork_turn_index": node.fork_turn_index,
        "seeded_reminder": node.seeded_reminder,
        "is_control": node.parent_id is None,
        "path": list(node.path),
        "backbone_session_id": node.backbone_session_id,
        "surface_turns": [s.fork_message_index for s in node.surfaces],
        "msg_count": len(node.payload.final_messages),
    }

def _build_fork_tree_header(
    *,
    nodes: list[ForkNode],
    audit_interval: int,
    extractor_interval: int,
    max_depth: int,
    max_total_nodes: int,
) -> dict[str, Any]:
    """Build the tree-topology dict written as the sidecar header.

    Schema (v1):

    * ``schema_version``: int — bumps on breaking field changes.
    * ``audit_interval`` / ``extractor_interval``: int — cadence the
      runner was driven at.
    * ``max_depth`` / ``max_total_nodes``: int — the guards the experiment
      ran under.
    * ``nodes``: list — one entry per :class:`ForkNode`, per
      :func:`_node_to_header_dict`: ``node_id``, ``parent_id``, ``depth``,
      ``fork_turn_index``, ``seeded_reminder``, ``is_control``, ``path``,
      ``backbone_session_id``, ``surface_turns``, ``msg_count``.

    The header excludes the audit ``ReplayRecord`` stream itself — that
    lives in the remaining lines of the sidecar.
    """
    return {
        "schema_version": 1,
        "audit_interval": audit_interval,
        "extractor_interval": extractor_interval,
        "max_depth": max_depth,
        "max_total_nodes": max_total_nodes,
        "nodes": [_node_to_header_dict(n) for n in nodes],
    }

def read_fork_tree_header(path: Path) -> dict[str, Any] | None:
    """Return the tree-topology header from a fork-tree replay sidecar.

    Reads only the first line. Returns ``None`` when the file is missing,
    empty, the first line is not JSON, the JSON is not a dict, or the dict
    does not carry :data:`FORK_TREE_HEADER_KEY`.

    See :func:`_build_fork_tree_header` for the header's field layout.
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    if not first:
        return None
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    header = obj.get(FORK_TREE_HEADER_KEY)
    if not isinstance(header, dict):
        return None
    return header

def _node_window(node: ForkNode) -> tuple[int, int | None]:
    """Return the ``[turn_lo, turn_hi]`` window of records this node owns.

    A node's own contribution to the trajectory begins right after its
    fork point (turns at/below the fork were inherited from the parent and
    are recorded under the parent's window). The root owns from turn 0.

    Upper bound: a node owns every record it produced; children fork from
    surface points but those surfaces still *belong* to this node's
    backbone, so there is no upper cut. (Unlike the old linear chain, a
    fork-tree node is not "superseded" by a single successor — it may have
    several children, and its full trajectory remains its own.)
    """
    if node.parent_id is None or node.fork_turn_index is None:
        return 0, None
    return node.fork_turn_index + 1, None

def write_fork_tree_replay(
    nodes: list[ForkNode],
    *,
    out_path: Path,
    header: dict[str, Any] | None = None,
) -> Path:
    """Materialise a fork-tree replay sidecar.

    Layout:

    1. Optional header line: ``{"__fork_tree_header__": {...}}`` carrying
       the tree topology (see :func:`_build_fork_tree_header`). Skipped by
       :func:`llmharness.replay.record.iter_records` (it carries no
       ``phase`` key), so existing record consumers ignore it.
    2. One :class:`ReplayRecord` JSONL row per surviving firing, each
       carrying its owning node's id under ``extras['node_id']`` and the
       parent link under ``extras['parent_node_id']`` so the tree
       reconstructs from parent links.

    Records are filtered to each node's :func:`_node_window` and rebound
    to ``nodes[0].backbone_session_id`` (the root) so the case viewer
    joins them against the root trajectory regardless of which node they
    came from. Returns the output path; an existing file is truncated
    first so re-running overwrites cleanly.
    """
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root_sid = nodes[0].backbone_session_id

    if header is not None:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {FORK_TREE_HEADER_KEY: header},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")

    for node in nodes:
        turn_lo, turn_hi = _node_window(node)
        for step in node.step_results:
            for record in (step.get("extractor_record"), step.get("auditor_record")):
                if record is None:
                    continue
                if not isinstance(record, ReplayRecord):
                    continue
                t = int(record.turn_index)
                if t < turn_lo:
                    continue
                if turn_hi is not None and t > turn_hi:
                    continue
                tagged = replace(
                    record,
                    extras={
                        **record.extras,
                        "node_id": node.node_id,
                        "parent_node_id": node.parent_id,
                    },
                )
                write_record(out_path, _rebind_record(tagged, session_id=root_sid))

    return out_path
