"""Pass 2 — intent alignment edges.

Links step purposes to Constraints and Outcomes via two edge kinds:

    addresses   (LLM)    a tool_call step's purpose relates to a constraint
    fulfills    (LLM)    a tool_call step achieved its declared purpose
                         (judged against the tool_result)

Idempotent per run: existing edges of these kinds are replaced wholesale
before rebuilding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..ir.models import Edge, StepRole, stable_id
from ..oracle import SessionFactory, _ask_model, _index_by_id

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex
    from ..ir.models import Step

_INTENT_EDGE_KINDS = frozenset({"addresses", "fulfills"})

_MAX_ADDRESSES_PAIRS = 100
_MAX_FULFILLS_PAIRS = 60


@dataclass(frozen=True, slots=True)
class IntentAlignmentResult:
    edges: list[Edge] = field(default_factory=list)
    n_addresses_pairs: int = 0
    n_fulfills_pairs: int = 0
    n_judged: int = 0
    n_capped: int = 0

    def to_artifact(self) -> dict[str, Any]:
        return {
            "edges": [
                {
                    "id": e.id, "kind": e.kind, "src": e.src, "dst": e.dst,
                    "quote": e.quote, "evidence_position": e.evidence_position,
                }
                for e in self.edges
            ],
            "n_addresses_pairs": self.n_addresses_pairs,
            "n_fulfills_pairs": self.n_fulfills_pairs,
            "n_judged": self.n_judged,
            "n_capped": self.n_capped,
        }


def _extract_purpose(step: Step) -> str:
    """Read purpose from tool_call JSON args in step content."""
    content = step.content
    lines = content.split("\n", 1)
    if len(lines) < 2:
        return ""
    try:
        args = json.loads(lines[1])
        return str(args.get("purpose", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""


def _tool_call_steps(
    index: TrajectoryIndex,
    run_id: str,
) -> list[Step]:
    """All tool_call steps for the given run, sorted by index."""
    steps: list[Step] = []
    for (rid, _), step in index.steps.items():
        if run_id and rid != run_id:
            continue
        if step.tool_name is not None:
            steps.append(step)
    return sorted(steps, key=lambda s: s.index)


_ADDRESSES_INSTRUCTIONS = """\
An agent performed actions during a task. Each action has a declared purpose.
Below are (action, constraint) pairs. For each pair, judge whether the action's
purpose directly relates to satisfying the constraint. Be strict: a tangential
or incidental connection is not enough; the purpose must target the constraint's
requirement.

Return ONLY:
{"verdicts": [{"id": 0, "outcome": "yes|no|unclear"}]}
"""

_FULFILLS_INSTRUCTIONS = """\
An agent performed actions during a task. Each action has a declared purpose
and a tool result showing what actually happened. For each pair, judge whether
the action achieved its declared purpose based on the tool result. Be strict:
partial achievement or ambiguous output counts as "no".

Return ONLY:
{"verdicts": [{"id": 0, "outcome": "yes|no|unclear"}]}
"""


async def _judge_addresses(
    index: TrajectoryIndex,
    run_id: str,
    model: str | None,
    session_factory: SessionFactory,
) -> tuple[list[Edge], int, int, int]:
    """LLM-judged: which tool_call steps address which constraints."""
    tc_steps = _tool_call_steps(index, run_id)
    steps_with_purpose = [
        (s, _extract_purpose(s)) for s in tc_steps
    ]
    steps_with_purpose = [(s, p) for s, p in steps_with_purpose if p]

    constraints = list(index.constraints.values())
    if not steps_with_purpose or not constraints:
        return [], 0, 0, 0

    pairs: list[dict[str, Any]] = []
    pair_keys: list[tuple[str, str]] = []
    for step, purpose in steps_with_purpose:
        for constraint in constraints:
            pairs.append({
                "id": len(pairs),
                "action_purpose": purpose,
                "action_tool": step.tool_name or "",
                "constraint": constraint.description,
            })
            pair_keys.append((step.step_id, constraint.id))

    total_pairs = len(pairs)
    n_capped = 0
    if total_pairs > _MAX_ADDRESSES_PAIRS:
        n_capped = total_pairs - _MAX_ADDRESSES_PAIRS
        logger.warning(
            "intent_alignment addresses: {} pairs capped to {} (dropped {})",
            total_pairs, _MAX_ADDRESSES_PAIRS, n_capped,
        )
        pairs = pairs[:_MAX_ADDRESSES_PAIRS]
        pair_keys = pair_keys[:_MAX_ADDRESSES_PAIRS]

    payload = json.dumps({"pairs": pairs}, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        _ADDRESSES_INSTRUCTIONS, payload, model,
        session_factory=session_factory, purpose="intent_addresses",
    )

    edges: list[Edge] = []
    n_judged = 0
    if raw is not None:
        by_id = _index_by_id(raw)
        for idx, (step_id, constraint_id) in enumerate(pair_keys):
            verdict = by_id.get(idx)
            if not verdict:
                continue
            n_judged += 1
            outcome = str(verdict.get("outcome", ""))
            if outcome == "yes":
                step = index.steps.get((run_id, step_id))
                r_id = step.run_id if step else run_id
                edge = Edge(
                    id=stable_id("edge", r_id, "addresses", step_id, constraint_id),
                    kind="addresses",
                    run_id=r_id,
                    src=step_id,
                    dst=constraint_id,
                )
                edges.append(edge)

    return edges, len(pairs), n_judged, n_capped


async def _judge_fulfills(
    index: TrajectoryIndex,
    run_id: str,
    model: str | None,
    session_factory: SessionFactory,
) -> tuple[list[Edge], int, int, int]:
    """LLM-judged: did each tool_call step achieve its declared purpose."""
    steps_by_id: dict[str, Any] = {}
    for (rid, sid), step in index.steps.items():
        if not run_id or rid == run_id:
            steps_by_id[sid] = step

    sorted_steps = sorted(steps_by_id.values(), key=lambda s: s.index)
    step_by_index: dict[int, Any] = {s.index: s for s in sorted_steps}

    tc_steps = _tool_call_steps(index, run_id)
    steps_with_purpose = [
        (s, _extract_purpose(s)) for s in tc_steps
    ]
    steps_with_purpose = [(s, p) for s, p in steps_with_purpose if p]

    if not steps_with_purpose:
        return [], 0, 0, 0

    pairs: list[dict[str, Any]] = []
    pair_keys: list[tuple[str, str]] = []
    for step, purpose in steps_with_purpose:
        result_step = step_by_index.get(step.index + 1)
        if not result_step or result_step.role != StepRole.TOOL_RESULT:
            continue
        result_content = result_step.content
        if not result_content.strip():
            continue
        pairs.append({
            "id": len(pairs),
            "purpose": purpose,
            "tool": step.tool_name or "",
            "result_excerpt": result_content[:2000],
        })
        pair_keys.append((step.step_id, result_step.step_id))

    total_pairs = len(pairs)
    n_capped = 0
    if total_pairs > _MAX_FULFILLS_PAIRS:
        n_capped = total_pairs - _MAX_FULFILLS_PAIRS
        logger.warning(
            "intent_alignment fulfills: {} pairs capped to {} (dropped {})",
            total_pairs, _MAX_FULFILLS_PAIRS, n_capped,
        )
        pairs = pairs[:_MAX_FULFILLS_PAIRS]
        pair_keys = pair_keys[:_MAX_FULFILLS_PAIRS]

    payload = json.dumps({"pairs": pairs}, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        _FULFILLS_INSTRUCTIONS, payload, model,
        session_factory=session_factory, purpose="intent_fulfills",
    )

    edges: list[Edge] = []
    n_judged = 0
    if raw is not None:
        by_id = _index_by_id(raw)
        for idx, (action_step_id, result_step_id) in enumerate(pair_keys):
            verdict = by_id.get(idx)
            if not verdict:
                continue
            n_judged += 1
            outcome = str(verdict.get("outcome", ""))
            if outcome == "yes":
                action_step = steps_by_id.get(action_step_id)
                r_id = action_step.run_id if action_step else run_id
                edge = Edge(
                    id=stable_id("edge", r_id, "fulfills", action_step_id, result_step_id),
                    kind="fulfills",
                    run_id=r_id,
                    src=action_step_id,
                    dst=result_step_id,
                )
                edges.append(edge)

    return edges, len(pairs), n_judged, n_capped


async def build_intent_alignment_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
) -> IntentAlignmentResult:
    """Build ``addresses``/``fulfills`` edges (intent alignment).

    Idempotent per run: existing edges of these kinds for this run are
    replaced wholesale.
    """
    stale = [
        eid for eid, e in index.edges.items()
        if e.kind in _INTENT_EDGE_KINDS and (not run_id or e.run_id == run_id)
    ]
    for eid in stale:
        del index.edges[eid]

    all_edges: list[Edge] = []
    n_addresses_pairs = 0
    n_fulfills_pairs = 0
    n_judged = 0
    n_capped = 0

    if session_factory is not None:
        addr_edges, addr_pairs, addr_judged, addr_capped = await _judge_addresses(
            index, run_id, model, session_factory,
        )
        all_edges.extend(addr_edges)
        n_addresses_pairs = addr_pairs
        n_judged += addr_judged
        n_capped += addr_capped

        ful_edges, ful_pairs, ful_judged, ful_capped = await _judge_fulfills(
            index, run_id, model, session_factory,
        )
        all_edges.extend(ful_edges)
        n_fulfills_pairs = ful_pairs
        n_judged += ful_judged
        n_capped += ful_capped
    else:
        logger.info("intent_alignment: no session_factory, skipping LLM passes (addresses/fulfills)")

    for edge in all_edges:
        index.edges[edge.id] = edge

    result = IntentAlignmentResult(
        edges=all_edges,
        n_addresses_pairs=n_addresses_pairs,
        n_fulfills_pairs=n_fulfills_pairs,
        n_judged=n_judged,
        n_capped=n_capped,
    )

    logger.info(
        "intent_alignment: {} addresses edges (from {} pairs), "
        "{} fulfills edges (from {} pairs), {} judged, {} capped",
        sum(1 for e in all_edges if e.kind == "addresses"),
        n_addresses_pairs,
        sum(1 for e in all_edges if e.kind == "fulfills"),
        n_fulfills_pairs,
        n_judged,
        n_capped,
    )

    return result
