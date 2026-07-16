"""Pass 2 — intent alignment edges.

Links step purposes to Constraints and Outcomes via two edge kinds:

    addresses   (LLM)    a tool_call step's purpose relates to a constraint
    fulfills    (LLM)    a tool_call step achieved its declared purpose
                         (judged against the tool_result)

Both judgments are made in a single LLM call with two sections.
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
    """Read purpose from tool_call JSON args in step content.

    Step content for a tool_call is ``[tool_call: name]\\n<json_args>``
    (produced by ``message_parts``). The JSON args line is always a single
    ``json.dumps`` without indent, so splitting on the first newline is
    safe. If the format doesn't match, returns empty (never crashes).
    """
    content = step.content
    json_start = content.find("\n")
    if json_start < 0:
        return ""
    try:
        args = json.loads(content[json_start + 1:])
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


async def build_intent_alignment_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
) -> IntentAlignmentResult:
    """Build ``addresses``/``fulfills`` edges in a single LLM call.

    Idempotent per run: existing edges of these kinds for this run are
    replaced wholesale.
    """
    stale = [
        eid for eid, e in index.edges.items()
        if e.kind in _INTENT_EDGE_KINDS and (not run_id or e.run_id == run_id)
    ]
    for eid in stale:
        del index.edges[eid]

    if session_factory is None:
        logger.info("intent_alignment: no session_factory, skipping")
        return IntentAlignmentResult()

    tc_steps = _tool_call_steps(index, run_id)
    steps_with_purpose = [(s, _extract_purpose(s)) for s in tc_steps]
    steps_with_purpose = [(s, p) for s, p in steps_with_purpose if p]

    if not steps_with_purpose:
        return IntentAlignmentResult()

    constraints = list(index.constraints.values())

    # --- build addresses pairs ---
    addr_pairs: list[dict[str, Any]] = []
    addr_keys: list[tuple[str, str]] = []
    n_addr_capped = 0
    if constraints:
        n_constraints = len(constraints)
        kept_steps = steps_with_purpose
        if len(steps_with_purpose) * n_constraints > _MAX_ADDRESSES_PAIRS:
            max_steps = max(1, _MAX_ADDRESSES_PAIRS // n_constraints)
            n_addr_capped = (len(steps_with_purpose) - max_steps) * n_constraints
            stride = len(steps_with_purpose) / max_steps
            kept_steps = [steps_with_purpose[int(i * stride)] for i in range(max_steps)]
        for step, purpose in kept_steps:
            for constraint in constraints:
                addr_pairs.append({
                    "id": len(addr_pairs),
                    "action_purpose": purpose,
                    "action_tool": step.tool_name or "",
                    "constraint": constraint.description,
                })
                addr_keys.append((step.step_id, constraint.id))

    # --- build fulfills pairs ---
    result_by_call_id: dict[str, Any] = {}
    for (rid, _sid), step in index.steps.items():
        if run_id and rid != run_id:
            continue
        if step.role == StepRole.TOOL_RESULT and step.call_id:
            result_by_call_id[step.call_id] = step

    ful_pairs: list[dict[str, Any]] = []
    ful_keys: list[tuple[str, str]] = []
    n_ful_capped = 0
    for step, purpose in steps_with_purpose:
        result_step = result_by_call_id.get(step.call_id or "")
        if not result_step:
            continue
        result_content = result_step.content
        if not result_content.strip():
            continue
        ful_pairs.append({
            "id": len(ful_pairs),
            "purpose": purpose,
            "tool": step.tool_name or "",
            "result_excerpt": result_content[:2000],
        })
        ful_keys.append((step.step_id, result_step.step_id))
    if len(ful_pairs) > _MAX_FULFILLS_PAIRS:
        n_ful_capped = len(ful_pairs) - _MAX_FULFILLS_PAIRS
        ful_pairs = ful_pairs[:_MAX_FULFILLS_PAIRS]
        ful_keys = ful_keys[:_MAX_FULFILLS_PAIRS]

    if not addr_pairs and not ful_pairs:
        return IntentAlignmentResult()

    # --- single LLM call ---
    payload = json.dumps({
        "addresses": addr_pairs,
        "fulfills": ful_pairs,
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        "intent_alignment", payload, model,
        session_factory=session_factory, purpose="intent_alignment",
        key="",
    )

    all_edges: list[Edge] = []
    n_judged = 0

    if isinstance(raw, dict):
        addr_verdicts: list[Any] = raw.get("addresses", [])
        ful_verdicts: list[Any] = raw.get("fulfills", [])

        if addr_verdicts:
            by_id = _index_by_id(addr_verdicts)
            for idx, (step_id, constraint_id) in enumerate(addr_keys):
                verdict = by_id.get(idx)
                if not verdict:
                    continue
                n_judged += 1
                if str(verdict.get("outcome", "")) == "yes":
                    looked_up = index.steps.get((run_id, step_id))
                    r_id = looked_up.run_id if looked_up else run_id
                    all_edges.append(Edge(
                        id=stable_id("edge", r_id, "addresses", step_id, constraint_id),
                        kind="addresses", run_id=r_id,
                        src=step_id, dst=constraint_id,
                    ))

        if ful_verdicts:
            by_id = _index_by_id(ful_verdicts)
            for idx, (action_step_id, result_step_id) in enumerate(ful_keys):
                verdict = by_id.get(idx)
                if not verdict:
                    continue
                n_judged += 1
                if str(verdict.get("outcome", "")) == "yes":
                    action_step = index.steps.get((run_id, action_step_id))
                    r_id = action_step.run_id if action_step else run_id
                    all_edges.append(Edge(
                        id=stable_id("edge", r_id, "fulfills", action_step_id, result_step_id),
                        kind="fulfills", run_id=r_id,
                        src=action_step_id, dst=result_step_id,
                    ))

    for edge in all_edges:
        index.edges[edge.id] = edge

    result = IntentAlignmentResult(
        edges=all_edges,
        n_addresses_pairs=len(addr_pairs),
        n_fulfills_pairs=len(ful_pairs),
        n_judged=n_judged,
        n_capped=n_addr_capped + n_ful_capped,
    )
    logger.info(
        "intent_alignment: {} addresses + {} fulfills edges "
        "(from {} + {} pairs, {} judged, {} capped)",
        sum(1 for e in all_edges if e.kind == "addresses"),
        sum(1 for e in all_edges if e.kind == "fulfills"),
        len(addr_pairs), len(ful_pairs), n_judged, result.n_capped,
    )
    return result
