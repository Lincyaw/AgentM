"""Pass 3 fold: value flow analysis.

Pure-code fold over valued references.  Produces two summaries an auditor
actually reads:

1. **Value timelines** -- per value-symbol, the sequence of distinct values
   with the step where each appeared.  Deduplicates consecutive repeats.
2. **Constraint checks** -- matches constraint target values against the
   final observed value of the same symbol.

No model calls (except constraint checks which are LLM-judged).
All inputs come from the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex


@dataclass(frozen=True, slots=True)
class ValuePoint:
    step_id: str
    step_index: int
    value: str
    kind: str          # tool_output (read) | tool_input (old before edit)


@dataclass(frozen=True, slots=True)
class ValueTimeline:
    symbol_name: str
    symbol_id: str
    points: tuple[ValuePoint, ...]


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    constraint_id: str
    description: str
    target_symbol: str
    target_value: str
    actual_value: str
    status: str         # "met" | "violated" | "irrelevant"
    reason: str = ""


def build_value_timelines(index: TrajectoryIndex) -> list[ValueTimeline]:
    """Per value-symbol, deduplicated sequence of value transitions."""
    step_index_cache: dict[str, int] = {}
    for (_, sid), step in index.steps.items():
        step_index_cache[sid] = step.index

    sym_refs: dict[str, list[tuple[int, str, str, str]]] = {}
    for ref in index.references.values():
        if not ref.value:
            continue
        sym = index.symbols.get(ref.symbol_id)
        if not sym or sym.entity_class != "value":
            continue
        si = step_index_cache.get(ref.step_id, 0)
        key = sym.id
        if key not in sym_refs:
            sym_refs[key] = []
        sym_refs[key].append((si, ref.step_id, ref.value, ref.kind))

    timelines: list[ValueTimeline] = []
    for sym_id, entries in sorted(sym_refs.items()):
        sym = index.symbols[sym_id]
        entries.sort(key=lambda e: (e[0], e[3] == "tool_output"))
        points: list[ValuePoint] = []
        prev_val: str | None = None
        for si, sid, val, kind in entries:
            if val != prev_val:
                points.append(ValuePoint(
                    step_id=sid, step_index=si, value=val, kind=kind,
                ))
                prev_val = val
        if len(points) >= 1:
            timelines.append(ValueTimeline(
                symbol_name=sym.canonical_name,
                symbol_id=sym_id,
                points=tuple(points),
            ))

    timelines.sort(key=lambda t: -len(t.points))
    return timelines


def _final_values(index: TrajectoryIndex) -> dict[str, str]:
    """Last observed value for each value-symbol, by step index."""
    step_idx: dict[str, int] = {}
    for (_, sid), step in index.steps.items():
        step_idx[sid] = step.index

    latest: dict[str, tuple[int, str]] = {}
    for ref in index.references.values():
        if not ref.value:
            continue
        sym = index.symbols.get(ref.symbol_id)
        if not sym:
            continue
        si = step_idx.get(ref.step_id, -1)
        prev = latest.get(sym.canonical_name)
        if prev is None or si > prev[0]:
            latest[sym.canonical_name] = (si, ref.value)

    return {name: val for name, (_, val) in latest.items()}


def _build_trajectory_context(index: TrajectoryIndex) -> str:
    """Build a compact trajectory context: task description + action log."""
    import json as _json

    from ..ir.models import StepRole

    steps = sorted(index.steps.values(), key=lambda s: s.index)

    # Task: first user step content (truncated)
    task_text = ""
    for s in steps:
        if s.role == StepRole.USER and s.content.strip():
            task_text = s.content[:3000]
            break

    # Action log: tool_call steps with purpose + outcome hint from next tool_result
    result_by_call: dict[str, str] = {}
    for s in steps:
        if s.role == StepRole.TOOL_RESULT and s.call_id:
            result_by_call[s.call_id] = s.content[:200]

    action_lines: list[str] = []
    for s in steps:
        if s.tool_name is None:
            continue
        purpose = ""
        content_start = s.content.find("\n")
        if content_start >= 0:
            try:
                args = _json.loads(s.content[content_start + 1:])
                if isinstance(args, dict):
                    purpose = str(args.get("purpose", ""))
            except (ValueError, TypeError):
                pass
        outcome = ""
        if s.call_id and s.call_id in result_by_call:
            outcome = result_by_call[s.call_id].replace("\n", " ")[:100]
        parts = [f"step {s.step_id} [{s.tool_name}]"]
        if purpose:
            parts.append(purpose)
        if outcome:
            parts.append(f"→ {outcome}")
        action_lines.append(": ".join(parts[:2]) + (f"  {parts[2]}" if len(parts) > 2 else ""))

    sections: list[str] = []
    if task_text:
        sections.append(f"## Task\n{task_text}")
    if action_lines:
        sections.append("## Action log\n" + "\n".join(action_lines))
    return "\n\n".join(sections)


async def build_constraint_checks(
    index: TrajectoryIndex,
    *,
    model: str | None = None,
    session_factory: Any = None,
) -> list[ConstraintCheck]:
    """LLM-judged constraint satisfaction against final observed values."""
    from loguru import logger

    constraints = list(index.constraints.values())
    if not constraints:
        return []

    finals = _final_values(index)
    if not finals:
        return []

    context = _build_trajectory_context(index)
    con_lines = [f"[{i}] {c.description}" for i, c in enumerate(constraints)]
    val_lines = [f"  {name}: {val}" for name, val in sorted(finals.items())]
    payload = (
        context
        + "\n\n## Constraints\n" + "\n".join(con_lines)
        + "\n\n## Final observed values\n" + "\n".join(val_lines)
    )

    if session_factory is None:
        logger.info("constraint_checks: no session_factory, skipping LLM pass")
        return []

    from ..oracle import _ask_model

    raw = await _ask_model(
        "constraint_check", payload,
        model=model, session_factory=session_factory,
        purpose="constraint_check", key="checks",
    )
    if raw is None:
        logger.warning("constraint_checks: model returned no result")
        return []

    checks: list[ConstraintCheck] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("id")
        if not isinstance(idx, int) or idx < 0 or idx >= len(constraints):
            continue
        status = str(item.get("status", ""))
        if status not in ("met", "violated", "irrelevant"):
            continue
        con = constraints[idx]
        checks.append(ConstraintCheck(
            constraint_id=con.id,
            description=con.description,
            target_symbol=str(item.get("symbol", "")),
            target_value=str(item.get("target", "")),
            actual_value=str(item.get("actual", "")),
            status=status,
            reason=str(item.get("reason", "")),
        ))

    return checks


def build_intent_coverage(index: TrajectoryIndex) -> list[dict[str, Any]]:
    """Per-constraint intent coverage from ``addresses``/``fulfills`` edges.

    Edges now use step_id as src (not call_id), so we read edge.src
    directly as a step_id.
    """
    constraints = list(index.constraints.values())
    if not constraints:
        return []

    addresses_by_constraint: dict[str, list[str]] = {}
    fulfills_src_steps: set[str] = set()

    for edge in index.edges.values():
        if edge.kind == "addresses":
            addresses_by_constraint.setdefault(edge.dst, []).append(edge.src)
        elif edge.kind == "fulfills":
            fulfills_src_steps.add(edge.src)

    coverage: list[dict[str, Any]] = []
    for constraint in constraints:
        addressing_step_ids = addresses_by_constraint.get(constraint.id, [])
        if not addressing_step_ids:
            coverage.append({
                "constraint_id": constraint.id,
                "description": constraint.description,
                "status": "unaddressed",
                "action_step_ids": [],
            })
            continue

        has_fulfills = any(sid in fulfills_src_steps for sid in addressing_step_ids)
        status = "verified_by_intent" if has_fulfills else "addressed"

        coverage.append({
            "constraint_id": constraint.id,
            "description": constraint.description,
            "status": status,
            "action_step_ids": addressing_step_ids,
        })

    return coverage


def build_value_flow_sync(index: TrajectoryIndex) -> dict[str, Any]:
    """Sync version for persistence.dump -- timelines only, no LLM."""
    timelines = build_value_timelines(index)
    return _format_value_flow(timelines, [])


async def build_value_flow(
    index: TrajectoryIndex,
    *,
    model: str | None = None,
    session_factory: Any = None,
) -> dict[str, Any]:
    """Run all value-flow folds and return a summary dict."""
    timelines = build_value_timelines(index)
    checks = await build_constraint_checks(
        index, model=model, session_factory=session_factory,
    )
    intent_cov = build_intent_coverage(index)
    return _format_value_flow(timelines, checks, intent_cov)


def _format_value_flow(
    timelines: list[ValueTimeline],
    checks: list[ConstraintCheck],
    intent_coverage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "value_timelines": [
            {
                "symbol": t.symbol_name,
                "points": [
                    {"step": p.step_id, "idx": p.step_index, "value": p.value, "kind": p.kind}
                    for p in t.points
                ],
            }
            for t in timelines
        ],
        "constraint_checks": [
            {
                "constraint_id": c.constraint_id,
                "description": c.description,
                "symbol": c.target_symbol,
                "target": c.target_value,
                "actual": c.actual_value,
                "status": c.status,
                "reason": c.reason,
            }
            for c in checks
        ],
        "intent_coverage": intent_coverage or [],
    }
