"""Pass 3 fold: value flow analysis.

Pure-code fold over Pass 1 actions + valued references.  Produces three
summaries an auditor actually reads:

1. **Value timelines** — per value-symbol, the sequence of distinct values
   with the step where each appeared.  Deduplicates consecutive repeats.
2. **Iteration cycles** — groups of write→execute→read, each showing what
   config changed and what metrics resulted.
3. **Constraint checks** — matches constraint target values against the
   final observed value of the same symbol.

No model calls.  All inputs come from the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex
    from ..ir.models import Action


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
class IterationCycle:
    index: int
    write_step_ids: tuple[str, ...]
    execute_step_id: str
    read_step_ids: tuple[str, ...]
    diffs: tuple[tuple[str, str, str], ...]   # (param, old, new)
    metrics: tuple[tuple[str, str], ...]       # (metric_name, value)


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
        # Deduplicate consecutive equal values.
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


def build_iteration_cycles(index: TrajectoryIndex) -> list[IterationCycle]:
    """Group actions into write→execute→read cycles."""
    step_index_for_id: dict[str, int] = {}
    for (_, sid), step in index.steps.items():
        step_index_for_id[sid] = step.index

    sorted_actions: list[tuple[int, Action]] = []
    for action in index.actions.values():
        si = step_index_for_id.get(action.step_id, 0)
        sorted_actions.append((si, action))
    sorted_actions.sort(key=lambda x: x[0])

    # Find execute actions as cycle boundaries.
    execute_indices: list[int] = []
    for i, (_, action) in enumerate(sorted_actions):
        if action.operation == "execute":
            execute_indices.append(i)

    if not execute_indices:
        return []

    # Collect valued refs by step for metric lookup.
    valued_by_step: dict[str, list[tuple[str, str]]] = {}
    for ref in index.references.values():
        if not ref.value:
            continue
        sym = index.symbols.get(ref.symbol_id)
        if not sym:
            continue
        if ref.step_id not in valued_by_step:
            valued_by_step[ref.step_id] = []
        valued_by_step[ref.step_id].append((sym.canonical_name, ref.value))

    cycles: list[IterationCycle] = []
    for ci, ei in enumerate(execute_indices):
        _, exec_action = sorted_actions[ei]

        # Look backward for writes before this execute (after previous execute).
        prev_boundary = execute_indices[ci - 1] + 1 if ci > 0 else 0
        writes: list[Action] = []
        all_diffs: list[tuple[str, str, str]] = []
        for j in range(prev_boundary, ei):
            _, action = sorted_actions[j]
            if action.operation == "write":
                writes.append(action)
                all_diffs.extend(action.diffs)

        # Look forward for reads after this execute (before next execute).
        next_boundary = execute_indices[ci + 1] if ci + 1 < len(execute_indices) else len(sorted_actions)
        reads: list[Action] = []
        for j in range(ei + 1, next_boundary):
            _, action = sorted_actions[j]
            if action.operation == "read":
                reads.append(action)

        # Collect metrics from read steps + the execute outcome's tool_result.
        metrics: dict[str, str] = {}
        # Check tool_result step right after execute.
        exec_step_idx = step_index_for_id.get(exec_action.step_id, 0)
        for sid, pairs in valued_by_step.items():
            si = step_index_for_id.get(sid, 0)
            if exec_step_idx < si <= (step_index_for_id.get(sorted_actions[next_boundary - 1][1].step_id, 0) if next_boundary > ei + 1 else exec_step_idx + 100):
                for name, val in pairs:
                    if name not in metrics:
                        metrics[name] = val

        cycles.append(IterationCycle(
            index=ci,
            write_step_ids=tuple(a.step_id for a in writes),
            execute_step_id=exec_action.step_id,
            read_step_ids=tuple(a.step_id for a in reads),
            diffs=tuple(all_diffs),
            metrics=tuple(sorted(metrics.items())),
        ))

    return cycles


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


_CONSTRAINT_CHECK_INSTRUCTIONS = """\
You are checking whether task constraints were satisfied by the agent's trajectory.

You will receive:
1. A list of constraints (id + description).
2. A table of final observed values for tracked symbols.

For each constraint, judge whether the final values satisfy it.

Reply with a JSON object:
```json
{"checks": [
  {"id": 0, "status": "met|violated|irrelevant", "symbol": "matched_symbol_or_empty", "target": "target_from_constraint", "actual": "observed_value", "reason": "one sentence"}
]}
```

- "met": the constraint is satisfied by the observed values.
- "violated": the constraint is clearly not satisfied.
- "irrelevant": the constraint is not checkable against numeric/string values (workflow instruction, structural rule, etc.).

Only output the JSON. No commentary.
"""


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

    # Build payload.
    con_lines = [f"[{i}] {c.description}" for i, c in enumerate(constraints)]
    val_lines = [f"  {name}: {val}" for name, val in sorted(finals.items())]
    payload = (
        "## Constraints\n" + "\n".join(con_lines)
        + "\n\n## Final observed values\n" + "\n".join(val_lines)
    )

    if session_factory is None:
        logger.info("constraint_checks: no session_factory, skipping LLM pass")
        return []

    from ..oracle import _ask_model

    raw = await _ask_model(
        _CONSTRAINT_CHECK_INSTRUCTIONS, payload,
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


def build_value_flow_sync(index: TrajectoryIndex) -> dict[str, Any]:
    """Sync version for persistence.dump — timelines + iterations only, no LLM."""
    timelines = build_value_timelines(index)
    cycles = build_iteration_cycles(index)
    return _format_value_flow(timelines, cycles, [])


async def build_value_flow(
    index: TrajectoryIndex,
    *,
    model: str | None = None,
    session_factory: Any = None,
) -> dict[str, Any]:
    """Run all value-flow folds and return a summary dict."""
    timelines = build_value_timelines(index)
    cycles = build_iteration_cycles(index)
    checks = await build_constraint_checks(
        index, model=model, session_factory=session_factory,
    )
    return _format_value_flow(timelines, cycles, checks)


def _format_value_flow(
    timelines: list[ValueTimeline],
    cycles: list[IterationCycle],
    checks: list[ConstraintCheck],
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
        "iterations": [
            {
                "index": c.index,
                "writes": list(c.write_step_ids),
                "execute": c.execute_step_id,
                "reads": list(c.read_step_ids),
                "diffs": [{"param": d[0], "old": d[1], "new": d[2]} for d in c.diffs],
                "metrics": dict(c.metrics),
            }
            for c in cycles
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
    }
