"""Tool atom: read the Pareto candidate frontier for a target scenario.

See ``.claude/designs/per-task-evolution-loop.md`` §11.1 / GEPA summary
§6.2.1 (Pareto pool).

Walks ``.agentm/decisions/<scenario>/candidates/*.json`` and returns the
frontier — a candidate is on the frontier iff it is the strict winner on
>=1 task across the pool. ``.pruned`` sidecars (written by
``tool_propose_change``'s pruning pass) mark dominated peers; this atom
honors the sidecar at read time so the frontier is consistent regardless
of when the prune ran. The cache regenerable property holds: re-running
this atom from the same on-disk state yields the same frontier.

Read-only — no writes. Constitution-protected paths are read directly
because no mutation is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_query_candidates",
    description=(
        "Read the Pareto candidate frontier for a scenario. Returns "
        "frontier members (strict winners on >=1 task) plus dominated "
        "ids for audit. Read-only."
    ),
    registers=("tool:query_candidates",),
    config_schema={
        "type": "object",
        "properties": {
            "default_scenario": {"type": "string"},
        },
        "additionalProperties": True,
    },
)


_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_scenario": {
            "type": "string",
            "description": (
                "Scenario key under .agentm/decisions/. If omitted, the "
                "atom's install-time default_scenario is used."
            ),
        },
    },
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    default_scenario = str(config.get("default_scenario") or "")
    cwd = Path(api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        scenario = str(args.get("target_scenario") or default_scenario or "")
        if not scenario:
            return _error(
                "target_scenario is required (or set default_scenario at "
                "install time)"
            )
        decisions_dir = cwd / ".agentm" / "decisions" / scenario
        candidates_dir = decisions_dir / "candidates"
        if not candidates_dir.is_dir():
            return _ok(
                json.dumps(
                    {"frontier": [], "dominated": []},
                    indent=2,
                )
            )

        records = _load_candidates(candidates_dir)
        win_map = _compute_win_tasks(records)

        frontier: list[dict[str, Any]] = []
        dominated: list[str] = []
        for cid, rec in records.items():
            wins = sorted(win_map.get(cid, []))
            if wins:
                frontier.append(
                    {
                        "candidate_id": cid,
                        "change_spec": rec.get("change_spec"),
                        "win_tasks": wins,
                        "score_summary": _score_summary(rec),
                        "parent_ids": _coerce_parent_ids(rec),
                    }
                )
            else:
                dominated.append(cid)

        # Sort frontier by win count desc, then by aggregate score desc,
        # for a stable presentation.
        frontier.sort(
            key=lambda r: (
                -len(r["win_tasks"]),
                -float(r["score_summary"].get("aggregate", 0.0)),
                r["candidate_id"],
            )
        )
        dominated.sort()
        return _ok(
            json.dumps(
                {
                    "scenario": scenario,
                    "frontier": frontier,
                    "dominated": dominated,
                },
                indent=2,
            )
        )

    api.register_tool(
        FunctionTool(
            name="query_candidates",
            description=(
                "Return the Pareto frontier of evolved candidates for a "
                "scenario. Each frontier entry lists the tasks the "
                "candidate uniquely wins on plus a score summary."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


# ---------------------------------------------------------------------------


def _load_candidates(candidates_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(candidates_dir.glob("c_*.json")):
        try:
            with p.open("r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        cid = rec.get("candidate_id")
        if not isinstance(cid, str):
            continue
        if not isinstance(rec.get("per_task_scores"), dict):
            continue
        out[cid] = rec
    return out


def _compute_win_tasks(
    records: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """For each task seen across the pool, find the strict argmax
    candidate. Returns a {candidate_id: [task_ids]} map for strict
    winners only — ties on a task contribute no inclusion claim.
    Mirrors ``tool_propose_change._prune_dominated_candidates`` so the
    two views agree."""
    task_ids: set[str] = set()
    for rec in records.values():
        for tid in rec.get("per_task_scores", {}).keys():
            task_ids.add(tid)
    wins: dict[str, list[str]] = {cid: [] for cid in records}
    for tid in task_ids:
        best_score = float("-inf")
        best_holders: list[str] = []
        for cid, rec in records.items():
            score = rec["per_task_scores"].get(tid)
            if not isinstance(score, (int, float)):
                continue
            score_f = float(score)
            if score_f > best_score:
                best_score = score_f
                best_holders = [cid]
            elif score_f == best_score:
                best_holders.append(cid)
        if len(best_holders) == 1:
            wins[best_holders[0]].append(tid)
    return wins


def _coerce_parent_ids(rec: dict[str, Any]) -> list[str]:
    """B-4 schema migration: candidate records emitted by Wave-Crossover
    write ``parent_ids: list[str]``; pre-B-4 fixtures (and tests) wrote
    ``parent_id: str | None``. Accept both shapes — the new field wins
    when present, else fall back to the legacy single-parent field. The
    return is always a list (possibly empty); downstream consumers
    don't need to know which schema the record was written under.
    """
    raw = rec.get("parent_ids")
    if isinstance(raw, list):
        return [str(x) for x in raw if isinstance(x, str) and x]
    legacy = rec.get("parent_id")
    if isinstance(legacy, str) and legacy:
        return [legacy]
    return []


def _score_summary(rec: dict[str, Any]) -> dict[str, float]:
    scores = rec.get("per_task_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return {"aggregate": 0.0, "task_count": 0.0}
    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    aggregate = sum(values) / len(values) if values else 0.0
    return {"aggregate": aggregate, "task_count": float(len(values))}


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
