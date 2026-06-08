"""Grouped query tool atom: ``query_traces``, ``query_candidates``, and
``query_module_feedback``.

Merges the former single-tool atoms ``tool_query_traces``,
``tool_query_candidates``, and ``tool_query_module_feedback`` into one
§11-compliant module. The LLM-facing tool names are unchanged.

query_traces — query observability JSONL traces by ``task_class``.
Returns lightweight summaries; use the ``read`` tool for full bodies.

query_candidates — read the Pareto candidate frontier for a target
scenario. Read-only.

query_module_feedback — surface the recent per-module feedback
distribution from eval-run summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult, TraceReader
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

MANIFEST = ExtensionManifest(
    name="query_tools",
    description=(
        "Register the query_traces, query_candidates, and "
        "query_module_feedback tools for evolution observability."
    ),
    registers=(
        "tool:query_traces",
        "tool:query_candidates",
        "tool:query_module_feedback",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "default_scenario": {"type": "string"},
        },
        "additionalProperties": True,
    },
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


# ---------------------------------------------------------------------------
# query_traces helpers
# ---------------------------------------------------------------------------

_TRACES_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "task_class": {
            "type": "string",
            "description": "Required task_class to filter by.",
        },
        "fingerprint": {
            "type": "object",
            "description": (
                "Optional exact-match filter against session.fingerprint "
                "atoms map. Each key is an atom name; value must equal the "
                "trace's fingerprint atom hash."
            ),
            "additionalProperties": {"type": "string"},
        },
        "n": {
            "type": "integer",
            "default": 50,
            "description": "Max number of most-recent traces to return.",
        },
        "include_eval_runs": {
            "type": "boolean",
            "default": True,
            "description": (
                "If False, exclude traces produced by tool_eval_run "
                "(those carry a non-null eval_run_id)."
            ),
        },
        "only_eval_runs": {
            "type": "boolean",
            "default": False,
            "description": (
                "If True, return ONLY eval-run traces. Mutually exclusive "
                "with include_eval_runs=False (only_eval_runs wins)."
            ),
        },
    },
    "required": ["task_class"],
    "additionalProperties": False,
}


def _summarize_trace(path: Path) -> dict[str, Any] | None:
    """Walk an OTLP/JSON trace once and extract load-bearing identity fields."""
    timestamp_ns: int | None = None
    task_class: str | None = None
    eval_run_id: str | None = None
    task_id: str | None = None
    fingerprint_atoms: dict[str, str] = {}
    stop_reason: str | None = None
    seen_identity = False

    if not path.is_file():
        return None
    reader = TraceReader(path)

    for record in reader.iter_log_records():
        event_name = record.event_name
        body = record.body
        if event_name == "agentm.session.fingerprint":
            seen_identity = True
            if isinstance(body, dict):
                task_meta = body.get("task_meta") or {}
                if isinstance(task_meta, dict):
                    task_class = task_meta.get("task_class") or task_class
                    eval_run_id = task_meta.get("eval_run_id") or eval_run_id
                    task_id = task_meta.get("task_id") or task_id
                atoms = body.get("atoms") or {}
                if isinstance(atoms, dict):
                    fingerprint_atoms = {
                        str(k): str(v) for k, v in atoms.items()
                    }
            if timestamp_ns is None:
                timestamp_ns = record.time_unix_nano
        elif event_name == "agentm.session.ready":
            seen_identity = True
            if timestamp_ns is None:
                timestamp_ns = record.time_unix_nano
        elif event_name == "agentm.agent.end" and isinstance(body, dict):
            sr = body.get("stop_reason")
            if isinstance(sr, str) and sr:
                stop_reason = sr
            cause = body.get("cause")
            if isinstance(cause, dict) and not stop_reason:
                cause_kind = cause.get("cause_kind")
                if isinstance(cause_kind, str):
                    stop_reason = cause_kind

    total_cost = 0.0
    total_turns = 0
    for span in reader.chat_calls():
        total_turns += 1
        for cost_key in ("agentm.cost_usd", "gen_ai.usage.cost_usd"):
            value = span.attributes.get(cost_key)
            if isinstance(value, (int, float)):
                total_cost += float(value)

    if not seen_identity:
        return None
    return {
        "trace_id": path.stem,
        "path": str(path),
        "timestamp": _ns_to_iso(timestamp_ns),
        "task_class": task_class,
        "eval_run_id": eval_run_id,
        "task_id": task_id,
        "stop_reason": stop_reason,
        "total_cost_usd": round(total_cost, 6),
        "total_turns": total_turns,
        "fingerprint_atoms": fingerprint_atoms,
    }


def _ns_to_iso(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    import datetime as dt

    return dt.datetime.fromtimestamp(value / 1e9, tz=dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# query_candidates helpers
# ---------------------------------------------------------------------------

_CANDIDATES_PARAMETERS: Final[dict[str, Any]] = {
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
    """For each task seen across the pool, find the strict argmax candidate."""
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
    """B-4 schema migration: accept both ``parent_ids`` and ``parent_id``."""
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


# ---------------------------------------------------------------------------
# query_module_feedback helpers
# ---------------------------------------------------------------------------

_FEEDBACK_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "target_scenario": {
            "type": "string",
            "description": (
                "Scenario task_class to filter by. If omitted, the atom's "
                "install-time default_scenario is used."
            ),
        },
        "n": {
            "type": "integer",
            "default": 20,
            "description": (
                "Max number of most-recent matching eval-run summaries to "
                "scan."
            ),
        },
    },
    "additionalProperties": False,
}


def _load_run(
    path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return ``(summary, task_records)`` for an eval-run JSONL."""
    summary: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("kind")
                if kind == "eval_run.summary" and summary is None:
                    summary = rec
                elif kind == "eval_run.task":
                    tasks.append(rec)
    except OSError:
        return None, []
    return summary, tasks


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    default_scenario = str(config.get("default_scenario") or "")
    cwd = Path(api.cwd)

    from agentm.core.lib.observability_dir import resolve_observability_dir

    obs_dir = resolve_observability_dir(cwd)

    # --- query_traces tool ------------------------------------------------

    async def _traces_execute(args: dict[str, Any]) -> ToolResult:
        task_class = str(args["task_class"])
        fingerprint_filter = args.get("fingerprint") or None
        if fingerprint_filter is not None and not isinstance(
            fingerprint_filter, dict
        ):
            return _error(
                "fingerprint must be a mapping of atom_name -> hash"
            )
        n = int(args.get("n", 50))
        include_eval_runs = bool(args.get("include_eval_runs", True))
        only_eval_runs = bool(args.get("only_eval_runs", False))

        if not obs_dir.is_dir():
            return _ok(json.dumps({"traces": []}))

        files = sorted(
            (p for p in obs_dir.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        out: list[dict[str, Any]] = []
        for path in files:
            summary = _summarize_trace(path)
            if summary is None:
                continue
            if summary.get("task_class") != task_class:
                continue
            has_eval_id = summary.get("eval_run_id") is not None
            if only_eval_runs and not has_eval_id:
                continue
            if not include_eval_runs and has_eval_id:
                continue
            if fingerprint_filter is not None:
                atoms = summary.get("fingerprint_atoms") or {}
                if not all(
                    atoms.get(k) == v for k, v in fingerprint_filter.items()
                ):
                    continue
            surfaced = {k: v for k, v in summary.items() if k != "fingerprint_atoms"}
            out.append(surfaced)
            if len(out) >= n:
                break

        return _ok(json.dumps({"traces": out}, indent=2))

    api.register_tool(
        FunctionTool(
            name="query_traces",
            description=(
                "Filter .agentm/observability/*.jsonl traces by task_class "
                "and optional fingerprint. Returns summaries only — fetch "
                "full bodies via the `read` tool."
            ),
            parameters=_TRACES_PARAMETERS,
            fn=_traces_execute,
        )
    )

    # --- query_candidates tool --------------------------------------------

    async def _candidates_execute(args: dict[str, Any]) -> ToolResult:
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
            parameters=_CANDIDATES_PARAMETERS,
            fn=_candidates_execute,
        )
    )

    # --- query_module_feedback tool ---------------------------------------

    async def _feedback_execute(args: dict[str, Any]) -> ToolResult:
        scenario = str(args.get("target_scenario") or default_scenario or "")
        n = max(1, int(args.get("n", 20)))
        eval_runs_dir = cwd / ".agentm" / "eval_runs"
        if not eval_runs_dir.is_dir():
            return _ok(
                json.dumps(
                    {
                        "module_distribution": {},
                        "total_recent": 0,
                        "scenario": scenario or None,
                    },
                    indent=2,
                )
            )

        files = sorted(
            (p for p in eval_runs_dir.glob("er_*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        module_distribution: dict[str, list[str]] = {}
        scanned = 0
        for path in files:
            if scanned >= n:
                break
            summary, task_records = _load_run(path)
            if summary is None:
                continue
            if scenario and summary.get("task_class") != scenario:
                continue
            scanned += 1
            for rec in task_records:
                mod_map = rec.get("module_feedback_union")
                if not isinstance(mod_map, dict):
                    continue
                for module_name, fb_text in mod_map.items():
                    if not isinstance(module_name, str) or not module_name:
                        continue
                    if not isinstance(fb_text, str) or not fb_text:
                        continue
                    module_distribution.setdefault(module_name, []).append(
                        fb_text
                    )

        ordered = {
            k: module_distribution[k] for k in sorted(module_distribution)
        }
        return _ok(
            json.dumps(
                {
                    "module_distribution": ordered,
                    "total_recent": scanned,
                    "scenario": scenario or None,
                },
                indent=2,
            )
        )

    api.register_tool(
        FunctionTool(
            name="query_module_feedback",
            description=(
                "Return the recent per-module feedback distribution from "
                "eval-run summaries. Surfaces grader fingering so the tuner "
                "can round-robin its mutation target."
            ),
            parameters=_FEEDBACK_PARAMETERS,
            fn=_feedback_execute,
        )
    )
