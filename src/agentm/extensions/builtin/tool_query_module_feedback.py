"""Tool atom: surface the recent per-module feedback distribution (B-5).

See ``.claude/designs/per-task-evolution-loop.md`` §11.5 / GEPA summary
§6.2.3 (per-module credit assignment).

Reads the most recent ``.agentm/eval_runs/<id>.jsonl`` summaries (the
``eval_run.summary`` line carries ``feedback_corpus`` written by A-3,
and the per-task ``module_feedback_union`` lines carry the grader's
module fingering) and projects them into a ``{module: [feedback_text]}``
distribution the tuner can use for round-robin module selection.

The atom is filterable by ``target_scenario`` so an eval-run summary
unrelated to the current scenario doesn't pollute the distribution. We
filter via ``task_class`` on the summary header, since task_class is the
field that anchors a scenario's eval suite (see ``tool_eval_run`` and
``tool_query_traces`` for the same convention).

Read-only — no writes. Pure scaffolding for the tuner prompt; no code
in the SDK enforces the round-robin policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_query_module_feedback",
    description=(
        "Read recent eval-run summaries and project the grader's "
        "module_feedback into a {module: [feedback_text]} distribution. "
        "Read-only. Used by the tuner for round-robin module selection."
    ),
    registers=("tool:query_module_feedback",),
    config_schema={
        "type": "object",
        "properties": {
            "default_scenario": {"type": "string"},
        },
        "additionalProperties": True,
    },
)


_PARAMETERS: Final[dict[str, Any]] = {
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    default_scenario = str(config.get("default_scenario") or "")
    cwd = Path(api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
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
            # The summary's feedback_corpus is the corpus of free-text
            # feedback entries (per task x sample). The per-task records
            # carry module_feedback_union — last-write-wins per task.
            # We bucket each module-keyed feedback under its module.
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

        # Sort module keys deterministically for stable output.
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
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


# ---------------------------------------------------------------------------


def _load_run(
    path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return ``(summary, task_records)`` for an eval-run JSONL. The
    summary is the single ``eval_run.summary`` header line; task records
    are the trailing ``eval_run.task`` lines.
    """
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


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])
