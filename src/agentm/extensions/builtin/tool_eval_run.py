"""Tool atom: run a pinned eval suite under a (possibly overridden)
fingerprint and aggregate per-task results.

See ``.claude/designs/per-task-evolution-loop.md`` §4 / §6.3.

Mechanism per call:

1. Read each task YAML in ``eval_dir/tasks/``.
2. For each task × ``samples_per_task``, spawn a child ``AgentSession``
   loaded with the ``target_scenario`` (configured at install time) and
   ``atom_source_overrides`` (passed per call).
3. Drive each child with the task's ``input.user_message`` and capture
   the assistant's final text reply.
4. Run the deterministic ``grader.py`` (or fall back to a noop 0.0 grade
   if absent) against each result.
5. Aggregate per-task mean/stddev and overall ``primary_score`` (mean of
   per-task means) plus universal guard metrics.
6. Append a per-task record to ``.agentm/eval_runs/<eval_run_id>.jsonl``.
7. Return the ``EvalRunResult`` payload.

The atom is deliberately conservative — it does not implement rubric
LLM grading (Phase 2). format_fix uses programmatic grading only.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.messages import AssistantMessage
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI
from agentm.harness.session_config import AgentSessionConfig


MANIFEST = ExtensionManifest(
    name="tool_eval_run",
    description=(
        "Spawn child sessions to run a pinned eval suite, optionally with "
        "atom source overrides. Aggregates primary + guard metrics and "
        "writes a per-task record to .agentm/eval_runs/<id>.jsonl."
    ),
    registers=("tool:eval_run",),
    config_schema={
        "type": "object",
        "properties": {
            "target_scenario": {"type": "string"},
            "eval_dir": {"type": "string"},
        },
        "additionalProperties": True,
    },
)


_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "eval_dir": {
            "type": "string",
            "description": (
                "Optional override of the configured eval directory; either "
                "absolute or scenario-relative."
            ),
        },
        "fingerprint": {
            "type": "object",
            "description": (
                "Optional fingerprint anchor (currently informational; "
                "recorded in the eval-run record)."
            ),
            "additionalProperties": {"type": "string"},
        },
        "atom_source_overrides": {
            "type": "object",
            "description": (
                "Map of atom_name -> new source string. Passed to each "
                "child session; overrides land in the child's eval-sandbox."
            ),
            "additionalProperties": {"type": "string"},
        },
        "samples_per_task": {
            "type": "integer",
            "default": 3,
            "description": "Number of independent runs per task.",
        },
        "holdout_only": {
            "type": "boolean",
            "default": False,
            "description": "If True, only run tasks marked holdout: true.",
        },
        "smoke": {
            "type": "boolean",
            "default": False,
            "description": "If True, run only the first 3 tasks (fast gating).",
        },
    },
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    target_scenario = config.get("target_scenario")
    default_eval_dir = config.get("eval_dir")
    cwd = Path(api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        eval_dir_arg = args.get("eval_dir") or default_eval_dir
        if not eval_dir_arg:
            return _error(
                "no eval_dir configured at install or supplied at call"
            )
        eval_dir = _resolve_eval_dir(cwd, eval_dir_arg)
        if not eval_dir.is_dir():
            return _error(f"eval_dir does not exist: {eval_dir}")

        overrides = args.get("atom_source_overrides") or None
        if overrides is not None and not isinstance(overrides, dict):
            return _error("atom_source_overrides must be a mapping")
        samples_per_task = max(1, int(args.get("samples_per_task", 3)))
        holdout_only = bool(args.get("holdout_only", False))
        smoke = bool(args.get("smoke", False))

        tasks = _load_tasks(eval_dir / "tasks")
        if not tasks:
            return _error(f"no tasks found in {eval_dir / 'tasks'}")
        if holdout_only:
            tasks = [t for t in tasks if t.get("holdout") is True]
        if smoke:
            tasks = tasks[:3]
        if not tasks:
            return _error("no tasks selected after filtering")

        grader = _load_grader(eval_dir / "grader.py")

        eval_run_id = f"er_{uuid.uuid4().hex[:12]}"
        task_class = _detect_task_class(tasks)

        per_task_records: list[dict[str, Any]] = []
        for task in tasks:
            task_id = str(task.get("id") or task.get("name") or "unknown")
            grades: list[float] = []
            tool_errors: list[int] = []
            turns_log: list[int] = []
            for sample_idx in range(samples_per_task):
                outcome = await _run_single_sample(
                    api=api,
                    target_scenario=target_scenario,
                    task=task,
                    task_class=task_class,
                    eval_run_id=eval_run_id,
                    task_id=task_id,
                    sample_idx=sample_idx,
                    overrides=overrides,
                )
                grade_value = grader(task, outcome["final_text"]) if grader else 0.0
                grades.append(float(grade_value))
                tool_errors.append(int(outcome.get("tool_errors", 0)))
                turns_log.append(int(outcome.get("turns", 0)))
            per_task_records.append(
                {
                    "task_id": task_id,
                    "holdout": bool(task.get("holdout", False)),
                    "samples": grades,
                    "grade_mean": _mean(grades),
                    "grade_stddev": _stddev(grades),
                    "tool_error_rate": _mean(
                        [1.0 if t else 0.0 for t in tool_errors]
                    ),
                    "turns_mean": _mean([float(t) for t in turns_log]),
                }
            )

        primary_scores = [r["grade_mean"] for r in per_task_records]
        primary_stderr = _stderr_of_mean(
            primary_scores, total_samples=samples_per_task * len(per_task_records)
        )
        primary_score = _mean(primary_scores)
        holdout_records = [r for r in per_task_records if r["holdout"]]
        holdout_score = (
            _mean([r["grade_mean"] for r in holdout_records])
            if holdout_records
            else None
        )

        guard_metrics = {
            "tool_error_rate": _mean(
                [r["tool_error_rate"] for r in per_task_records]
            ),
            "turns_mean": _mean([r["turns_mean"] for r in per_task_records]),
        }

        eval_runs_dir = cwd / ".agentm" / "eval_runs"
        eval_runs_dir.mkdir(parents=True, exist_ok=True)
        run_path = eval_runs_dir / f"{eval_run_id}.jsonl"
        _append_run_records(
            run_path,
            eval_run_id=eval_run_id,
            task_class=task_class,
            fingerprint=args.get("fingerprint"),
            atom_source_overrides=overrides,
            primary_score=primary_score,
            primary_stderr=primary_stderr,
            holdout_score=holdout_score,
            guard_metrics=guard_metrics,
            per_task=per_task_records,
            samples_per_task=samples_per_task,
        )

        result = {
            "eval_run_id": eval_run_id,
            "primary_score": primary_score,
            "primary_score_stderr": primary_stderr,
            "guard_metrics": guard_metrics,
            "holdout_score": holdout_score,
            "task_count": len(per_task_records),
            "samples_per_task": samples_per_task,
            "fingerprint": args.get("fingerprint"),
            "per_task": per_task_records,
            "record_path": str(run_path),
        }
        return _ok(json.dumps(result, indent=2))

    api.register_tool(
        FunctionTool(
            name="eval_run",
            description=(
                "Run the pinned eval suite under the configured target "
                "scenario. Optionally apply atom_source_overrides to "
                "evaluate proposed atom versions without mutating the "
                "working tree. Returns aggregated metrics + eval_run_id."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


# ---------------------------------------------------------------------------


def _resolve_eval_dir(cwd: Path, value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (cwd / value).resolve()


def _load_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    if not tasks_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict):
            payload.setdefault("id", path.stem)
            out.append(payload)
    return out


def _detect_task_class(tasks: list[dict[str, Any]]) -> str | None:
    for task in tasks:
        tc = task.get("task_class")
        if isinstance(tc, str) and tc:
            return tc
    return None


def _load_grader(grader_path: Path) -> Any:
    """Import ``grader.py`` and return its ``grade(task, output) -> float``
    callable, or ``None`` if the file is missing or imports fail.
    """
    if not grader_path.is_file():
        return None
    module_name = f"_agentm_grader_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, grader_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        sys.modules.pop(module_name, None)
        return None
    fn = getattr(module, "grade", None)
    return fn if callable(fn) else None


async def _run_single_sample(
    *,
    api: ExtensionAPI,
    target_scenario: str | None,
    task: dict[str, Any],
    task_class: str | None,
    eval_run_id: str,
    task_id: str,
    sample_idx: int,
    overrides: dict[str, str] | None,
) -> dict[str, Any]:
    """Spawn one child session, drive the task prompt, return the final
    assistant text + light counters."""
    child_config = AgentSessionConfig(
        cwd=api.cwd,
        scenario=target_scenario,
        provider=None,
        task_class=task_class,
        eval_run_id=eval_run_id,
        eval_task_id=task_id,
        purpose=f"eval_run:{eval_run_id}:{task_id}:{sample_idx}",
        atom_source_overrides=dict(overrides) if overrides else None,
    )
    user_message = _extract_user_message(task)
    final_text = ""
    tool_errors = 0
    turns = 0
    try:
        child = await api.spawn_child_session(child_config)
    except Exception as exc:  # noqa: BLE001
        return {
            "final_text": f"<spawn-failed: {exc}>",
            "tool_errors": 1,
            "turns": 0,
        }
    try:
        t0 = time.perf_counter()
        try:
            messages = await child.prompt(user_message)
        except Exception as exc:  # noqa: BLE001
            return {
                "final_text": f"<prompt-failed: {exc}>",
                "tool_errors": 1,
                "turns": 0,
            }
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                turns += 1
                # Capture last text content as the answer.
                for block in msg.content:
                    if hasattr(block, "text") and isinstance(block.text, str):
                        final_text = block.text
        del t0
    finally:
        try:
            await child.shutdown()
        except Exception:  # noqa: BLE001 - best effort
            pass

    return {
        "final_text": final_text,
        "tool_errors": tool_errors,
        "turns": turns,
    }


def _extract_user_message(task: dict[str, Any]) -> str:
    inp = task.get("input") or {}
    if isinstance(inp, dict):
        msg = inp.get("user_message")
        if isinstance(msg, str):
            return msg
    if isinstance(task.get("user_message"), str):
        return str(task["user_message"])
    return json.dumps(inp) if inp else ""


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _stderr_of_mean(values: list[float], total_samples: int) -> float:
    """Standard error of the mean, used by tool_propose_change for the
    statistical-sanity gate. Combines per-task variance and uses the
    total observation count as the divisor."""
    if total_samples < 2 or not values:
        return 0.0
    sigma = _stddev(values)
    return sigma / math.sqrt(max(1, total_samples))


def _append_run_records(
    path: Path,
    *,
    eval_run_id: str,
    task_class: str | None,
    fingerprint: dict[str, str] | None,
    atom_source_overrides: dict[str, str] | None,
    primary_score: float,
    primary_stderr: float,
    holdout_score: float | None,
    guard_metrics: dict[str, float],
    per_task: list[dict[str, Any]],
    samples_per_task: int,
) -> None:
    overrides_meta = (
        sorted(atom_source_overrides.keys())
        if isinstance(atom_source_overrides, dict)
        else None
    )
    header = {
        "kind": "eval_run.summary",
        "eval_run_id": eval_run_id,
        "task_class": task_class,
        "fingerprint": fingerprint,
        "atom_source_overrides": overrides_meta,
        "primary_score": primary_score,
        "primary_score_stderr": primary_stderr,
        "holdout_score": holdout_score,
        "guard_metrics": guard_metrics,
        "samples_per_task": samples_per_task,
        "task_count": len(per_task),
        "at": time.time(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for record in per_task:
            fh.write(
                json.dumps({"kind": "eval_run.task", "eval_run_id": eval_run_id, **record})
                + "\n"
            )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
