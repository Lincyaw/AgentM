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
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Final, TypedDict

import yaml

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.messages import AssistantMessage
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.session_config import AgentSessionConfig


class GradeResult(TypedDict):
    """mu_f feedback function output (design §3.2). Graders may return
    this dict directly or a bare ``float``; ``_normalize_grade`` wraps
    the latter so downstream aggregation always sees the full shape.

    ``failure_kind`` is a free-text grader-supplied tag distinguishing
    runtime failure from metric regression so downstream observers (the
    4-floor gate, reflection prompts) can branch. Conventional values:
    ``"ok" | "correctness" | "runtime" | "timeout" | "regression"`` —
    convention only, never enforced. ``None`` means the grader did not
    label this sample.
    """

    score: float
    dimensions: dict[str, float]
    feedback_text: str
    module_feedback: dict[str, str]
    failure_kind: str | None


# Cap on the feedback_corpus written to the eval-run summary. Without a
# cap a long run with chatty graders blows the JSONL line size; 200 is a
# soft ceiling that covers typical eval suites (40 tasks x 5 samples).
_FEEDBACK_CORPUS_CAP = 200


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
            "max_cost_usd": {
                "type": "number",
                "description": (
                    "Per-tuning-session USD budget. When exceeded, the "
                    "eval run aborts BETWEEN tasks (never mid-task)."
                ),
            },
            "rollouts_budget": {
                "type": ["integer", "null"],
                "description": (
                    "B-6: per-tuning-session rollout cap. Each task x "
                    "sample counts as one rollout. Aborts BETWEEN tasks "
                    "when the cap is hit; refuses the call entirely when "
                    "already exhausted at start."
                ),
            },
        },
        "additionalProperties": True,
    },
)


_PARAMETERS: Final[dict[str, Any]] = {
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
    max_cost_usd_raw = config.get("max_cost_usd")
    max_cost_usd: float | None
    try:
        max_cost_usd = (
            float(max_cost_usd_raw) if max_cost_usd_raw is not None else None
        )
    except (TypeError, ValueError):
        max_cost_usd = None
    rollouts_budget_raw = config.get("rollouts_budget")
    rollouts_budget: int | None
    try:
        rollouts_budget = (
            int(rollouts_budget_raw) if rollouts_budget_raw is not None else None
        )
    except (TypeError, ValueError):
        rollouts_budget = None
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

        scenario_key = _scenario_key(target_scenario)
        budget = _load_budget(cwd, scenario_key)
        # B-6: refuse the call entirely if a configured cap is already
        # exhausted before any work happens. Mid-run aborts continue to
        # land between tasks (loop guard below).
        if (
            max_cost_usd is not None
            and budget["usd_used"] >= max_cost_usd
        ):
            return _error(
                f"budget_exhausted: usd_used={budget['usd_used']:.4f} "
                f">= max_cost_usd={max_cost_usd:.4f}"
            )
        if (
            rollouts_budget is not None
            and budget["rollouts_used"] >= rollouts_budget
        ):
            return _error(
                f"budget_exhausted: rollouts_used={budget['rollouts_used']} "
                f">= rollouts_budget={rollouts_budget}"
            )
        usd_used_in_run = 0.0
        aborted_due_to_budget = False

        per_task_records: list[dict[str, Any]] = []
        feedback_corpus: list[dict[str, Any]] = []
        for task in tasks:
            # A-5 budget cap: enforced BETWEEN tasks (never mid-task) so
            # we don't stop a child session in flight. Last-writer-wins
            # if two eval runs race; B-6 hardens this.
            if (
                max_cost_usd is not None
                and budget["usd_used"] >= max_cost_usd
            ):
                aborted_due_to_budget = True
                break
            if (
                rollouts_budget is not None
                and budget["rollouts_used"] >= rollouts_budget
            ):
                aborted_due_to_budget = True
                break
            task_id = str(task.get("id") or task.get("name") or "unknown")
            grades: list[float] = []
            tool_errors: list[int] = []
            turns_log: list[int] = []
            feedback_texts: list[str] = []
            failure_kinds: list[str | None] = []
            # Last-write-wins union per design §3.2 / task A-3 acceptance:
            # multiple samples on the same task overwrite earlier module
            # feedback. Documented here so callers don't expect joining.
            module_feedback_union: dict[str, str] = {}
            task_cost_usd = 0.0
            task_rollouts = 0
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
                child_session_id = outcome.get("session_id")
                if isinstance(child_session_id, str) and child_session_id:
                    task_cost_usd += _read_trace_cost_usd(
                        cwd, child_session_id
                    )
                task_rollouts += 1
                raw_grade = (
                    grader(task, outcome["final_text"]) if grader else 0.0
                )
                grade = _normalize_grade(raw_grade)
                grades.append(grade["score"])
                tool_errors.append(int(outcome.get("tool_errors", 0)))
                turns_log.append(int(outcome.get("turns", 0)))
                feedback_texts.append(grade["feedback_text"])
                failure_kinds.append(grade.get("failure_kind"))
                module_feedback_union.update(grade["module_feedback"])
                if (
                    len(feedback_corpus) < _FEEDBACK_CORPUS_CAP
                    and grade["feedback_text"]
                ):
                    feedback_corpus.append(
                        {
                            "task_id": task_id,
                            "sample_idx": sample_idx,
                            "feedback_text": grade["feedback_text"],
                        }
                    )
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
                    "feedback_texts": feedback_texts,
                    "failure_kinds": failure_kinds,
                    "module_feedback_union": module_feedback_union,
                    "usd_used": task_cost_usd,
                }
            )
            usd_used_in_run += task_cost_usd
            budget["usd_used"] = budget["usd_used"] + task_cost_usd
            budget["rollouts_used"] = budget["rollouts_used"] + task_rollouts
            budget["updated_at"] = time.time()
            _save_budget(cwd, scenario_key, budget)

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
        # Aggregate per-sample ``failure_kind`` tags into a single
        # summary-level signal. Convention: most-common non-"ok" tag
        # wins; if every sample is "ok" or None, the summary tag is
        # "ok" (when any sample was tagged "ok") or None (when no
        # grader emitted a tag at all). This is a heuristic — observers
        # branching on the summary tag have access to the per-task
        # ``failure_kinds`` list for richer analysis.
        all_kinds: list[str] = []
        for rec in per_task_records:
            for fk in rec.get("failure_kinds") or []:
                if isinstance(fk, str) and fk:
                    all_kinds.append(fk)
        run_failure_kind: str | None
        if not all_kinds:
            run_failure_kind = None
        else:
            non_ok = [k for k in all_kinds if k != "ok"]
            if non_ok:
                run_failure_kind = max(set(non_ok), key=non_ok.count)
            else:
                run_failure_kind = "ok"

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
            feedback_corpus=feedback_corpus,
            usd_used_in_run=usd_used_in_run,
            aborted_due_to_budget=aborted_due_to_budget,
            failure_kind=run_failure_kind,
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
            "usd_used_in_run": usd_used_in_run,
            "aborted_due_to_budget": aborted_due_to_budget,
            "failure_kind": run_failure_kind,
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
    child_session_id: str | None = None
    # Per-task env vars: tasks may declare top-level ``env: {KEY: VAL}`` or
    # rely on the ``input.fixtures: [path]`` convention (rca-style scenarios
    # consume AGENTM_RCA_DATA_DIR). Set them on os.environ around the spawn
    # so atoms loaded by the child see them at install time, then restore.
    env_overrides = _collect_task_env(task)
    saved_env: dict[str, str | None] = {}
    for k, v in env_overrides.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        child = await api.spawn_child_session(child_config)
    except Exception as exc:  # noqa: BLE001
        return {
            "final_text": f"<spawn-failed: {exc}>",
            "tool_errors": 1,
            "turns": 0,
            "session_id": None,
        }
    try:
        child_session_id = getattr(child, "session_id", None)
        t0 = time.perf_counter()
        try:
            messages = await child.prompt(user_message)
        except Exception as exc:  # noqa: BLE001
            return {
                "final_text": f"<prompt-failed: {exc}>",
                "tool_errors": 1,
                "turns": 0,
                "session_id": child_session_id,
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
        for k, prev in saved_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev

    return {
        "final_text": final_text,
        "tool_errors": tool_errors,
        "turns": turns,
        "session_id": child_session_id,
    }


def _collect_task_env(task: dict[str, Any]) -> dict[str, str]:
    """Resolve per-task environment overrides.

    Honors two sources, in this precedence:

    1. Explicit ``env: {KEY: VAL}`` mapping at the top of the task YAML.
    2. Convention: ``input.fixtures: [path, ...]`` → ``AGENTM_RCA_DATA_DIR``
       points at the first entry. This keeps existing rca eval YAMLs
       working without an explicit ``env`` block.
    """
    out: dict[str, str] = {}
    fixtures = (task.get("input") or {}).get("fixtures") if isinstance(task.get("input"), dict) else None
    if isinstance(fixtures, list) and fixtures:
        first = fixtures[0]
        if isinstance(first, str) and first:
            out["AGENTM_RCA_DATA_DIR"] = first
    explicit = task.get("env")
    if isinstance(explicit, dict):
        for k, v in explicit.items():
            if isinstance(k, str) and isinstance(v, (str, int, float)):
                out[k] = str(v)
    return out


def _extract_user_message(task: dict[str, Any]) -> str:
    inp = task.get("input") or {}
    if isinstance(inp, dict):
        msg = inp.get("user_message")
        if isinstance(msg, str):
            return msg
    if isinstance(task.get("user_message"), str):
        return str(task["user_message"])
    return json.dumps(inp) if inp else ""


def _scenario_key(target_scenario: str | None) -> str:
    """Map ``target_scenario`` (which may be a path or a name) to a stable
    directory key under ``.agentm/decisions/<key>/``."""
    if not target_scenario:
        return "default"
    p = Path(target_scenario)
    # If it looks like a path (absolute or contains a separator), use the
    # basename so the budget lands under a clean directory name.
    if p.is_absolute() or "/" in target_scenario or "\\" in target_scenario:
        return p.name or "default"
    return target_scenario


def _budget_path(cwd: Path, scenario_key: str) -> Path:
    return cwd / ".agentm" / "decisions" / scenario_key / "budget.json"


def _load_budget(cwd: Path, scenario_key: str) -> dict[str, Any]:
    """Load ``budget.json`` for the scenario or return a fresh zeroed
    record. The file is constitution-protected (.agentm/decisions/** in
    core-manifest.yaml); only this atom + tool_propose_change write
    here."""
    path = _budget_path(cwd, scenario_key)
    if not path.is_file():
        return {
            "scenario": scenario_key,
            "rollouts_used": 0,
            "usd_used": 0.0,
            "updated_at": 0.0,
        }
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {
            "scenario": scenario_key,
            "rollouts_used": 0,
            "usd_used": 0.0,
            "updated_at": 0.0,
        }
    # Coerce shape — protect against earlier writes from a different code
    # version.
    return {
        "scenario": str(data.get("scenario") or scenario_key),
        "rollouts_used": int(data.get("rollouts_used") or 0),
        "usd_used": float(data.get("usd_used") or 0.0),
        "updated_at": float(data.get("updated_at") or 0.0),
    }


def _save_budget(cwd: Path, scenario_key: str, budget: dict[str, Any]) -> None:
    """Atomic write via temp + os.replace. Race semantics for concurrent
    eval runs are last-writer-wins (B-6 hardens this)."""
    import os

    path = _budget_path(cwd, scenario_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(budget, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_trace_cost_usd(cwd: Path, session_id: str) -> float:
    """Sum ``cost_usd`` across all ``llm.request.end`` records in the
    child session's observability JSONL. Returns 0.0 if the trace is
    missing or unreadable — the budget under-counts rather than aborts."""
    trace_path = cwd / ".agentm/observability" / f"{session_id}.jsonl"
    if not trace_path.is_file():
        return 0.0
    total = 0.0
    try:
        with trace_path.open("r", encoding="utf-8") as fh:
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
                if rec.get("kind") != "llm.request.end":
                    continue
                attrs = rec.get("attributes") or {}
                cost = attrs.get("cost_usd") if isinstance(attrs, dict) else None
                if isinstance(cost, (int, float)):
                    total += float(cost)
    except OSError:
        return 0.0
    return total


def _normalize_grade(value: Any) -> GradeResult:
    """Adapt a grader return value to the μ_f shape (design §3.2).

    - Bare ``float`` / ``int`` / ``bool`` is wrapped with empty diagnostic
      fields (back-compat shim — keeps third-party scalar graders working).
    - A ``dict`` is interpreted as a partial GradeResult; missing fields
      are filled with safe defaults; unexpected fields are dropped.
    - Anything else is treated as score 0.0 with a hint that the grader
      misbehaved (recorded in feedback_text for debuggability).
    """
    if isinstance(value, bool):
        # bool is a subclass of int — handle before the int branch.
        return {
            "score": float(value),
            "dimensions": {},
            "feedback_text": "",
            "module_feedback": {},
            "failure_kind": None,
        }
    if isinstance(value, (int, float)):
        return {
            "score": float(value),
            "dimensions": {},
            "feedback_text": "",
            "module_feedback": {},
            "failure_kind": None,
        }
    if isinstance(value, dict):
        score_raw = value.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        dims_raw = value.get("dimensions") or {}
        dimensions: dict[str, float] = {}
        if isinstance(dims_raw, dict):
            for k, v in dims_raw.items():
                try:
                    dimensions[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        feedback_text_raw = value.get("feedback_text", "")
        feedback_text = (
            feedback_text_raw if isinstance(feedback_text_raw, str) else ""
        )
        module_feedback_raw = value.get("module_feedback") or {}
        module_feedback: dict[str, str] = {}
        if isinstance(module_feedback_raw, dict):
            for k, v in module_feedback_raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    module_feedback[k] = v
        fk_raw = value.get("failure_kind")
        failure_kind = fk_raw if isinstance(fk_raw, str) and fk_raw else None
        return {
            "score": score,
            "dimensions": dimensions,
            "feedback_text": feedback_text,
            "module_feedback": module_feedback,
            "failure_kind": failure_kind,
        }
    return {
        "score": 0.0,
        "dimensions": {},
        "feedback_text": f"<grader returned unsupported type: {type(value).__name__}>",
        "module_feedback": {},
        "failure_kind": None,
    }


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
    feedback_corpus: list[dict[str, Any]],
    usd_used_in_run: float,
    aborted_due_to_budget: bool,
    failure_kind: str | None,
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
        "feedback_corpus": feedback_corpus,
        "usd_used_in_run": usd_used_in_run,
        "aborted_due_to_budget": aborted_due_to_budget,
        "failure_kind": failure_kind,
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
