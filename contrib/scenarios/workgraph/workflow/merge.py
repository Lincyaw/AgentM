"""WorkGraph merge workflow.

This module-mode workflow implements one merge scheduling pass over the
WorkGraph filesystem task bus. It claims verified tasks, serializes merges for
the same repo/base through a filesystem lock, and delegates all git/GitHub
operations to a short-lived merger agent running in ARL agent_env.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentm.extensions.builtin.workflow import WorkflowContext
from pydantic import BaseModel, Field

from .common import (
    _agent_env_session_from_report,
    _agent_env_target_configured,
    _as_float,
    _as_str,
    _config_int,
    _config_str,
    _csv_field,
    _deps_satisfied,
    _done_ids,
    _ensure_dirs,
    _field,
    _is_noneish,
    _load_config,
    _operations_config_from_config,
    _report_field,
    _report_optional_value,
    _safe_lock_name,
    _shared_attach_session_configured,
    _state_root,
    _structured_report_text,
    _task_id,
    _validation_commands,
)

DEFAULT_MERGER_SCENARIO = "workgraph/agents/merger"


@dataclass(slots=True)
class TaskFile:
    task_id: str
    path: Path
    text: str
    depends: list[str]
    locks: list[str]
    repo: str
    base: str
    validation: list[str]


@dataclass(slots=True)
class ClaimedMergeTask:
    task: TaskFile
    merging_path: Path
    source_queue: str
    lock_paths: list[Path]


class MergeReport(BaseModel):
    status: Literal[
        "merged",
        "auto_merge",
        "failed",
        "conflict",
        "needs_human",
        "pending",
    ] = Field(description="Merge outcome for the verified delivery.")
    agent_env_session: str = Field(
        default="",
        description="ARL agent_env session id used for merge operations, or none.",
    )
    pr: str = Field(default="", description="Pull request URL or number.")
    branch: str = Field(default="", description="Delivery branch merged or pending.")
    merged_commit: str = Field(
        default="",
        description="Commit sha visible on the remote base branch after merge, or none.",
    )
    report: str = Field(
        description="Human-readable markdown report with commands, blockers, and evidence."
    )


def _read_task(path: Path, default_repo: str, default_base: str) -> TaskFile:
    text = path.read_text(encoding="utf-8")
    return TaskFile(
        task_id=_task_id(path, text),
        path=path,
        text=text,
        depends=_csv_field(text, "Depends"),
        locks=_csv_field(text, "Locks"),
        repo=_field(text, "Repo") or default_repo,
        base=_field(text, "Base") or default_base,
        validation=_validation_commands(text),
    )


def _merge_lock_name(task: TaskFile) -> str:
    digest = hashlib.sha256(f"{task.repo}\n{task.base}".encode()).hexdigest()[:16]
    base = _safe_lock_name(task.base)[:60] or "base"
    return f"merge_{base}_{digest}"


def _acquire_merge_lock(root: Path, task: TaskFile) -> list[Path] | None:
    path = root / "locks" / _merge_lock_name(task)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(f"{task.task_id}\n{task.path.name}\n")
    except FileExistsError:
        return None
    return [path]


def _release_locks(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _claim_tasks(
    root: Path,
    max_parallel: int,
    default_repo: str,
    default_base: str,
) -> list[ClaimedMergeTask]:
    done = _done_ids(root)
    claimed: list[ClaimedMergeTask] = []
    held_names: set[str] = set()
    for queue_name in ("merge_pending", "verified"):
        for path in sorted((root / queue_name).glob("*.md")):
            if len(claimed) >= max_parallel:
                break
            task = _read_task(path, default_repo, default_base)
            if not _deps_satisfied(task.depends, done):
                continue
            lock_name = _merge_lock_name(task)
            if lock_name in held_names:
                continue
            lock_paths = _acquire_merge_lock(root, task)
            if lock_paths is None:
                continue
            merging_path = root / "merging" / path.name
            if merging_path.exists():
                _release_locks(lock_paths)
                continue
            path.rename(merging_path)
            task.path = merging_path
            claimed.append(
                ClaimedMergeTask(
                    task=task,
                    merging_path=merging_path,
                    source_queue=queue_name,
                    lock_paths=lock_paths,
                )
            )
            held_names.add(lock_name)
        if len(claimed) >= max_parallel:
            break
    return claimed


def _merge_status(text: object) -> str:
    lower = str(text or "").lower()
    for status in ("merged", "failed", "conflict", "needs_human", "pending"):
        if f"status: {status}" in lower:
            return status
    if "status: auto_merge" in lower or "status: auto-merge" in lower:
        return "auto_merge"
    return "unknown"


def _normalize_merge_status(status: str) -> str:
    if status in {
        "merged",
        "auto_merge",
        "pending",
        "failed",
        "conflict",
        "needs_human",
    }:
        return status
    return "failed"


def _merge_report_text(result: object) -> str:
    if isinstance(result, MergeReport):
        return _structured_report_text(
            [
                ("Status", result.status),
                ("AgentEnvSession", result.agent_env_session),
                ("PR", result.pr),
                ("Branch", result.branch),
                ("MergedCommit", result.merged_commit),
            ],
            result.report,
        )
    return str(result)


def _agent_env_session_from_result(result: object, text: str) -> str:
    if isinstance(result, MergeReport):
        value = result.agent_env_session
    else:
        value = _agent_env_session_from_report(text)
    return "" if _is_noneish(value) else value.strip()


def _merge_summary(
    result: object,
    text: str,
    status: str,
) -> dict[str, object]:
    if isinstance(result, MergeReport):
        pr = result.pr
        branch = result.branch
        merged_commit = result.merged_commit
        agent_env_session = result.agent_env_session
    else:
        pr = _report_field(text, "PR")
        branch = _report_field(text, "Branch")
        merged_commit = _report_field(text, "MergedCommit")
        agent_env_session = _agent_env_session_from_report(text)
    return {
        "status": status,
        "pr": _report_optional_value(pr),
        "branch": _report_optional_value(branch),
        "merged_commit": _report_optional_value(merged_commit),
        "agent_env_session": _report_optional_value(agent_env_session),
    }


def _result_dir(root: Path, task: TaskFile) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", task.task_id).strip("-")
    path = root / "results" / (safe or task.path.stem)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_result_file(root: Path, task: TaskFile, name: str) -> str:
    path = _result_dir(root, task) / name
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _write_merge_result(
    root: Path,
    task: TaskFile,
    merger: str,
    agent_env_session: str = "",
) -> None:
    path = _result_dir(root, task)
    (path / "task.md").write_text(task.text, encoding="utf-8")
    (path / "merge.md").write_text(merger, encoding="utf-8")
    session_file = path / "merge_agent_env_session.txt"
    if agent_env_session:
        session_file.write_text(
            f"{agent_env_session}\n",
            encoding="utf-8",
        )
    else:
        session_file.unlink(missing_ok=True)


def _move_finished(root: Path, claim: ClaimedMergeTask, status: str) -> Path:
    if status == "merged":
        target_dir = root / "done"
    elif status in {"auto_merge", "pending"}:
        target_dir = root / "merge_pending"
    elif status in {"conflict", "needs_human"}:
        target_dir = root / "conflicts"
    else:
        target_dir = root / "failed"
    target = target_dir / claim.merging_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(claim.merging_path), str(target))
    return target


def _context_for_task(
    root: Path,
    claim: ClaimedMergeTask,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    task = claim.task
    coder_result = _read_result_file(root, task, "result.md")
    verifier_result = _read_result_file(root, task, "validation.md")
    branch = _report_field(coder_result, "Branch")
    pr = _report_field(coder_result, "PR") or _report_field(verifier_result, "PR")
    remote = _report_field(coder_result, "Remote")
    context: dict[str, object] = {
        "role": "merger",
        "task_id": task.task_id,
        "task_file": task.path.name,
        "source_queue": claim.source_queue,
        "task": task.text,
        "repo": task.repo,
        "base": task.base,
        "locks": task.locks,
        "validation": task.validation,
        "coder_result": coder_result,
        "verifier_result": verifier_result,
        "branch": "" if _is_noneish(branch) else branch,
        "pr": "" if _is_noneish(pr) else pr,
        "remote": "" if _is_noneish(remote) else remote,
        "execution": (
            "Run inside the ARL agent_env sandbox. Do not use the host/control "
            "repository as the worktree. Perform GitHub and git operations for "
            "exactly one verified delivery branch or PR. If no PR exists yet, "
            "create it from the delivery branch before merge operations. "
            "Fetch the latest remote base, rebase the delivery branch onto "
            "origin/<base>, immediately push only the worker delivery branch "
            "with --force-with-lease after a successful rebase, verify the "
            "remote delivery branch was updated, run validation, and merge "
            "through gh with rebase semantics. After a successful merge, fetch "
            "origin/<base> immediately and report MergedCommit as the remote "
            "base HEAD that contains the delivery. If source_queue is "
            "merge_pending, first check whether the PR is already merged; if "
            "it is still waiting on checks or branch protection, report "
            "Status: auto_merge again. Never print credentials."
        ),
    }
    if extra:
        context.update(extra)
    return context


def _atom_config(
    operations: dict[str, object],
    context: dict[str, object],
) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {"workgraph_context": dict(context)}
    if operations:
        config["operations"] = dict(operations)
    return config


async def _run_one(
    ctx: WorkflowContext,
    root: Path,
    claim: ClaimedMergeTask,
    operations: dict[str, object],
) -> dict[str, object]:
    task = claim.task
    merger_scenario = _as_str(ctx.args.get("merger_scenario"), DEFAULT_MERGER_SCENARIO)
    timeout = _as_float(ctx.args.get("agent_timeout_seconds"), 3600.0)
    agent_env_session = ""

    try:
        ctx.log(f"merging {task.task_id}")
        merger_result = await ctx.agent(
            f"Merge WorkGraph task {task.task_id}.",
            scenario=merger_scenario,
            atom_config=_atom_config(
                operations,
                _context_for_task(root, claim),
            ),
            timeout=timeout,
            schema=MergeReport,
            retry=1,
            trace_label=f"{task.task_id}:merger",
        )
        merger_text = _merge_report_text(merger_result)
        merge_status_raw: str = (
            merger_result.status
            if isinstance(merger_result, MergeReport)
            else _merge_status(merger_text)
        )
        merge_status = _normalize_merge_status(merge_status_raw)
        agent_env_session = _agent_env_session_from_result(
            merger_result,
            merger_text,
        )

        _write_merge_result(root, task, merger_text, agent_env_session)
        target = _move_finished(root, claim, merge_status)
        return {
            "task_id": task.task_id,
            "status": merge_status,
            "from": str(claim.merging_path),
            "to": str(target),
            "source_queue": claim.source_queue,
            "agent_env_session": agent_env_session or None,
            "result_dir": str(_result_dir(root, task)),
            "merge": _merge_summary(merger_result, merger_text, merge_status),
        }
    except Exception as exc:
        ctx.log(f"{task.task_id}: merge workflow failed: {type(exc).__name__}: {exc}")
        merger_text = (
            f"Status: failed\n\nWorkflow exception: {type(exc).__name__}: {exc}"
        )
        _write_merge_result(root, task, merger_text, agent_env_session)
        target = _move_finished(root, claim, "failed")
        return {
            "task_id": task.task_id,
            "status": "failed",
            "from": str(claim.merging_path),
            "to": str(target),
            "source_queue": claim.source_queue,
            "error": f"{type(exc).__name__}: {exc}",
            "agent_env_session": agent_env_session or None,
            "result_dir": str(_result_dir(root, task)),
        }


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    root = _state_root(ctx)
    _ensure_dirs(root)
    config_root = _load_config(ctx, root)
    max_parallel = max(
        1,
        _config_int(ctx.args, config_root, ("merge_max_parallel", "max_parallel"), 1),
    )
    default_repo = _config_str(ctx.args, config_root, "repo")
    default_base = _config_str(ctx.args, config_root, "base", "main")
    operations = _operations_config_from_config(
        ctx.args,
        config_root,
        backend_error="WorkGraph merge agents require operations backend 'agent_env'",
    )
    if _shared_attach_session_configured(operations):
        raise RuntimeError(
            "WorkGraph merge does not accept a shared agent_env attach_session "
            "for automatic task claiming. Use args.agent_env.image or "
            "AGENTM_AGENT_ENV_IMAGE so each claimed merge gets its own ARL "
            "sandbox."
        )
    if not _agent_env_target_configured(operations):
        raise RuntimeError(
            "WorkGraph merge agents require an ARL agent_env target. "
            "Pass args.agent_env.image or set AGENTM_AGENT_ENV_IMAGE."
        )

    ctx.phase("claim")
    claimed = _claim_tasks(root, max_parallel, default_repo, default_base)
    if not claimed:
        return {
            "status": "idle",
            "state_dir": str(root),
            "claimed": 0,
            "results": [],
        }

    ctx.log(f"claimed {len(claimed)} merge task(s)")
    ctx.phase("merge")
    try:
        raw_results = await ctx.parallel(
            [_run_one(ctx, root, claim, operations) for claim in claimed]
        )
    finally:
        for claim in claimed:
            _release_locks(claim.lock_paths)

    results: list[dict[str, object]] = []
    for claim, result in zip(claimed, raw_results, strict=True):
        if result is not None:
            results.append(result)
            continue
        target = (
            _move_finished(root, claim, "failed")
            if claim.merging_path.exists()
            else root / "failed" / claim.merging_path.name
        )
        results.append(
            {
                "task_id": claim.task.task_id,
                "status": "failed",
                "from": str(claim.merging_path),
                "to": str(target),
                "source_queue": claim.source_queue,
                "error": "workflow parallel item failed",
            }
        )

    return {
        "status": "complete",
        "state_dir": str(root),
        "claimed": len(claimed),
        "results": results,
    }
