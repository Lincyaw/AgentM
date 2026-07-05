"""WorkGraph development workflow.

This module-mode workflow implements one scheduling pass over a lightweight
filesystem task bus. The durable contract is intentionally small: task files
are Markdown, and only ``Depends``, ``Locks``, ``Repo``, ``Base``, and the
``## Validation`` section are interpreted by the scheduler. Worker agents own
git operations and report through their final response; the workflow records
those responses under the local state directory. Verified tasks move to
``verified/``; only the merge workflow moves tasks to ``done/`` after they
land in the base branch.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentm.extensions.builtin.workflow import WorkflowContext
from pydantic import BaseModel, Field

from .common import (
    _as_float,
    _as_str,
    _agent_env_session_from_report,
    _agent_env_target_configured,
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

DEFAULT_CODER_SCENARIO = "workgraph/agents/coder"
DEFAULT_VERIFIER_SCENARIO = "workgraph/agents/verifier"


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
class ClaimedTask:
    task: TaskFile
    running_path: Path
    lock_paths: list[Path]


class CoderReport(BaseModel):
    status: Literal["success", "failed", "conflict"] = Field(
        description="Coder delivery status."
    )
    agent_env_session: str = Field(
        default="",
        description="ARL agent_env session id to reuse for verifier calls, or none.",
    )
    branch: str = Field(default="", description="Local or remote delivery branch.")
    commit: str = Field(default="", description="Delivery commit sha.")
    remote: str = Field(
        default="",
        description="Pushed remote branch name or URL; required for success unless PR is set.",
    )
    pr: str = Field(default="", description="Pull request URL or number.")
    report: str = Field(
        description="Human-readable markdown report with changed files, validation, and notes."
    )


class VerifierReport(BaseModel):
    status: Literal["passed", "failed"] = Field(
        description="Independent verification status."
    )
    agent_env_session: str = Field(
        default="",
        description="ARL agent_env session id used for verification, or none.",
    )
    report: str = Field(
        description="Human-readable markdown report with commands and evidence."
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


def _acquire_locks(root: Path, task: TaskFile) -> list[Path] | None:
    acquired: list[Path] = []
    for lock in task.locks:
        path = root / "locks" / _safe_lock_name(lock)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(f"{task.task_id}\n{task.path.name}\n")
        except FileExistsError:
            for held in acquired:
                held.unlink(missing_ok=True)
            return None
        acquired.append(path)
    return acquired


def _release_locks(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _claim_tasks(
    root: Path,
    max_parallel: int,
    default_repo: str,
    default_base: str,
) -> list[ClaimedTask]:
    done = _done_ids(root)
    claimed: list[ClaimedTask] = []
    held_names: set[str] = set()
    for path in sorted((root / "ready").glob("*.md")):
        if len(claimed) >= max_parallel:
            break
        task = _read_task(path, default_repo, default_base)
        if not _deps_satisfied(task.depends, done):
            continue
        lock_names = {_safe_lock_name(lock) for lock in task.locks}
        if held_names & lock_names:
            continue
        lock_paths = _acquire_locks(root, task)
        if lock_paths is None:
            continue
        running_path = root / "running" / path.name
        if running_path.exists():
            _release_locks(lock_paths)
            continue
        path.rename(running_path)
        task.path = running_path
        claimed.append(
            ClaimedTask(task=task, running_path=running_path, lock_paths=lock_paths)
        )
        held_names.update(lock_names)
    return claimed


def _normalize_report(text: object) -> str:
    """Lowercase and strip markdown decorations so 'Status: x', '**Status:** x',
    '`Status`: x', and '- **Status**: x' all parse the same way."""
    return re.sub(r"[*_`]", "", str(text or "")).lower()


def _status(text: object) -> str:
    normalized = _normalize_report(text)
    for status in ("conflict", "failed", "passed", "success", "resolved"):
        if re.search(rf"status\s*:\s*{status}", normalized):
            return status
    if re.search(r"status\s*:\s*needs_human", normalized):
        return "needs_human"
    if "pull request" in normalized or "\npr:" in normalized:
        return "success"
    return "unknown"


def _coder_report_text(result: object) -> str:
    if isinstance(result, CoderReport):
        return _structured_report_text(
            [
                ("Status", result.status),
                ("AgentEnvSession", result.agent_env_session),
                ("Branch", result.branch),
                ("Commit", result.commit),
                ("Remote", result.remote),
                ("PR", result.pr),
            ],
            result.report,
        )
    return str(result)


def _verifier_report_text(result: object) -> str:
    if isinstance(result, VerifierReport):
        return _structured_report_text(
            [
                ("Status", result.status),
                ("AgentEnvSession", result.agent_env_session),
            ],
            result.report,
        )
    return str(result)


def _agent_env_session_from_result(result: object, text: str) -> str:
    if isinstance(result, (CoderReport, VerifierReport)):
        value = result.agent_env_session
    else:
        value = _agent_env_session_from_report(text)
    return "" if _is_noneish(value) else value.strip()


def _remote_delivery_present(text: str, *, remote: str = "", pr: str = "") -> bool:
    return not (
        _is_noneish(remote or _report_field(text, "remote"))
        and _is_noneish(pr or _report_field(text, "pr"))
    )


def _coerce_coder_status(
    coder_status: str,
    coder_text: str,
    *,
    remote: str = "",
    pr: str = "",
) -> tuple[str, str | None]:
    if coder_status not in {"success", "failed", "conflict"}:
        return (
            "failed",
            (
                f"Coder reported unrecognized status {coder_status or 'unknown'!r}; "
                "expected success, failed, or conflict."
            ),
        )
    if coder_status != "success":
        return coder_status, None
    if _remote_delivery_present(coder_text, remote=remote, pr=pr):
        return coder_status, None
    return (
        "failed",
        (
            "Coder reported success without a remote branch or PR. "
            "Sandbox-local commits are not a WorkGraph delivery."
        ),
    )


def _coder_summary(
    result: object,
    text: str,
    status: str,
) -> dict[str, object]:
    if isinstance(result, CoderReport):
        branch = result.branch
        commit = result.commit
        remote = result.remote
        pr = result.pr
    else:
        branch = _report_field(text, "Branch")
        commit = _report_field(text, "Commit")
        remote = _report_field(text, "Remote")
        pr = _report_field(text, "PR")
    return {
        "status": status,
        "branch": _report_optional_value(branch),
        "commit": _report_optional_value(commit),
        "remote": _report_optional_value(remote),
        "pr": _report_optional_value(pr),
    }


def _verifier_summary(
    result: object,
    text: str,
    status: str,
) -> dict[str, object]:
    if isinstance(result, VerifierReport):
        agent_env_session = result.agent_env_session
    else:
        agent_env_session = _agent_env_session_from_report(text)
    return {
        "status": status,
        "agent_env_session": _report_optional_value(agent_env_session),
    }


def _result_dir(root: Path, task: TaskFile) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", task.task_id).strip("-")
    path = root / "results" / (safe or task.path.stem)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_result(
    root: Path,
    task: TaskFile,
    coder: str,
    verifier: str,
    agent_env_session: str = "",
) -> None:
    path = _result_dir(root, task)
    (path / "task.md").write_text(task.text, encoding="utf-8")
    (path / "result.md").write_text(coder, encoding="utf-8")
    (path / "validation.md").write_text(verifier, encoding="utf-8")
    session_file = path / "agent_env_session.txt"
    if agent_env_session:
        session_file.write_text(
            f"{agent_env_session}\n",
            encoding="utf-8",
        )
    else:
        session_file.unlink(missing_ok=True)


def _move_finished(root: Path, claim: ClaimedTask, status: str) -> Path:
    if status in {"verified", "passed"}:
        target_dir = root / "verified"
    elif status == "conflict":
        target_dir = root / "conflicts"
    else:
        target_dir = root / "failed"
    target = target_dir / claim.running_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(claim.running_path), str(target))
    return target


def _agent_env_session_file(root: Path, task: TaskFile) -> Path:
    return _result_dir(root, task) / "agent_env_session.txt"


def _read_agent_env_session(root: Path, task: TaskFile) -> str:
    path = _agent_env_session_file(root, task)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _operations_for_session(
    operations: dict[str, object],
    agent_env_session: str,
) -> dict[str, object]:
    config = dict(operations)
    if agent_env_session:
        config["attach_session"] = agent_env_session
    return config


def _context_for_task(
    task: TaskFile,
    role: str,
    extra: dict[str, object] | None = None,
    *,
    devbox: bool = False,
) -> dict[str, object]:
    if devbox:
        task_dir = f"/workspace/tasks/{task.task_id}"
        execution = (
            "Run inside a SHARED long-lived ARL devbox sandbox; other tasks "
            "run here concurrently. Your isolated workspace for this task is "
            f"{task_dir} — never read or write any other /workspace/tasks/ "
            "directory, and never work outside your workspace except shared "
            "caches. If your workspace already exists from a previous "
            "attempt, inspect its state (git status/log) and CONTINUE from "
            "that progress instead of restarting. Otherwise clone the "
            "repository there. Shared build caches (~/.m2, ~/.cargo, pip and "
            "npm caches) are warm — reuse them; do not clear them. "
            "Do not use the host/control repository as the worktree. Include "
            "AgentEnvSession in the final response so this workflow can "
            "reuse the sandbox for follow-up agent calls in the same task. "
            "Use repository credentials only when the scenario, image, or "
            "ARL config_env provides them inside the sandbox, and never "
            "print those credentials. A coder success requires Remote or PR "
            "to be non-empty; a sandbox-local commit without a pushed branch "
            "is a failed delivery."
        )
    else:
        execution = (
            "Run inside the ARL agent_env sandbox. Do not use the host/control "
            "repository as the worktree. Include AgentEnvSession in the final "
            "response so this workflow can reuse the sandbox for follow-up "
            "agent calls in the same task. Use repository credentials only "
            "when the scenario, image, or ARL config_env provides them inside "
            "the sandbox, and never print those credentials. A coder success "
            "requires Remote or PR to be non-empty; a sandbox-local commit "
            "without a pushed branch is a failed delivery."
        )
    context: dict[str, object] = {
        "role": role,
        "task_id": task.task_id,
        "task_file": task.path.name,
        "task": task.text,
        "repo": task.repo,
        "base": task.base,
        "locks": task.locks,
        "validation": task.validation,
        "execution": execution,
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
    claim: ClaimedTask,
    operations: dict[str, object],
) -> dict[str, object]:
    task = claim.task
    coder_scenario = _as_str(ctx.args.get("coder_scenario"), DEFAULT_CODER_SCENARIO)
    verifier_scenario = _as_str(
        ctx.args.get("verifier_scenario"),
        DEFAULT_VERIFIER_SCENARIO,
    )
    timeout = _as_float(ctx.args.get("agent_timeout_seconds"), 7200.0)
    devbox = bool(operations.get("attach_session"))
    agent_env_session = ""

    try:
        ctx.log(f"coding {task.task_id}")
        coder_result = await ctx.agent(
            f"Implement WorkGraph task {task.task_id}.",
            scenario=coder_scenario,
            atom_config=_atom_config(
                _operations_for_session(operations, agent_env_session),
                _context_for_task(task, "coder", devbox=devbox),
            ),
            timeout=timeout,
            schema=CoderReport,
            retry=1,
            trace_label=f"{task.task_id}:coder",
        )
        coder_text = _coder_report_text(coder_result)
        coder_status: str
        if isinstance(coder_result, CoderReport):
            coder_status = coder_result.status
            coder_remote = coder_result.remote
            coder_pr = coder_result.pr
        else:
            coder_status = _status(coder_text)
            coder_remote = _report_field(coder_text, "Remote")
            coder_pr = _report_field(coder_text, "PR")
        coder_status, delivery_error = _coerce_coder_status(
            coder_status,
            coder_text,
            remote=coder_remote,
            pr=coder_pr,
        )
        agent_env_session = (
            _agent_env_session_from_result(coder_result, coder_text)
            or agent_env_session
        )
        verifier_status = "failed"
        verifier_result: object | None = None

        if coder_status == "conflict":
            verifier_text = (
                "Status: failed\n\nCoder reported a conflict before verification."
            )
            final_status = "conflict"
        elif coder_status == "failed":
            reason = delivery_error or "Coder reported failure before verification."
            verifier_text = f"Status: failed\n\n{reason}"
            final_status = "failed"
        else:
            ctx.log(f"verifying {task.task_id}")
            verifier_result = await ctx.agent(
                f"Verify WorkGraph task {task.task_id}.",
                scenario=verifier_scenario,
                atom_config=_atom_config(
                    _operations_for_session(operations, agent_env_session),
                    _context_for_task(
                        task,
                        "verifier",
                        {
                            "coder_result": coder_text,
                            "agent_env_session": agent_env_session,
                        },
                        devbox=devbox,
                    ),
                ),
                timeout=timeout,
                schema=VerifierReport,
                retry=1,
                trace_label=f"{task.task_id}:verifier",
            )
            verifier_text = _verifier_report_text(verifier_result)
            agent_env_session = (
                _agent_env_session_from_result(verifier_result, verifier_text)
                or agent_env_session
            )
            verifier_status = (
                verifier_result.status
                if isinstance(verifier_result, VerifierReport)
                else _status(verifier_text)
            )
            if verifier_status == "passed" and re.search(
                r"^\s*[-*]?\s*\[blocker\]", verifier_text, re.IGNORECASE | re.MULTILINE
            ):
                # Verdict discipline enforced in code: the verifier prompt says
                # blockers MUST fail the delivery, but models have reported
                # passed alongside blocker findings. The finding list wins.
                ctx.log(
                    f"{task.task_id}: verifier reported passed but listed "
                    "[blocker] findings; downgrading to failed"
                )
                verifier_text = (
                    "Status: failed (downgraded by workflow: [blocker] findings "
                    "present despite passed verdict)\n\n" + verifier_text
                )
                verifier_status = "failed"
            final_status = "verified" if verifier_status == "passed" else "failed"

        _write_result(root, task, coder_text, verifier_text, agent_env_session)
        target = _move_finished(root, claim, final_status)
        return {
            "task_id": task.task_id,
            "status": final_status,
            "from": str(claim.running_path),
            "to": str(target),
            "coder_status": coder_status,
            "agent_env_session": agent_env_session or None,
            "result_dir": str(_result_dir(root, task)),
            "coder": _coder_summary(coder_result, coder_text, coder_status),
            "verifier": _verifier_summary(
                verifier_result,
                verifier_text,
                verifier_status,
            ),
        }
    except Exception as exc:
        ctx.log(
            f"{task.task_id}: development workflow failed: {type(exc).__name__}: {exc}"
        )
        coder_text = (
            f"Status: failed\n\nWorkflow exception: {type(exc).__name__}: {exc}"
        )
        verifier_text = "Status: failed\n\nVerifier was not completed."
        _write_result(root, task, coder_text, verifier_text, agent_env_session)
        target = _move_finished(root, claim, "failed")
        return {
            "task_id": task.task_id,
            "status": "failed",
            "from": str(claim.running_path),
            "to": str(target),
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
        _config_int(ctx.args, config_root, ("develop_max_parallel", "max_parallel"), 1),
    )
    default_repo = _config_str(ctx.args, config_root, "repo")
    default_base = _config_str(ctx.args, config_root, "base", "main")
    operations = _operations_config_from_config(
        ctx.args,
        config_root,
        backend_error="WorkGraph development workers require operations backend 'agent_env'",
    )
    devbox_session = str(operations.pop("devbox_session", "") or "").strip()
    if devbox_session:
        # Devbox mode: all agents attach to one long-lived shared sandbox
        # (no per-hour lifetime, warm build caches); tasks isolate via
        # per-task workspaces under /workspace/tasks/<task-id>.
        operations["attach_session"] = devbox_session
    elif _shared_attach_session_configured(operations):
        raise RuntimeError(
            "WorkGraph develop does not accept a shared agent_env attach_session "
            "for automatic task claiming. Use agent_env.devbox_session for the "
            "shared-devbox mode, or args.agent_env.image / "
            "AGENTM_AGENT_ENV_IMAGE so each claimed task gets its own ARL "
            "sandbox; the workflow only attaches follow-up agents to a "
            "per-task sandbox recorded under results/<task-id>/."
        )
    if not devbox_session and not _agent_env_target_configured(operations):
        raise RuntimeError(
            "WorkGraph development workers require an ARL agent_env target. "
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

    ctx.log(f"claimed {len(claimed)} task(s)")
    ctx.phase("develop")
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
            if claim.running_path.exists()
            else root / "failed" / claim.running_path.name
        )
        results.append(
            {
                "task_id": claim.task.task_id,
                "status": "failed",
                "from": str(claim.running_path),
                "to": str(target),
                "error": "workflow parallel item failed",
            }
        )

    return {
        "status": "complete",
        "state_dir": str(root),
        "claimed": len(claimed),
        "results": results,
    }
