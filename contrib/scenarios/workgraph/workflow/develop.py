"""WorkGraph development workflow.

This module-mode workflow implements one scheduling pass over a lightweight
filesystem task bus. The durable contract is intentionally small: task files
are Markdown, and only ``Depends``, ``Locks``, ``Repo``, ``Base``, and the
``## Validation`` section are interpreted by the scheduler. Worker agents
report exclusively through structured ``submit_result`` payloads
(:class:`CoderReport` / :class:`VerifierReport`); a worker that returns
anything else is recorded as failed with the raw payload as evidence. The
workflow records reports under the local state directory. Verified tasks move
to ``verified/``; only the merge workflow moves tasks to ``done/`` after they
land in the base branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentm.extensions.builtin.workflow import WorkflowContext
from pydantic import BaseModel, Field

from .common import (
    ExecMode,
    TaskFile,
    _as_float,
    _as_str,
    _atom_config,
    _config_int,
    _config_str,
    _deps_satisfied,
    _done_ids,
    _ensure_dirs,
    _is_noneish,
    _load_config,
    _move_task,
    _operations_config_from_config,
    _read_task,
    _release_locks,
    _report_optional_value,
    _resolve_exec_mode,
    _result_dir,
    _review_standards,
    _safe_lock_name,
    _state_root,
    _structured_report_text,
    _try_acquire_lock_file,
)

DEFAULT_CODER_SCENARIO = "workgraph/agents/coder"
DEFAULT_VERIFIER_SCENARIO = "workgraph/agents/verifier"


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


class VerifierFinding(BaseModel):
    severity: Literal["blocker", "major", "nit"] = Field(
        description="Finding severity; any blocker fails the delivery."
    )
    location: str = Field(default="", description="file:line or component.")
    finding: str = Field(description="One-line description of the finding.")


class VerifierReport(BaseModel):
    status: Literal["passed", "failed"] = Field(
        description="Independent verification status."
    )
    agent_env_session: str = Field(
        default="",
        description="ARL agent_env session id used for verification, or none.",
    )
    findings: list[VerifierFinding] = Field(
        default_factory=list,
        description="Review findings; the workflow derives failure from blockers.",
    )
    report: str = Field(
        description="Human-readable markdown report with commands and evidence."
    )


def _acquire_locks(root: Path, task: TaskFile) -> list[Path] | None:
    acquired: list[Path] = []
    for lock in task.locks:
        path = root / "locks" / _safe_lock_name(lock)
        if not _try_acquire_lock_file(path, task.task_id, task.path.name):
            for held in acquired:
                held.unlink(missing_ok=True)
            return None
        acquired.append(path)
    return acquired


async def _run_one_then_release(
    ctx: WorkflowContext,
    root: Path,
    claim: ClaimedTask,
    operations: dict[str, object],
    mode: ExecMode,
    review_standards: str,
) -> dict[str, object]:
    # Release this task's locks as soon as IT finishes; holding them until
    # the whole claimed batch drains starves retries of failed siblings.
    try:
        return await _run_one(ctx, root, claim, operations, mode, review_standards)
    finally:
        _release_locks(claim.lock_paths)


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


def _coder_report_text(report: CoderReport) -> str:
    return _structured_report_text(
        [
            ("Status", report.status),
            ("AgentEnvSession", report.agent_env_session),
            ("Branch", report.branch),
            ("Commit", report.commit),
            ("Remote", report.remote),
            ("PR", report.pr),
        ],
        report.report,
    )


def _findings_section(findings: list[VerifierFinding]) -> str:
    if not findings:
        return "## Review\n\n- no findings"
    lines = "\n".join(
        f"- [{f.severity}]"
        + (f" {f.location}" if f.location else "")
        + f" — {f.finding}"
        for f in findings
    )
    return f"## Review\n\n{lines}"


def _verifier_report_text(report: VerifierReport) -> str:
    body = f"{report.report}\n\n{_findings_section(report.findings)}"
    return _structured_report_text(
        [
            ("Status", report.status),
            ("AgentEnvSession", report.agent_env_session),
        ],
        body,
    )


def _session_from(report: CoderReport | VerifierReport) -> str:
    value = report.agent_env_session
    return "" if _is_noneish(value) else value.strip()


def _coerce_coder_status(report: CoderReport) -> tuple[str, str | None]:
    """A coder success requires a remote delivery (pushed branch or PR)."""
    if report.status != "success":
        return report.status, None
    if _is_noneish(report.remote) and _is_noneish(report.pr):
        return (
            "failed",
            (
                "Coder reported success without a remote branch or PR. "
                "Sandbox-local commits are not a WorkGraph delivery."
            ),
        )
    return report.status, None


_LEGACY_BLOCKER_RE = re.compile(
    r"^\s*[-*]?\s*\[blocker\]", re.IGNORECASE | re.MULTILINE
)


def _effective_verifier_status(report: VerifierReport) -> tuple[str, str | None]:
    """Derive the verdict from findings; any blocker fails the delivery.

    The structured ``findings`` list is authoritative. The legacy text scan
    remains as a guard for models that still write ``[blocker]`` lines into
    the free-text report instead of the findings field.
    """
    if report.status != "passed":
        return report.status, None
    if any(f.severity == "blocker" for f in report.findings):
        return "failed", "blocker findings present despite passed verdict"
    if _LEGACY_BLOCKER_RE.search(report.report):
        return "failed", "[blocker] lines in report despite passed verdict"
    return "passed", None


def _coder_summary(report: CoderReport | None, status: str) -> dict[str, object]:
    if report is None:
        return {"status": status, "branch": None, "commit": None,
                "remote": None, "pr": None}
    return {
        "status": status,
        "branch": _report_optional_value(report.branch),
        "commit": _report_optional_value(report.commit),
        "remote": _report_optional_value(report.remote),
        "pr": _report_optional_value(report.pr),
    }


def _verifier_summary(report: VerifierReport | None, status: str) -> dict[str, object]:
    session = _session_from(report) if report is not None else ""
    return {
        "status": status,
        "agent_env_session": session or None,
    }


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
    return _move_task(claim.running_path, target_dir)


def _operations_for_session(
    operations: dict[str, object],
    agent_env_session: str,
) -> dict[str, object]:
    config = dict(operations)
    if agent_env_session:
        config["attach_session"] = agent_env_session
    return config


_EXECUTION_COMMON = (
    "Do not use the host/control repository as the worktree. Include "
    "AgentEnvSession in the final response so this workflow can reuse the "
    "sandbox for follow-up agent calls in the same task. Use repository "
    "credentials only when the scenario, image, or ARL config_env provides "
    "them inside the sandbox, and never print those credentials. A coder "
    "success requires Remote or PR to be non-empty; a sandbox-local commit "
    "without a pushed branch is a failed delivery."
)


def _execution_text(mode: ExecMode, task: TaskFile) -> str:
    if not mode.devbox:
        return f"Run inside the ARL agent_env sandbox. {_EXECUTION_COMMON}"
    task_dir = f"/workspace/tasks/{task.task_id}"
    devbox_prefix = (
        "Run inside a SHARED long-lived ARL devbox sandbox; other tasks "
        "run here concurrently. Your isolated workspace for this task is "
        f"{task_dir} — never read or write any other /workspace/tasks/ "
        "directory, and never work outside your workspace except shared "
        "caches. If your workspace already exists from a previous "
        "attempt, inspect its state (git status/log) and CONTINUE from "
        "that progress instead of restarting. Otherwise clone the "
        "repository there. Shared build caches (~/.m2, ~/.cargo, pip and "
        "npm caches) are warm — reuse them; do not clear them."
    )
    return f"{devbox_prefix} {_EXECUTION_COMMON}"


def _context_for_task(
    task: TaskFile,
    role: str,
    mode: ExecMode,
    review_standards: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "role": role,
        "task_id": task.task_id,
        "task_file": task.path.name,
        "task": task.text,
        "repo": task.repo,
        "base": task.base,
        "locks": task.locks,
        "validation": task.validation,
        "execution": _execution_text(mode, task),
    }
    if review_standards:
        context["review_standards"] = review_standards
    if extra:
        context.update(extra)
    return context


async def _run_one(
    ctx: WorkflowContext,
    root: Path,
    claim: ClaimedTask,
    operations: dict[str, object],
    mode: ExecMode,
    review_standards: str,
) -> dict[str, object]:
    task = claim.task
    coder_scenario = _as_str(ctx.args.get("coder_scenario"), DEFAULT_CODER_SCENARIO)
    verifier_scenario = _as_str(
        ctx.args.get("verifier_scenario"),
        DEFAULT_VERIFIER_SCENARIO,
    )
    timeout = _as_float(ctx.args.get("agent_timeout_seconds"), 7200.0)
    agent_env_session = ""

    try:
        ctx.log(f"coding {task.task_id}")
        coder_result = await ctx.agent(
            f"Implement WorkGraph task {task.task_id}.",
            scenario=coder_scenario,
            atom_config=_atom_config(
                _operations_for_session(operations, agent_env_session),
                _context_for_task(task, "coder", mode, review_standards),
            ),
            timeout=timeout,
            schema=CoderReport,
            retry=1,
            trace_label=f"{task.task_id}:coder",
        )
        coder_report = coder_result if isinstance(coder_result, CoderReport) else None
        if coder_report is not None:
            coder_text = _coder_report_text(coder_report)
            coder_status, delivery_error = _coerce_coder_status(coder_report)
            agent_env_session = _session_from(coder_report)
        else:
            coder_text = (
                "Status: failed\n\nCoder returned no structured report; raw "
                f"result: {coder_result!r}"
            )
            coder_status, delivery_error = (
                "failed",
                "Coder returned no structured report.",
            )
        verifier_status = "failed"
        verifier_report: VerifierReport | None = None

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
                        mode,
                        review_standards,
                        {
                            "coder_result": coder_text,
                            "agent_env_session": agent_env_session,
                        },
                    ),
                ),
                timeout=timeout,
                schema=VerifierReport,
                retry=1,
                trace_label=f"{task.task_id}:verifier",
            )
            if isinstance(verifier_result, VerifierReport):
                verifier_report = verifier_result
                verifier_text = _verifier_report_text(verifier_report)
                agent_env_session = _session_from(verifier_report) or agent_env_session
                verifier_status, downgrade_reason = _effective_verifier_status(
                    verifier_report
                )
                if downgrade_reason is not None:
                    ctx.log(
                        f"{task.task_id}: verifier verdict downgraded to failed "
                        f"({downgrade_reason})"
                    )
                    verifier_text = (
                        f"Status: failed (downgraded by workflow: {downgrade_reason})"
                        f"\n\n{verifier_text}"
                    )
            else:
                verifier_text = (
                    "Status: failed\n\nVerifier returned no structured report; "
                    f"raw result: {verifier_result!r}"
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
            "coder": _coder_summary(coder_report, coder_status),
            "verifier": _verifier_summary(verifier_report, verifier_status),
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
    mode = _resolve_exec_mode(operations, workflow="develop")
    review_standards = _review_standards(ctx, root, config_root)

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
            [
                _run_one_then_release(
                    ctx, root, claim, operations, mode, review_standards
                )
                for claim in claimed
            ]
        )
    finally:
        # Safety net for claims whose _run_one_then_release never ran
        # (e.g. the parallel gather aborted before scheduling them);
        # _release_locks is owner-checked and idempotent.
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
