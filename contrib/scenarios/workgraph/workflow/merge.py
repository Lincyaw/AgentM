"""WorkGraph merge workflow.

This module-mode workflow implements one merge scheduling pass over the
WorkGraph filesystem task bus. It claims verified tasks, serializes merges for
the same repo/base through a filesystem lock, and delegates all git/GitHub
operations to a short-lived merger agent running in ARL agent_env.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.lib import expand_path_from_cwd
from agentm.extensions.builtin.workflow import WorkflowContext

DEFAULT_STATE_DIR = ".agentm/workgraph"
DEFAULT_MERGER_SCENARIO = "workgraph/agents/merger"
TASK_HEADER_FIELDS = {"depends", "locks", "repo", "base"}
CONFIG_FILENAMES = ("config.toml", "workgraph.toml")


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


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _as_float(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _workflow_cwd(ctx: WorkflowContext) -> str:
    explicit = ctx.args.get("cwd")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    run = getattr(ctx, "_run", None)
    api = getattr(run, "api", None)
    api_cwd = getattr(api, "cwd", None)
    if isinstance(api_cwd, str) and api_cwd:
        return api_cwd
    return str(Path.cwd())


def _state_root(ctx: WorkflowContext) -> Path:
    raw = _as_str(ctx.args.get("state_dir"), DEFAULT_STATE_DIR)
    return expand_path_from_cwd(raw, _workflow_cwd(ctx))


def _config_path(ctx: WorkflowContext, root: Path) -> Path | None:
    raw = ctx.args.get("config") or ctx.args.get("config_path")
    if isinstance(raw, str) and raw.strip():
        return expand_path_from_cwd(raw, _workflow_cwd(ctx))
    for name in CONFIG_FILENAMES:
        path = root / name
        if path.is_file():
            return path
    return None


def _load_config(ctx: WorkflowContext, root: Path) -> dict[str, object]:
    path = _config_path(ctx, root)
    if path is None:
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"WorkGraph config {path} is not valid TOML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"WorkGraph config {path} must be a TOML table")
    return data


def _config_str(
    args: dict[str, object],
    config: dict[str, object],
    key: str,
    default: str = "",
) -> str:
    arg_value = args.get(key)
    if isinstance(arg_value, str) and arg_value.strip():
        return arg_value
    config_value = config.get(key)
    if isinstance(config_value, str) and config_value.strip():
        return config_value
    return default


def _config_int(
    args: dict[str, object],
    config: dict[str, object],
    keys: tuple[str, ...],
    default: int,
) -> int:
    for key in keys:
        if key in args:
            return _as_int(args.get(key), default)
    for key in keys:
        if key in config:
            return _as_int(config.get(key), default)
    return default


def _ensure_dirs(root: Path) -> None:
    for name in (
        "ready",
        "running",
        "verified",
        "merging",
        "merge_pending",
        "done",
        "failed",
        "conflicts",
        "locks",
        "results",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def _task_metadata(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            break
        if not stripped or stripped.startswith("# "):
            continue
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        normalized = key.strip().lower()
        if normalized in TASK_HEADER_FIELDS and normalized not in fields:
            fields[normalized] = value.strip()
    return fields


def _field(text: str, name: str) -> str:
    return _task_metadata(text).get(name.lower(), "")


def _csv_field(text: str, name: str) -> list[str]:
    raw = _field(text, name)
    if not raw or raw.lower() in {"none", "n/a", "-"}:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validation_commands(text: str) -> list[str]:
    lines = text.splitlines()
    commands: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lower() == "## validation"
            continue
        if not in_section:
            continue
        if stripped.startswith("- "):
            commands.append(stripped[2:].strip(" `"))
        elif stripped and not stripped.startswith("#"):
            commands.append(stripped.strip(" `"))
    return commands


def _task_id(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                return re.sub(r"\s+", "-", heading).strip("-")
    return path.stem


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


def _done_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in (root / "done").glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        ids.add(path.stem)
        ids.add(_task_id(path, text))
        for token in re.findall(r"\bREQ-[A-Za-z0-9_.-]+\b", text):
            ids.add(token)
    return ids


def _deps_satisfied(task: TaskFile, done: set[str]) -> bool:
    return all(dep in done for dep in task.depends)


def _safe_lock_name(lock: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", lock).strip("_") or "global"


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
            if not _deps_satisfied(task, done):
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


def _report_field(text: str, name: str) -> str:
    target = name.lower()
    for raw_line in text.splitlines():
        key, separator, value = raw_line.partition(":")
        if separator and key.strip().lower() == target:
            return value.strip()
    return ""


def _is_noneish(value: str) -> bool:
    return not value.strip() or value.strip().lower() in {"none", "null", "n/a", "-"}


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


def _agent_env_session_from_report(text: str) -> str:
    value = _report_field(text, "AgentEnvSession")
    return "" if _is_noneish(value) else value


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


def _config_env_var_names(config_env: dict[str, object]) -> set[str]:
    raw = config_env.get("envVars") or config_env.get("env_vars")
    if not isinstance(raw, list):
        return set()
    names: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _github_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    if not shutil.which("gh"):
        return ""
    try:
        completed = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _forward_git_env(config: dict[str, object]) -> None:
    token = _github_token()
    if not token:
        return

    raw_config_env = config.get("config_env")
    if raw_config_env is None:
        config_env: dict[str, object] = {}
    elif isinstance(raw_config_env, dict):
        config_env = dict(raw_config_env)
    else:
        raise ValueError("WorkGraph args.agent_env.config_env must be an object")

    raw_vars = config_env.get("vars")
    if raw_vars is None:
        vars_map: dict[str, object] = {}
    elif isinstance(raw_vars, dict):
        vars_map = dict(raw_vars)
    else:
        raise ValueError("WorkGraph args.agent_env.config_env.vars must be an object")

    existing = {str(name) for name in vars_map} | _config_env_var_names(config_env)
    for name, value in {
        "GH_TOKEN": os.environ.get("GH_TOKEN") or token,
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN") or token,
    }.items():
        if name not in existing:
            vars_map[name] = value

    config_env["vars"] = vars_map
    config["config_env"] = config_env


def _operations_config(args: dict[str, object]) -> dict[str, object]:
    return _operations_config_from_config(args, {})


def _operations_config_from_config(
    args: dict[str, object],
    config_root: dict[str, object],
) -> dict[str, object]:
    config: dict[str, object] = {}
    raw_default = config_root.get("agent_env")
    if raw_default is not None and not isinstance(raw_default, dict):
        raise ValueError("WorkGraph config [agent_env] must be an object")
    if isinstance(raw_default, dict):
        config.update(raw_default)

    raw = args.get("agent_env")
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("WorkGraph args.agent_env must be an object")
    if isinstance(raw, dict):
        config.update(raw)
    backend = config.get("backend", "agent_env")
    if backend != "agent_env":
        raise ValueError("WorkGraph merge agents require operations backend 'agent_env'")
    config["backend"] = "agent_env"
    config["delete_on_shutdown"] = False
    _forward_git_env(config)
    return config


def _agent_env_target_configured(operations: dict[str, object]) -> bool:
    return any(
        isinstance(value, str) and value.strip()
        for value in (
            operations.get("image"),
            os.environ.get("AGENTM_AGENT_ENV_IMAGE"),
        )
    )


def _shared_attach_session_configured(operations: dict[str, object]) -> bool:
    return any(
        isinstance(value, str) and value.strip()
        for value in (
            operations.get("attach_session"),
            os.environ.get("AGENTM_AGENT_ENV_ATTACH_SESSION"),
        )
    )


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
            "Rebase onto the latest base, push only the worker delivery branch "
            "with --force-with-lease after a successful rebase, run validation, "
            "and merge through gh. If source_queue is merge_pending, first "
            "check whether the PR is already merged; if it is still waiting on "
            "checks or branch protection, report Status: auto_merge again. "
            "Never print credentials."
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
            trace_label=f"{task.task_id}:merger",
        )
        merger_text = str(merger_result)
        merge_status = _merge_status(merger_text)
        agent_env_session = _agent_env_session_from_report(merger_text)

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
        }
    except Exception as exc:
        ctx.log(f"{task.task_id}: merge workflow failed: {type(exc).__name__}: {exc}")
        merger_text = f"Status: failed\n\nWorkflow exception: {type(exc).__name__}: {exc}"
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
    operations = _operations_config_from_config(ctx.args, config_root)
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
