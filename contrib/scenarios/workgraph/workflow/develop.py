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

import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext

DEFAULT_STATE_DIR = ".agentm/workgraph"
DEFAULT_CODER_SCENARIO = "workgraph/agents/coder"
DEFAULT_VERIFIER_SCENARIO = "workgraph/agents/verifier"
TASK_HEADER_FIELDS = {"depends", "locks", "repo", "base"}
CONFIG_FILENAMES = ("config.toml", "workgraph.toml")


@dataclass
class TaskFile:
    task_id: str
    path: Path
    text: str
    depends: list[str]
    locks: list[str]
    repo: str
    base: str
    validation: list[str]


@dataclass
class ClaimedTask:
    task: TaskFile
    running_path: Path
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
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = Path(_workflow_cwd(ctx)) / root
    return root


def _config_path(ctx: WorkflowContext, root: Path) -> Path | None:
    raw = ctx.args.get("config") or ctx.args.get("config_path")
    if isinstance(raw, str) and raw.strip():
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(_workflow_cwd(ctx)) / path
        return path
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
        if not _deps_satisfied(task, done):
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


def _status(text: object) -> str:
    lower = str(text or "").lower()
    for status in ("conflict", "failed", "passed", "success", "resolved"):
        if f"status: {status}" in lower:
            return status
    if "status: needs_human" in lower:
        return "needs_human"
    if "pull request" in lower or "\npr:" in lower:
        return "success"
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


def _remote_delivery_present(text: str) -> bool:
    return not (
        _is_noneish(_report_field(text, "remote"))
        and _is_noneish(_report_field(text, "pr"))
    )


def _coerce_coder_status(coder_status: str, coder_text: str) -> tuple[str, str | None]:
    if coder_status != "success":
        return coder_status, None
    if _remote_delivery_present(coder_text):
        return coder_status, None
    return (
        "failed",
        (
            "Coder reported success without a remote branch or PR. "
            "Sandbox-local commits are not a WorkGraph delivery."
        ),
    )


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
        raise ValueError("WorkGraph development workers require operations backend 'agent_env'")
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


def _agent_env_session_from_report(text: str) -> str:
    value = _report_field(text, "AgentEnvSession")
    return "" if _is_noneish(value) else value


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
        "execution": (
            "Run inside the ARL agent_env sandbox. Do not use the host/control "
            "repository as the worktree. Include AgentEnvSession in the final "
            "response so this workflow can reuse the sandbox for follow-up "
            "agent calls in the same task. Use repository credentials only "
            "when the scenario, image, or ARL config_env provides them inside "
            "the sandbox, and never print those credentials. A coder success "
            "requires Remote or PR to be non-empty; a sandbox-local commit "
            "without a pushed branch is a failed delivery."
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
    claim: ClaimedTask,
    operations: dict[str, object],
) -> dict[str, object]:
    task = claim.task
    coder_scenario = _as_str(ctx.args.get("coder_scenario"), DEFAULT_CODER_SCENARIO)
    verifier_scenario = _as_str(
        ctx.args.get("verifier_scenario"),
        DEFAULT_VERIFIER_SCENARIO,
    )
    timeout = _as_float(ctx.args.get("agent_timeout_seconds"), 3600.0)
    agent_env_session = ""

    try:
        ctx.log(f"coding {task.task_id}")
        coder_result = await ctx.agent(
            f"Implement WorkGraph task {task.task_id}.",
            scenario=coder_scenario,
            atom_config=_atom_config(
                _operations_for_session(operations, agent_env_session),
                _context_for_task(task, "coder"),
            ),
            timeout=timeout,
            trace_label=f"{task.task_id}:coder",
        )
        coder_text = str(coder_result)
        coder_status = _status(coder_text)
        coder_status, delivery_error = _coerce_coder_status(
            coder_status,
            coder_text,
        )
        agent_env_session = (
            _agent_env_session_from_report(coder_text) or agent_env_session
        )

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
                    ),
                ),
                timeout=timeout,
                trace_label=f"{task.task_id}:verifier",
            )
            verifier_text = str(verifier_result)
            agent_env_session = (
                _agent_env_session_from_report(verifier_text)
                or agent_env_session
            )
            final_status = (
                "verified" if _status(verifier_text) == "passed" else "failed"
            )

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
        }
    except Exception as exc:
        coder_text = f"Status: failed\n\nWorkflow exception: {type(exc).__name__}: {exc}"
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
    operations = _operations_config_from_config(ctx.args, config_root)
    if _shared_attach_session_configured(operations):
        raise RuntimeError(
            "WorkGraph develop does not accept a shared agent_env attach_session "
            "for automatic task claiming. Use args.agent_env.image or "
            "AGENTM_AGENT_ENV_IMAGE so each claimed task gets its own ARL "
            "sandbox; the workflow only attaches follow-up agents to a "
            "per-task sandbox recorded under results/<task-id>/."
        )
    if not _agent_env_target_configured(operations):
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
