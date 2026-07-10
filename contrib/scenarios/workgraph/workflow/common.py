"""Shared helpers for WorkGraph workflow scripts."""

from __future__ import annotations

import os
import re
import time
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from agentm.core.lib import agentm_home_dir, expand_path_from_cwd
from agentm.extensions.builtin.workflow import WorkflowContext

TASK_HEADER_FIELDS = {"depends", "locks", "repo", "base"}
CONFIG_FILENAMES = ("config.toml", "workgraph.toml")


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
    return ctx.cwd or str(Path.cwd())


def _state_root(ctx: WorkflowContext) -> Path:
    raw = _as_str(ctx.args.get("state_dir"))
    if not raw.strip():
        return agentm_home_dir() / "workgraph"
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


def _next_attempt(root: Path, task: TaskFile) -> int:
    """Increment and return this task's attempt counter (1-based).

    Persisted in ``results/<task-id>/attempts.txt``. The counter goes into
    the worker context, so a bus-level retry of an unchanged task file gets
    a distinct workflow-journal key and actually re-runs instead of being
    served the previous attempt's cached report (re-running an identical
    semantic failure with identical context is re-sampling the same
    failure mode; the attempt number plus previous findings are the
    'vary something' — reliability-substrate.md §4.1 C3)."""
    path = _result_dir(root, task) / "attempts.txt"
    previous = 0
    if path.is_file():
        try:
            previous = int(path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            logger.warning("workgraph: corrupt attempts counter {}; resetting", path)
            previous = 0
    attempt = previous + 1
    path.write_text(f"{attempt}\n", encoding="utf-8")
    return attempt


def _atom_config(
    operations: dict[str, object],
    context: dict[str, object],
) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {"workgraph_context": dict(context)}
    if operations:
        config["operations"] = dict(operations)
    return config


def _move_task(source: Path, target_dir: Path) -> Path:
    """Move a task file into a queue directory.

    A same-named file already in the target queue is replaced (a later pass
    over the same task supersedes the earlier artifact) — loudly, so a
    surprising overwrite is visible in the logs.
    """
    target = target_dir / source.name
    if target.exists():
        logger.warning(
            "workgraph: replacing existing {} with newer result for {}",
            target,
            source.name,
        )
        target.unlink()
    shutil.move(str(source), str(target))
    return target


def _task_id(path: Path, text: str) -> str:
    # Task ids flow into branch names, workspace paths, and uv project
    # dirs — anything outside [A-Za-z0-9._-] (colons, CJK, commas) breaks
    # them, so prefer the REQ token and otherwise emit a sanitized slug.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                match = re.search(r"\bREQ-[A-Za-z0-9_.]+\b", heading)
                if match:
                    return match.group(0)
                slug = re.sub(r"[^A-Za-z0-9._-]+", "-", heading).strip("-")
                slug = re.sub(r"-{2,}", "-", slug)[:64].strip("-")
                if slug:
                    return slug
    return path.stem


def _done_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in (root / "done").glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        ids.add(path.stem)
        ids.add(_task_id(path, text))
        for token in re.findall(r"\bREQ-[A-Za-z0-9_.-]+\b", text):
            ids.add(token)
    return ids


def _deps_satisfied(depends: list[str], done: set[str]) -> bool:
    return all(dep in done for dep in depends)


def _safe_lock_name(lock: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", lock).strip("_") or "global"


_STALE_LOCK_SECONDS = 4 * 3600


def _lock_is_stale(path: Path) -> bool:
    """A lock is stale when its recorded owner pid is dead, or (for locks
    without owner metadata) when it is older than _STALE_LOCK_SECONDS."""
    pid = _lock_owner_pid(path)
    if pid is not None and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False
    try:
        age = time.time() - path.stat().st_mtime
    except OSError as exc:
        logger.warning("workgraph: cannot stat lock {}: {}", path, exc)
        return False
    return age > _STALE_LOCK_SECONDS


def _lock_owner_pid(path: Path) -> int | None:
    """Read the owner PID from a lock file, or None if absent/unparseable."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.debug("workgraph: cannot read lock {}: {}", path, exc)
        return None
    if len(lines) >= 3:
        try:
            return int(lines[2])
        except ValueError:
            pass
    return None


def _try_acquire_lock_file(path: Path, task_id: str, task_filename: str) -> bool:
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(f"{task_id}\n{task_filename}\n{os.getpid()}\n")
            return True
        except FileExistsError:
            if not _lock_is_stale(path):
                return False
            path.unlink(missing_ok=True)
    return False


def _release_locks(paths: list[Path]) -> None:
    """Delete lock files owned by this process."""
    for path in paths:
        pid = _lock_owner_pid(path)
        if pid is not None and pid != os.getpid():
            continue
        path.unlink(missing_ok=True)


def _report_field(text: str, name: str) -> str:
    target = name.lower()
    for raw_line in text.splitlines():
        line = re.sub(r"[*_`]", "", raw_line).strip()
        line = re.sub(r"^[-•]\s*", "", line)
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == target:
            return value.strip()
    return ""


def _is_noneish(value: str) -> bool:
    return not value.strip() or value.strip().lower() in {"none", "null", "n/a", "-"}


def _report_display_value(value: str) -> str:
    return "none" if _is_noneish(value) else value.strip()


def _report_optional_value(value: str) -> str | None:
    return None if _is_noneish(value) else value.strip()


def _drop_leading_report_header(text: str, names: set[str]) -> str:
    lines = text.strip().splitlines()
    index = 0
    saw_header = False
    while index < len(lines):
        raw_line = lines[index]
        key, separator, _value = raw_line.partition(":")
        if separator and key.strip().lower() in names:
            saw_header = True
            index += 1
            continue
        if saw_header and not raw_line.strip():
            index += 1
            break
        break
    return "\n".join(lines[index:]).strip()


def _structured_report_text(
    fields: list[tuple[str, str]],
    body: str,
) -> str:
    header = "\n".join(
        f"{name}: {_report_display_value(value)}" for name, value in fields
    )
    names = {name.lower() for name, _value in fields}
    clean_body = _drop_leading_report_header(body, names)
    return f"{header}\n\n{clean_body}" if clean_body else header


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
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "workgraph: `gh auth token` failed ({}); agents run without a "
            "GitHub token and pushes may fail inside the sandbox",
            exc,
        )
        return ""
    if completed.returncode != 0:
        logger.warning(
            "workgraph: `gh auth token` exited {}; agents run without a "
            "GitHub token and pushes may fail inside the sandbox",
            completed.returncode,
        )
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


def _operations_config_from_config(
    args: dict[str, object],
    config_root: dict[str, object],
    *,
    backend_error: str,
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
        raise ValueError(backend_error)
    config["backend"] = "agent_env"
    config["delete_on_shutdown"] = False
    _forward_git_env(config)
    return config


@dataclass(slots=True)
class ExecMode:
    """Resolved execution mode for worker agents.

    ``devbox`` means every agent attaches to one long-lived shared sandbox
    (warm build caches, no per-hour lifetime) and tasks isolate through
    per-task workspaces; otherwise each claimed task gets its own sandbox.
    """

    devbox: bool
    devbox_session: str


def _resolve_exec_mode(
    operations: dict[str, object],
    *,
    workflow: str,
) -> ExecMode:
    """Pop ``devbox_session`` from *operations* and validate the mode.

    Mutates *operations*: in devbox mode the shared session becomes the
    ``attach_session`` for every spawned agent.
    """
    devbox_session = str(operations.pop("devbox_session", "") or "").strip()
    if devbox_session:
        operations["attach_session"] = devbox_session
        return ExecMode(devbox=True, devbox_session=devbox_session)
    if _shared_attach_session_configured(operations):
        raise RuntimeError(
            f"WorkGraph {workflow} does not accept a shared agent_env "
            "attach_session for automatic task claiming. Use "
            "agent_env.devbox_session for the shared-devbox mode, or "
            "args.agent_env.image / AGENTM_AGENT_ENV_IMAGE so each claimed "
            "task gets its own ARL sandbox."
        )
    if not _agent_env_target_configured(operations):
        raise RuntimeError(
            f"WorkGraph {workflow} agents require an ARL agent_env target. "
            "Pass args.agent_env.image or set AGENTM_AGENT_ENV_IMAGE."
        )
    return ExecMode(devbox=False, devbox_session="")


def _review_standards(
    ctx: WorkflowContext,
    root: Path,
    config_root: dict[str, object],
) -> str:
    """Optional per-project review standards injected into worker context.

    Resolution order: ``args.review_standards`` (inline text) >
    ``args.review_standards_file`` / config ``review_standards_file`` (path,
    relative to the state root) > ``<state_root>/review_standards.md``.
    Keeps project-specific contract rules out of the generic agent prompts.
    """
    inline = ctx.args.get("review_standards")
    if isinstance(inline, str) and inline.strip():
        return inline.strip()
    raw_path = _config_str(ctx.args, config_root, "review_standards_file")
    path = (
        expand_path_from_cwd(raw_path, str(root))
        if raw_path.strip()
        else root / "review_standards.md"
    )
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("workgraph: cannot read review standards {}: {}", path, exc)
        return ""


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
