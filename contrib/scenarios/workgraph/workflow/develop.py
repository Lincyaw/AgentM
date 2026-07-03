"""WorkGraph development workflow.

This module-mode workflow implements one scheduling pass over a lightweight
filesystem task bus. The durable contract is intentionally small: task files
are Markdown, and only ``Depends``, ``Locks``, ``Repo``, ``Base``, and the
``## Validation`` section are interpreted by the scheduler. Worker agents own
git operations and report through their final response; the workflow records
those responses under the local state directory.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext

DEFAULT_STATE_DIR = ".agentm/workgraph"
DEFAULT_CODER_SCENARIO = "workgraph/agents/coder"
DEFAULT_VERIFIER_SCENARIO = "workgraph/agents/verifier"


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


def _ensure_dirs(root: Path) -> None:
    for name in (
        "ready",
        "running",
        "done",
        "failed",
        "conflicts",
        "locks",
        "results",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def _field(text: str, name: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s*(.*?)\s*$", re.I | re.M)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


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


def _result_dir(root: Path, task: TaskFile) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", task.task_id).strip("-")
    path = root / "results" / (safe or task.path.stem)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_result(root: Path, task: TaskFile, coder: str, verifier: str) -> None:
    path = _result_dir(root, task)
    (path / "task.md").write_text(task.text, encoding="utf-8")
    (path / "result.md").write_text(coder, encoding="utf-8")
    (path / "validation.md").write_text(verifier, encoding="utf-8")


def _move_finished(root: Path, claim: ClaimedTask, status: str) -> Path:
    if status == "passed":
        target_dir = root / "done"
    elif status == "conflict":
        target_dir = root / "conflicts"
    else:
        target_dir = root / "failed"
    target = target_dir / claim.running_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(claim.running_path), str(target))
    return target


def _operations_config(args: dict[str, object]) -> dict[str, object]:
    raw = args.get("agent_env")
    if isinstance(raw, dict) and raw:
        config = dict(raw)
        config.setdefault("backend", "agent_env")
        return config
    return {}


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

    try:
        ctx.log(f"coding {task.task_id}")
        coder_result = await ctx.agent(
            f"Implement WorkGraph task {task.task_id}.",
            scenario=coder_scenario,
            atom_config=_atom_config(
                operations,
                _context_for_task(task, "coder"),
            ),
            timeout=timeout,
            trace_label=f"{task.task_id}:coder",
        )
        coder_text = str(coder_result)
        coder_status = _status(coder_text)

        if coder_status == "conflict":
            verifier_text = (
                "Status: failed\n\nCoder reported a conflict before verification."
            )
            final_status = "conflict"
        elif coder_status == "failed":
            verifier_text = "Status: failed\n\nCoder reported failure before verification."
            final_status = "failed"
        else:
            ctx.log(f"verifying {task.task_id}")
            verifier_result = await ctx.agent(
                f"Verify WorkGraph task {task.task_id}.",
                scenario=verifier_scenario,
                atom_config=_atom_config(
                    operations,
                    _context_for_task(
                        task,
                        "verifier",
                        {"coder_result": coder_text},
                    ),
                ),
                timeout=timeout,
                trace_label=f"{task.task_id}:verifier",
            )
            verifier_text = str(verifier_result)
            final_status = "passed" if _status(verifier_text) == "passed" else "failed"

        _write_result(root, task, coder_text, verifier_text)
        target = _move_finished(root, claim, final_status)
        return {
            "task_id": task.task_id,
            "status": final_status,
            "from": str(claim.running_path),
            "to": str(target),
            "coder_status": coder_status,
            "result_dir": str(_result_dir(root, task)),
        }
    except Exception as exc:
        coder_text = f"Status: failed\n\nWorkflow exception: {type(exc).__name__}: {exc}"
        verifier_text = "Status: failed\n\nVerifier was not completed."
        _write_result(root, task, coder_text, verifier_text)
        target = _move_finished(root, claim, "failed")
        return {
            "task_id": task.task_id,
            "status": "failed",
            "from": str(claim.running_path),
            "to": str(target),
            "error": f"{type(exc).__name__}: {exc}",
            "result_dir": str(_result_dir(root, task)),
        }


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    root = _state_root(ctx)
    _ensure_dirs(root)
    max_parallel = max(1, _as_int(ctx.args.get("max_parallel"), 1))
    default_repo = _as_str(ctx.args.get("repo"))
    default_base = _as_str(ctx.args.get("base"), "main")
    operations = _operations_config(ctx.args)

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
