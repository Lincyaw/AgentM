"""ALE (Agents' Last Exam) bridge — task loading and sandbox-session shim.

ALE tasks live in an external checkout of ``agents-last-exam``. Each task is
``tasks/<domain>/<task>/main.py`` defining three hooks decorated with the
``cua_bench`` markers::

    @cb.tasks_config  def load() -> [cb.Task]          # prompt + metadata
    @cb.setup_task    async def start(cfg, session)     # pre-agent prep
    @cb.evaluate_task async def evaluate(cfg, session)  # grader -> [0,1]

The hooks run on the HOST and reach the sandbox only through ``session``.
This module provides the two pieces that let them run against an ARL
sandbox instead of ALE's cua-server VMs:

* a stub ``cua_bench`` module (tasks only use ``cb.Task``, the three
  decorators, and the ``cb.DesktopSession`` annotation — the real package
  drags in docker/datasets/matplotlib for nothing we need), and
* :class:`ArlGraderSession`, implementing the session API surface the
  graders actually call (``run_command`` / ``read_file`` / ``read_bytes``
  / ``file_exists`` / ...) on top of an ``arl.SandboxSession``.

Task data staging follows ALE's ``local:<host_dir>`` layout:
``<root>/<domain>/<task>/<variant>/{input,software,reference}`` — input and
software are staged before the agent, reference only after it finishes.
"""

from __future__ import annotations

import asyncio
import functools
import io
import importlib.util
import json
import os
import posixpath
import sys
import tarfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Remote root for staged task data inside the ARL pod. Lives under the
# workspace volume so arl upload/download (workspace-relative) can reach it
# and it survives for the grader after the agent session shuts down.
DEFAULT_WORKSPACE = "/workspace"
DEFAULT_REMOTE_ROOT = f"{DEFAULT_WORKSPACE}/agenthle"


# ---------------------------------------------------------------------------
# cua_bench stub
# ---------------------------------------------------------------------------

def install_cb_stub() -> None:
    """Register a minimal ``cua_bench`` in ``sys.modules`` (idempotent).

    Skipped when the real package is importable.
    """
    if "cua_bench" in sys.modules:
        return
    if importlib.util.find_spec("cua_bench") is not None:
        return

    mod = types.ModuleType("cua_bench")

    @dataclass
    class Task:
        description: str
        task_id: Optional[str] = None
        metadata: Optional[dict] = None
        computer: Optional[dict] = None

    def _tag(kind: str):
        def deco(_arg=None, /, *args, **kwargs):
            def wrap(fn):
                fn._td_type = kind
                return fn
            if callable(_arg):
                return wrap(_arg)
            return wrap
        return deco

    mod.Task = Task
    mod.DesktopSession = object  # annotation-only in task code
    mod.tasks_config = _tag("tasks_config")
    mod.setup_task = _tag("setup_task")
    mod.evaluate_task = _tag("evaluate_task")
    sys.modules["cua_bench"] = mod


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

@dataclass
class AleTask:
    """One loaded ALE task variant."""

    domain: str
    name: str
    variant_index: int
    prompt: str
    task_obj: Any                       # cb.Task from load()
    module: Any                         # the imported main.py module
    task_dir: Path                      # tasks/<domain>/<name> in the ALE repo
    timeout_s: int = 7200
    variant_name: str = "base"
    card: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"{self.domain}/{self.name}"


def _ensure_ale_paths(ale_repo: Path) -> None:
    """Make ``tasks.*`` (linux_runtime, common_config, utils) importable."""
    repo = str(ale_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def discover_tasks(ale_repo: Path) -> list[tuple[str, str]]:
    """List ``(domain, task)`` pairs that ship a main.py."""
    out: list[tuple[str, str]] = []
    tasks_root = ale_repo / "tasks"
    for card in sorted(tasks_root.glob("*/*/task_card.json")):
        task_dir = card.parent
        if (task_dir / "main.py").is_file():
            out.append((task_dir.parent.name, task_dir.name))
    return out


def load_task(
    ale_repo: Path,
    domain: str,
    name: str,
    *,
    variant_index: int = 0,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> AleTask:
    """Import a task's main.py and materialize one variant.

    ``REMOTE_ROOT_DIR`` must be set before ``tasks.linux_runtime`` is first
    imported — its dataclass default is captured at import time.
    """
    os.environ.setdefault("REMOTE_ROOT_DIR", remote_root)
    if os.environ["REMOTE_ROOT_DIR"] != remote_root:
        raise RuntimeError(
            f"REMOTE_ROOT_DIR already pinned to {os.environ['REMOTE_ROOT_DIR']!r}; "
            f"one process serves one remote root (wanted {remote_root!r})"
        )
    install_cb_stub()
    _ensure_ale_paths(ale_repo)

    task_dir = ale_repo / "tasks" / domain / name
    main_py = task_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(main_py)

    # Mirror ALE's TaskLoader sys.path behaviour: task dir + scripts/.
    for extra in (str(task_dir), str(task_dir / "scripts")):
        if Path(extra).is_dir() and extra not in sys.path:
            sys.path.insert(0, extra)

    mod_name = f"ale_task_{domain}_{name}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    variants = module.load()
    if not variants:
        raise RuntimeError(f"{domain}/{name}: load() returned no variants")
    if variant_index >= len(variants):
        raise IndexError(
            f"{domain}/{name}: variant {variant_index} out of range "
            f"({len(variants)} variants)"
        )
    task_obj = variants[variant_index]

    card: dict[str, Any] = {}
    card_path = task_dir / "task_card.json"
    if card_path.is_file():
        card = json.loads(card_path.read_text())
    vm = card.get("vm") or {}
    raw_timeout = vm.get("timeout") or vm.get("timeout_s") or card.get("timeout")
    try:
        timeout_s = int(raw_timeout) if raw_timeout else 7200
    except (TypeError, ValueError):
        timeout_s = 7200

    meta = task_obj.metadata or {}
    return AleTask(
        domain=domain,
        name=name,
        variant_index=variant_index,
        prompt=task_obj.description,
        task_obj=task_obj,
        module=module,
        task_dir=task_dir,
        timeout_s=timeout_s,
        variant_name=str(meta.get("variant") or meta.get("VARIANT_NAME") or "base"),
        card=card,
    )


# ---------------------------------------------------------------------------
# Score extraction (mirrors ale_run lifecycle._extract_score)
# ---------------------------------------------------------------------------

def extract_score(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("score", "final_score"):
            if key in value:
                return float(value[key])
        raise ValueError(f"evaluate() dict has no score/final_score: {value!r}")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("evaluate() returned an empty list")
        return float(value[0])
    return float(value)


# ---------------------------------------------------------------------------
# ARL-backed grader session
# ---------------------------------------------------------------------------

class CommandResult(dict):
    """cua-bench run_command returns a dict; some task helpers use attributes."""

    @property
    def stdout(self) -> str:
        return self.get("stdout", "") or ""

    @property
    def stderr(self) -> str:
        return self.get("stderr", "") or ""

    @property
    def return_code(self) -> int:
        return int(self.get("return_code", 0) or 0)

    returncode = return_code


    @property
    def success(self) -> bool:
        return bool(self.get("success"))


class ArlGraderSession:
    """The session API surface ALE task hooks use, over ``arl.SandboxSession``.

    All methods are async (task code awaits them); the sync arl SDK calls run
    in a thread. Paths are absolute pod paths; file transfer converts them to
    workspace-relative for the arl file API and falls back to shell for paths
    outside the workspace.
    """

    def __init__(self, session: Any, *, workspace: str = DEFAULT_WORKSPACE,
                 os_type: str = "linux") -> None:
        self._session = session
        self._workspace = workspace.rstrip("/") or "/"
        self.os_type = os_type

    # -- helpers ---------------------------------------------------------

    def _rel(self, path: str) -> str | None:
        p = posixpath.normpath(path)
        if self._workspace == "/":
            return p.lstrip("/") or None
        if p == self._workspace:
            return None
        if p.startswith(self._workspace + "/"):
            return p[len(self._workspace) + 1:] or None
        return None

    async def _thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    async def _exec(self, command: str, *, timeout: float | None = None) -> CommandResult:
        step = {
            "name": "cmd",
            "command": ["bash", "-lc", command],
        }
        if timeout and timeout > 0:
            step["timeout_seconds"] = int(timeout)
        resp = await self._thread(self._session.execute, [step])
        result = resp.results[0]
        output = getattr(result, "output", None)
        stdout = getattr(output, "stdout", "") if output else ""
        stderr = getattr(output, "stderr", "") if output else ""
        exit_code = getattr(output, "exit_code", None) if output else None
        if exit_code is None:
            exit_code = 0 if getattr(result, "status", "") in ("succeeded", "success") else 1
        return CommandResult(
            success=int(exit_code) == 0,
            stdout=stdout or "",
            stderr=stderr or "",
            return_code=int(exit_code),
        )

    # -- API used by task hooks ------------------------------------------

    async def run_command(self, command: str, *, check: bool = True,
                          timeout: float | None = None) -> CommandResult:
        result = await self._exec(command, timeout=timeout)
        if check and not result.success:
            raise RuntimeError(
                f"command failed rc={result.return_code}: {command[:200]} "
                f"stderr={result.stderr[:300]}"
            )
        return result

    async def read_bytes(self, path: str) -> bytes:
        rel = self._rel(path)
        if rel is not None:
            return await self._thread(self._session.download_file, rel)
        result = await self._exec(f"base64 -w0 {_q(path)}")
        if not result.success:
            raise FileNotFoundError(f"{path}: {result.stderr[:200]}")
        import base64
        return base64.b64decode(result.stdout.strip())

    async def read_file(self, path: str) -> str:
        return (await self.read_bytes(path)).decode("utf-8", errors="replace")

    async def write_bytes(self, path: str, data: bytes) -> None:
        rel = self._rel(path)
        if rel is not None:
            await self._thread(self._session.upload_file, rel, data)
            return
        import base64
        encoded = base64.b64encode(data).decode("ascii")
        result = await self._exec(
            f"mkdir -p {_q(posixpath.dirname(path))} && "
            f"printf %s {_q(encoded)} | base64 -d > {_q(path)}"
        )
        if not result.success:
            raise RuntimeError(f"write_bytes({path}) failed: {result.stderr[:200]}")

    async def write_file(self, path: str, content: str) -> None:
        await self.write_bytes(path, content.encode("utf-8"))

    async def write_text(self, path: str, content: str) -> None:
        await self.write_file(path, content)

    async def file_exists(self, path: str) -> bool:
        return (await self._exec(f"test -f {_q(path)}")).success

    async def directory_exists(self, path: str) -> bool:
        return (await self._exec(f"test -d {_q(path)}")).success

    async def list_dir(self, path: str) -> list[str]:
        result = await self._exec(f"ls -1A {_q(path)}")
        if not result.success:
            return []
        return [line for line in result.stdout.splitlines() if line]

    async def screenshot(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "GUI actions are not available on the ARL backend; "
            "this ALE task needs a desktop sandbox"
        )

    async def check_status(self) -> bool:
        return (await self._exec("echo ok")).success


def _q(text: str) -> str:
    """Shell-quote for bash -lc composition."""
    return "'" + text.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Data staging
# ---------------------------------------------------------------------------

def _tar_dir(local_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for entry in sorted(local_dir.rglob("*")):
            tar.add(entry, arcname=str(entry.relative_to(local_dir)))
    return buf.getvalue()


async def stage_dir(
    grader: ArlGraderSession,
    local_dir: Path,
    remote_dir: str,
    *,
    executable: bool = False,
) -> None:
    """Upload a host directory tree into the pod at ``remote_dir``."""
    payload = _tar_dir(local_dir)
    stage_path = posixpath.join(remote_dir, ".ale_stage.tgz")
    await grader.run_command(f"mkdir -p {_q(remote_dir)}")
    await grader.write_bytes(stage_path, payload)
    await grader.run_command(
        f"tar -xzf {_q(stage_path)} -C {_q(remote_dir)} && rm -f {_q(stage_path)}"
    )
    if executable:
        await grader.run_command(
            f"find {_q(remote_dir)} -type f -exec chmod +x {{}} +", check=False,
        )


async def stage_task_input(
    grader: ArlGraderSession,
    task: AleTask,
    data_root: Path,
    remote_root: str,
) -> None:
    """Stage input/ + software/ before the agent runs; create output/."""
    host_base = data_root / task.domain / task.name / task.variant_name
    remote_base = posixpath.join(remote_root, task.domain, task.name, task.variant_name)
    input_dir = host_base / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"task data missing: {input_dir} (expected ALE local layout "
            f"<data_root>/<domain>/<task>/<variant>/input)"
        )
    await grader.run_command(f"mkdir -p {_q(posixpath.join(remote_base, 'output'))}")
    await stage_dir(grader, input_dir, posixpath.join(remote_base, "input"))
    software_dir = host_base / "software"
    if software_dir.is_dir():
        await stage_dir(
            grader, software_dir, posixpath.join(remote_base, "software"),
            executable=True,
        )


async def stage_task_reference(
    grader: ArlGraderSession,
    task: AleTask,
    data_root: Path,
    remote_root: str,
) -> None:
    """Stage reference/ AFTER the agent finished — the hidden answer key."""
    host_ref = data_root / task.domain / task.name / task.variant_name / "reference"
    if not host_ref.is_dir():
        raise FileNotFoundError(
            f"reference data missing: {host_ref} — grading would be meaningless"
        )
    remote_ref = posixpath.join(
        remote_root, task.domain, task.name, task.variant_name, "reference",
    )
    await stage_dir(grader, host_ref, remote_ref)


# ---------------------------------------------------------------------------
# Baked-image staging: ONE private data image per task carries input/,
# software/ and reference/. The agent's executor container cannot read the
# private container's filesystem — only the shared workspace — so staging
# runs INSIDE the data container (busybox → sh, not bash) and the timing of
# the two copies preserves ALE's reference secrecy.
# ---------------------------------------------------------------------------

async def _exec_in_container(
    sandbox: Any, container: str, name: str, script: str, *, timeout: int = 900,
) -> None:
    resp = await asyncio.to_thread(
        sandbox.execute_container, container,
        [{"name": name, "command": ["sh", "-c", script],
          "timeout_seconds": timeout}],
    )
    result = resp.results[0]
    output = getattr(result, "output", None)
    exit_code = getattr(output, "exit_code", None) if output else None
    if exit_code is None:
        exit_code = 0 if getattr(result, "status", "") in ("succeeded", "success") else 1
    if int(exit_code) != 0:
        stderr = (getattr(output, "stderr", "") or "")[:300] if output else ""
        raise RuntimeError(
            f"{name} via private container {container!r} failed "
            f"(rc={exit_code}): {stderr}"
        )


async def stage_input_from_container(
    sandbox: Any,
    task: AleTask,
    remote_root: str,
    *,
    container: str = "ale-data",
    baked_root: str = "/ale",
) -> None:
    """Pre-agent: copy input/ + software/ from the data container; mk output/."""
    baked = posixpath.join(baked_root, task.domain, task.name, task.variant_name)
    remote_base = posixpath.join(remote_root, task.domain, task.name, task.variant_name)
    script = (
        f"test -d {_q(baked + '/input')} || "
        f"{{ echo 'baked task data missing at {baked}/input' >&2; exit 3; }}; "
        f"mkdir -p {_q(remote_base + '/output')} {_q(remote_base + '/input')} && "
        f"cp -a {_q(baked + '/input')}/. {_q(remote_base + '/input')}/ && "
        f"if [ -d {_q(baked + '/software')} ]; then "
        f"mkdir -p {_q(remote_base + '/software')} && "
        f"cp -a {_q(baked + '/software')}/. {_q(remote_base + '/software')}/; fi"
    )
    await _exec_in_container(sandbox, container, "stage-input", script)


async def stage_reference_from_container(
    sandbox: Any,
    task: AleTask,
    remote_root: str,
    *,
    container: str = "ale-data",
    baked_root: str = "/ale",
) -> None:
    """Post-agent ONLY: copy reference/ (the answer key) into the workspace."""
    baked_ref = posixpath.join(
        baked_root, task.domain, task.name, task.variant_name, "reference",
    )
    remote_ref = posixpath.join(
        remote_root, task.domain, task.name, task.variant_name, "reference",
    )
    script = (
        f"test -d {_q(baked_ref)} || "
        f"{{ echo 'baked reference missing at {baked_ref}' >&2; exit 3; }}; "
        f"mkdir -p {_q(remote_ref)} && cp -a {_q(baked_ref)}/. {_q(remote_ref)}/"
    )
    await _exec_in_container(sandbox, container, "stage-ref", script, timeout=600)


__all__ = [
    "AleTask",
    "ArlGraderSession",
    "CommandResult",
    "DEFAULT_REMOTE_ROOT",
    "DEFAULT_WORKSPACE",
    "discover_tasks",
    "extract_score",
    "install_cb_stub",
    "load_task",
    "stage_input_from_container",
    "stage_reference_from_container",
    "stage_task_input",
    "stage_task_reference",
]
