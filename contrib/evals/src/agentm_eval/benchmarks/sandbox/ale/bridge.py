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
import threading
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:  # long grader commands (uv sync + sims) outlive the execute HTTP window
    from arl import GatewayOperationTimeout
except ImportError:  # pragma: no cover - optional/older SDKs
    GatewayOperationTimeout = ()  # type: ignore[assignment]

# Remote root for staged task data inside the ARL pod. Lives under the
# workspace volume so arl upload/download (workspace-relative) can reach it
# and it survives for the grader after the agent session shuts down.
DEFAULT_WORKSPACE = "/workspace"
DEFAULT_REMOTE_ROOT = f"{DEFAULT_WORKSPACE}/agenthle"

# Serializes the task-import critical section (sys.path + sys.modules mutation
# and exec_module). Tasks run concurrently across threads (`-j`), and each
# `from score_outputs import ...` binds into the task module at exec time, so
# imports must not interleave; once loaded, bound names are thread-safe.
_IMPORT_LOCK = threading.Lock()


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

    @dataclass(slots=True)
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

    setattr(mod, "Task", Task)
    setattr(mod, "DesktopSession", object)  # annotation-only in task code
    setattr(mod, "tasks_config", _tag("tasks_config"))
    setattr(mod, "setup_task", _tag("setup_task"))
    setattr(mod, "evaluate_task", _tag("evaluate_task"))
    sys.modules["cua_bench"] = mod


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

@dataclass(slots=True)
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

    # Import is a global-state critical section (sys.path/sys.modules); serialize
    # it so concurrent task loads (`-j`) don't interleave their imports.
    with _IMPORT_LOCK:
        # Mirror ALE's TaskLoader sys.path behaviour (task dir + scripts/), but
        # MOVE these to the front every load: many tasks' scripts/ dirs pile up
        # on sys.path, and `import score_outputs` resolves by sys.path order, so
        # this task's scripts/ must win — hence remove-then-insert, not
        # insert-if-absent. scripts/ ends up at index 0.
        for extra in (str(task_dir), str(task_dir / "scripts")):
            if Path(extra).is_dir():
                while extra in sys.path:
                    sys.path.remove(extra)
                sys.path.insert(0, extra)

        # Tasks import helpers by generic names from their own scripts/ dir
        # (`from score_outputs import ...`). One process serves many tasks, so a
        # previously-loaded task's `score_outputs` would stay cached in
        # sys.modules and shadow this task's — raising "cannot import name X
        # from score_outputs". Drop any module imported from a *foreign* task
        # scripts/ dir so this task re-imports its own. Shared framework modules
        # (tasks/*.py, not under a scripts/ dir) and this task's own are kept.
        tasks_root = f"{ale_repo / 'tasks'}{os.sep}"
        scripts_marker = f"{os.sep}scripts{os.sep}"
        this_scripts = f"{task_dir / 'scripts'}{os.sep}"
        for m_name, m in list(sys.modules.items()):
            m_file = getattr(m, "__file__", None) or ""
            if (m_file.startswith(tasks_root) and scripts_marker in m_file
                    and not m_file.startswith(this_scripts)):
                del sys.modules[m_name]

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
    # The variant name drives every staged path (<root>/<domain>/<name>/<variant>).
    # Prefer an explicit key, but most tasks don't set one and instead encode the
    # real variant only in their metadata *paths* (task_dir/input_dir =
    # .../<name>/<variant>/...). Deriving it from there — rather than defaulting
    # to "base" — is essential for the non-"base" tasks (e.g. zscaler_fy2025,
    # n10_critical_u01_correlators), whose data lives under that real variant dir
    # and whose grader reads input_dir from it.
    variant_name = str(meta.get("variant") or meta.get("VARIANT_NAME") or "")
    if not variant_name:
        marker = f"/{name}/"
        for key in ("task_dir", "input_dir", "software_dir", "output_dir"):
            p = meta.get(key)
            if isinstance(p, str) and marker in p:
                variant_name = p.split(marker, 1)[1].split("/", 1)[0]
                break
        variant_name = variant_name or "base"
    return AleTask(
        domain=domain,
        name=name,
        variant_index=variant_index,
        prompt=task_obj.description,
        task_obj=task_obj,
        module=module,
        task_dir=task_dir,
        timeout_s=timeout_s,
        variant_name=variant_name,
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


def _decode_result(resp: Any) -> tuple[int, str, str]:
    """Decode the first step of an arl execute/execute_container response into
    ``(exit_code, stdout, stderr)``. Trusts ``output.exit_code`` and falls back
    to the ``status`` string when the runner didn't surface a code."""
    result = resp.results[0]
    output = getattr(result, "output", None)
    stdout = getattr(output, "stdout", "") if output else ""
    stderr = getattr(output, "stderr", "") if output else ""
    exit_code = getattr(output, "exit_code", None) if output else None
    if exit_code is None:
        exit_code = 0 if getattr(result, "status", "") in ("succeeded", "success") else 1
    return int(exit_code), stdout or "", stderr or ""


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
        step: dict[str, Any] = {
            "name": "cmd",
            "command": ["bash", "-lc", command],
        }
        if timeout and timeout > 0:
            # The gateway reads the camelCase key (StepRequest json tag
            # "timeoutSeconds"); execute() forwards the raw dict unvalidated,
            # so a snake_case key would be silently dropped and the command
            # would inherit the short default sidecar deadline instead.
            step["timeoutSeconds"] = int(timeout)
        op_id = str(uuid.uuid4())
        try:
            resp = await self._thread(
                self._session.execute, [step], operation_id=op_id
            )
        except GatewayOperationTimeout as exc:
            # The command outlived the execute HTTP window (e.g. a grader's
            # `uv sync` + simulation). The op keeps running server-side; poll
            # its operation_id until it finishes instead of scoring a false 0.
            resp = await self._poll_operation(
                getattr(exc, "operation_id", op_id) or op_id, timeout=timeout
            )
        exit_code, stdout, stderr = _decode_result(resp)
        return CommandResult(
            success=exit_code == 0, stdout=stdout, stderr=stderr,
            return_code=exit_code,
        )

    async def _poll_operation(self, op_id: str, *, timeout: float | None = None):
        """Wait for a pending execute operation to complete and return its
        ExecuteResponse. Polls the gateway's operation endpoint; the op is done
        once ``result`` is populated (even on non-zero exit), and an ``error``
        means infra failure."""
        client = getattr(self._session, "_client", None)
        sid = getattr(self._session, "session_id", None)
        if client is None or sid is None:
            raise RuntimeError(
                "execute operation timed out and this session cannot poll it"
            )
        # Poll a bit past the command's own timeout; default generously since
        # graders run offline sims that legitimately take many minutes.
        ceiling = float(timeout) if timeout and timeout > 0 else 3600.0
        deadline = ceiling + 120.0
        waited = 0.0
        interval = 5.0
        while True:
            info = await self._thread(client.get_execute_operation, sid, op_id)
            if getattr(info, "result", None) is not None:
                return info.result
            err = getattr(info, "error", "") or ""
            status = (getattr(info, "status", "") or "").lower()
            if err or status in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError(
                    f"execute operation {op_id} failed: {err or status}"
                )
            if waited >= deadline:
                raise RuntimeError(
                    f"execute operation {op_id} still pending after {waited:.0f}s"
                )
            await asyncio.sleep(interval)
            waited += interval

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
            return []  # non-existent path → empty listing
        # `ls -1A <file>` prints the file's own name, not a directory listing;
        # callers that recurse on "listable" (e.g. cvrp's _download_remote_tree)
        # would then treat a file as a dir and build bogus paths. Match
        # os.listdir semantics: raise on a non-directory so they fall back to
        # downloading it as a file.
        if not (await self._exec(f"test -d {_q(path)}")).success:
            raise NotADirectoryError(path)
        return [line for line in result.stdout.splitlines() if line]

    async def screenshot(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "GUI actions are not available on the ARL backend; "
            "this ALE task needs a desktop sandbox"
        )

    async def check_status(self) -> bool:
        return (await self._exec("echo ok")).success

    @property
    def interface(self) -> "_GraderInterface":
        """cua_bench exposes filesystem helpers under ``session.interface``;
        ALE graders only ever call ``create_dir``/``delete_dir`` on it."""
        return _GraderInterface(self)


class _GraderInterface:
    """Minimal ``session.interface`` shim — the only methods ALE graders use
    are directory create/delete, mapped to shell mkdir/rm in the sandbox."""

    def __init__(self, session: "ArlGraderSession") -> None:
        self._session = session

    async def create_dir(self, path: str) -> None:
        await self._session._exec(f"mkdir -p {_q(path)}")

    async def delete_dir(self, path: str) -> None:
        await self._session._exec(f"rm -rf {_q(path)}")


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
    await upload_remote_dir(grader, _tar_dir(local_dir), remote_dir)
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
) -> bool:
    """Stage reference/ AFTER the agent finished — the hidden answer key.

    Returns True if a reference was staged. A no-op returning False when no
    reference exists locally: some ALE graders score the agent output against
    the input evidence with a rubric and never read reference/, so they can be
    graded without the (often gated) answer key. Callers should record the
    return value so a graded-without-key result is distinguishable from one
    graded against a real reference.
    """
    host_ref = data_root / task.domain / task.name / task.variant_name / "reference"
    if not host_ref.is_dir() or not any(host_ref.iterdir()):
        logger.info(
            "[{}/{}] no local reference/ — skipping (grader may not need it)",
            task.domain, task.name,
        )
        return False
    remote_ref = posixpath.join(
        remote_root, task.domain, task.name, task.variant_name, "reference",
    )
    await stage_dir(grader, host_ref, remote_ref)
    return True


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
    exit_code, _, stderr = _decode_result(resp)
    if exit_code != 0:
        raise RuntimeError(
            f"{name} via private container {container!r} failed "
            f"(rc={exit_code}): {stderr[:300]}"
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


async def download_remote_dir(grader: ArlGraderSession, remote_dir: str) -> bytes:
    """Tar a sandbox dir and pull it to host as gzip bytes (binary-safe).

    Tars into a workspace-relative path so read_bytes takes the native SDK
    download_file path rather than base64-over-shell (which inflates a large
    archive ~33% and buffers it as a command-result string)."""
    tgz = posixpath.join(grader._workspace, ".ale_dl.tgz")
    await grader.run_command(
        f"tar -czf {_q(tgz)} -C {_q(remote_dir)} . 2>/dev/null || "
        f"tar -czf {_q(tgz)} -T /dev/null", check=False,
    )
    try:
        return await grader.read_bytes(tgz)
    finally:
        await grader.run_command(f"rm -f {_q(tgz)}", check=False)


async def upload_remote_dir(
    grader: ArlGraderSession, tgz: bytes, remote_dir: str,
) -> None:
    """Restore a gzip tarball (from download_remote_dir) into a sandbox dir."""
    stage = posixpath.join(remote_dir, ".ale_out.tgz")
    await grader.run_command(f"mkdir -p {_q(remote_dir)}")
    await grader.write_bytes(stage, tgz)
    await grader.run_command(
        f"tar -xzf {_q(stage)} -C {_q(remote_dir)} && rm -f {_q(stage)}"
    )


async def stage_reference_from_container(
    sandbox: Any,
    task: AleTask,
    remote_root: str,
    *,
    container: str = "ale-data",
    baked_root: str = "/ale",
) -> bool:
    """Post-agent ONLY: copy reference/ (the answer key) into the workspace.

    Returns True if reference was staged, False if the baked image has no
    reference/ (rubric-graded tasks that don't need an answer key).
    """
    baked_ref = posixpath.join(
        baked_root, task.domain, task.name, task.variant_name, "reference",
    )
    remote_ref = posixpath.join(
        remote_root, task.domain, task.name, task.variant_name, "reference",
    )
    script = (
        f"test -d {_q(baked_ref)} || exit 0; "
        f"mkdir -p {_q(remote_ref)} && cp -a {_q(baked_ref)}/. {_q(remote_ref)}/"
    )
    await _exec_in_container(sandbox, container, "stage-ref", script, timeout=600)
    check = (f"test -d {_q(remote_ref)} && echo yes || echo no")
    resp = await asyncio.to_thread(
        sandbox.execute_container, container,
        [{"name": "check-ref", "command": ["sh", "-c", check],
          "timeout_seconds": 30}],
    )
    _, stdout, _ = _decode_result(resp)
    staged = stdout.strip() == "yes"
    if not staged:
        logger.info(
            "[{}/{}] no baked reference/ — skipping (grader may not need it)",
            task.domain, task.name,
        )
    return staged


__all__ = [
    "AleTask",
    "ArlGraderSession",
    "CommandResult",
    "DEFAULT_REMOTE_ROOT",
    "DEFAULT_WORKSPACE",
    "discover_tasks",
    "extract_score",
    "install_cb_stub",
    "download_remote_dir",
    "load_task",
    "stage_input_from_container",
    "stage_reference_from_container",
    "upload_remote_dir",
    "stage_task_input",
    "stage_task_reference",
]
