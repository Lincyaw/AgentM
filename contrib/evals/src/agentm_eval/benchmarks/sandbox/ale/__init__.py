"""ALE (Agents' Last Exam) adapter — real professional tasks in ARL sandboxes.

Flow per task (the ALE contract, re-hosted on agent_env):

  1. load ``tasks/<domain>/<task>/main.py`` from the ALE checkout (host side)
  2. create an ARL sandbox, stage ``input/`` + ``software/`` under
     ``/workspace/agenthle/<domain>/<task>/<variant>/``, create ``output/``
  3. run the task's ``start()`` setup hook against the sandbox
  4. run an AgentM session (scenario ``ale:arl``) ATTACHED to the same
     sandbox with the task prompt; the agent's bash/file tools execute
     in the pod
  5. stage the hidden ``reference/`` (only now — the answer key must not
     exist while the agent runs), run the task's ``evaluate()`` grader on
     the host through the sandbox shim, normalize the score to [0,1]
  6. record a TaskResult; delete the sandbox

Scope: Linux CLI tasks. GUI/Windows/GPU ALE tasks are out of scope on the
agent_env backend (no desktop in the pod).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from loguru import logger

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult

from . import bridge

_TRANSIENT_CREATE_MARKERS = (
    "pool at maximum capacity", "queued for", "context deadline exceeded",
    "Gateway error (429)", "Gateway error (500)", "Gateway error (503)",
    "Server disconnected", "Connection reset", "Connection refused",
    "ReadTimeout", "timed out",
)
_MAX_CREATE_ATTEMPTS = 20


@dataclass(slots=True)
class AleRunConfig:
    ale_repo: Path
    scenario: str
    model: str | None
    exp_id: str
    out_dir: Path
    data_root: Path | None = None      # upload mode: host task-data root
    image: str | None = None           # sandbox runtime image (both modes)
    task_images: str | None = None     # baked mode: registry prefix of per-task images
    images_tag: str = "latest"
    profile: str = "default"
    agent_timeout: float = 0.0        # 0 → use the task card timeout
    eval_timeout: float = 7200.0
    max_turns: int | None = None
    keep_sandbox: bool = True
    remote_root: str = bridge.DEFAULT_REMOTE_ROOT
    workspace: str = bridge.DEFAULT_WORKSPACE
    defer_grading: bool = False        # run --no-eval: solve + save output, skip grade
    outputs_dir: Path | None = None    # where agent-phase outputs are saved/loaded
    reference_root: Path | None = None  # host reference source for grade phase

    @property
    def baked(self) -> bool:
        return self.task_images is not None

    def require_data_root(self) -> Path:
        if self.data_root is None:
            raise ValueError("data_root is required when baked task images are not used")
        return self.data_root

    def require_task_images(self) -> str:
        if self.task_images is None:
            raise ValueError("task_images is required in baked mode")
        return self.task_images

    def require_outputs_dir(self) -> Path:
        if self.outputs_dir is None:
            raise ValueError("outputs_dir is required for deferred grading")
        return self.outputs_dir


# ---------------------------------------------------------------------------
# Single-task pipeline
# ---------------------------------------------------------------------------

def _run_scored(task_ref, cfg, attempt, async_fn, *, log_suffix=""):
    """Run one task coroutine in a worker thread with a per-thread log sink,
    stamp latency, and persist the score.json. Shared by run + grade."""
    started = time.monotonic()
    domain, _, name = task_ref.partition("/")
    thread_id = threading.get_ident()
    sink_id = logger.add(
        str(cfg.out_dir / "artifacts" / f"{domain}__{name}{log_suffix}.log"),
        filter=lambda record: record["thread"].id == thread_id,
        enqueue=True, backtrace=False, diagnose=False,
    )
    try:
        result = asyncio.run(async_fn(task_ref, cfg, attempt))
    except Exception as e:  # noqa: BLE001
        logger.exception("ALE {} crashed", task_ref)
        result = TaskResult(task_id=task_ref, status="error", error=str(e)[:2000])
    finally:
        logger.remove(sink_id)
    result.latency_ms = int((time.monotonic() - started) * 1000)
    (cfg.out_dir / "artifacts" / f"{domain}__{name}.score.json").write_text(
        json.dumps(result.to_dict(), indent=2)
    )
    return result


def run_one(task_ref: str, cfg: AleRunConfig, attempt: int = 0) -> TaskResult:
    """Run one ``domain/task`` end to end (agent + optional grade)."""
    return _run_scored(task_ref, cfg, attempt, _run_one_async)


def grade_one(task_ref: str, cfg: AleRunConfig, attempt: int = 0) -> TaskResult:
    """Grade a previously-saved agent output (phase 2 of ``run --no-eval``).

    Recreates a minimal sandbox, uploads the archived output/ and the host
    reference/, then runs the task's evaluate() through the shim.
    """
    return _run_scored(task_ref, cfg, attempt, _grade_one_async, log_suffix=".grade")


async def _grade_one_async(task_ref: str, cfg: AleRunConfig, attempt: int) -> TaskResult:
    import arl

    domain, _, name = task_ref.partition("/")
    task = bridge.load_task(cfg.ale_repo, domain, name, remote_root=cfg.remote_root)
    out_dir = task.task_obj.metadata.get("remote_output_dir")
    archive = cfg.require_outputs_dir() / f"{domain}__{name}__{attempt}.output.tgz"
    if not archive.is_file():
        return TaskResult(task_id=task_ref, status="error",
                          error=f"no saved output archive: {archive}")

    sandbox = arl.SandboxSession(image=cfg.image, profile=cfg.profile,
                                 idle_timeout_seconds=7200)
    await asyncio.to_thread(sandbox.create_sandbox)
    grader = bridge.ArlGraderSession(sandbox, workspace=cfg.workspace)
    try:
        # restore agent output (upload_remote_dir creates out_dir itself)
        await bridge.upload_remote_dir(grader, archive.read_bytes(), out_dir)
        # optionally restage input/ for graders that read it (rubric graders)
        if cfg.data_root is not None:
            await bridge.stage_task_input(grader, task, cfg.data_root, cfg.remote_root)
        ref_root = cfg.reference_root or cfg.data_root
        ref_staged = (
            await bridge.stage_task_reference(grader, task, ref_root, cfg.remote_root)
            if ref_root is not None
            else False
        )
        raw = await asyncio.wait_for(
            task.module.evaluate(task.task_obj, grader), timeout=cfg.eval_timeout,
        )
        reward = bridge.extract_score(raw)
        logger.info("[{}] reward={} (raw={})", task_ref, reward, raw)
        return TaskResult(
            task_id=task_ref,
            status="pass" if reward >= 1.0 else "fail",
            score={"reward": reward},
            metadata={"raw_eval": str(raw)[:500], "attempt": attempt,
                      "phase": "grade", "output_archive": str(archive),
                      "reference_staged": ref_staged},
        )
    finally:
        try:
            await asyncio.to_thread(sandbox.delete_sandbox)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[{}] sandbox cleanup failed: {}", task_ref, exc)


async def _run_one_async(task_ref: str, cfg: AleRunConfig, attempt: int) -> TaskResult:
    import arl

    domain, _, name = task_ref.partition("/")
    task = bridge.load_task(
        cfg.ale_repo, domain, name, remote_root=cfg.remote_root,
    )
    logger.info("[{}] loaded: timeout={}s prompt={} chars",
                task_ref, task.timeout_s, len(task.prompt))

    if cfg.baked:
        from .images import DATA_CONTAINER_NAME, data_image_ref
        data_image = data_image_ref(
            cfg.require_task_images(), domain, name, cfg.images_tag,
        )
        sandbox_image = cfg.image          # shared runtime image (pool-friendly)
        private_containers = [{
            "name": DATA_CONTAINER_NAME,
            "image": data_image,
            "mountWorkspace": True,
        }]
        logger.info("[{}] runtime {} + data {}", task_ref, sandbox_image, data_image)
    else:
        sandbox_image = cfg.image
        private_containers = None

    sandbox = arl.SandboxSession(
        image=sandbox_image,
        profile=cfg.profile,
        idle_timeout_seconds=max(task.timeout_s + 3600, 7200),
        # Large private-container data images (e.g. idp_ensemble_scoring ships
        # a 12 GB input layer) can take minutes to pull before the pod is
        # ready; give allocation a generous window so the claim isn't reaped
        # as "not ready" mid-pull.
        allocation_timeout_seconds=2400,
        private_containers=private_containers,
    )
    await asyncio.to_thread(sandbox.create_sandbox)
    arl_session_id = sandbox.session_id
    logger.info("[{}] sandbox {} created", task_ref, arl_session_id)

    grader = bridge.ArlGraderSession(sandbox, workspace=cfg.workspace)
    agent_session_id = ""
    try:
        # -- stage input + setup hook (pre-agent) -------------------------
        if cfg.baked:
            await bridge.stage_input_from_container(sandbox, task, cfg.remote_root)
        else:
            await bridge.stage_task_input(
                grader, task, cfg.require_data_root(), cfg.remote_root,
            )
        start_fn = getattr(task.module, "start", None)
        if callable(start_fn):
            try:
                await start_fn(task.task_obj, grader)
            except Exception as e:  # noqa: BLE001
                logger.warning("[{}] setup hook failed (continuing): {}", task_ref, e)

        # -- agent ---------------------------------------------------------
        agent_timeout = cfg.agent_timeout or float(task.timeout_s)
        agent_result = await _run_agent(
            task_ref, task.prompt, arl_session_id, cfg, agent_timeout, attempt,
        )
        agent_session_id = agent_result.get("session_id", "")
        session_ids = [agent_session_id] if agent_session_id else []
        if agent_result.get("error"):
            return TaskResult(
                task_id=task_ref, status="error",
                error=f"agent: {agent_result['error']}",
                session_ids=session_ids,
                metadata={"arl_session": arl_session_id},
            )

        # -- agent-only phase: persist output/, defer grading --------------
        out_dir = task.task_obj.metadata.get("remote_output_dir")
        if cfg.defer_grading:
            saved = "no-output-dir"
            if out_dir and cfg.outputs_dir:
                tgz = await bridge.download_remote_dir(grader, out_dir)
                dest = cfg.outputs_dir / f"{domain}__{name}__{attempt}.output.tgz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tgz)
                saved = str(dest)
            return TaskResult(
                task_id=task_ref, status="solved",  # ungraded phase (see result.py enum)
                session_ids=session_ids,
                metadata={
                    "arl_session": arl_session_id,
                    "agent_timed_out": bool(agent_result.get("timed_out")),
                    "output_archive": saved, "remote_output_dir": out_dir,
                    "attempt": attempt, "phase": "agent",
                },
            )

        # -- reference + grade (post-agent) --------------------------------
        ref_staged = False
        if cfg.baked:
            ref_staged = await bridge.stage_reference_from_container(
                sandbox, task, cfg.remote_root,
            )
        else:
            ref_root = cfg.reference_root or cfg.require_data_root()
            ref_staged = await bridge.stage_task_reference(
                grader, task, ref_root, cfg.remote_root,
            )
        raw = await asyncio.wait_for(
            task.module.evaluate(task.task_obj, grader), timeout=cfg.eval_timeout,
        )
        reward = bridge.extract_score(raw)
        logger.info("[{}] reward={} (raw={})", task_ref, reward, raw)

        timed_out = bool(agent_result.get("timed_out"))
        status = "fail" if (reward < 1.0 or timed_out) else "pass"
        return TaskResult(
            task_id=task_ref,
            status=status,
            score={"reward": reward},
            session_ids=session_ids,
            metadata={
                "arl_session": arl_session_id,
                "agent_timed_out": timed_out,
                "raw_eval": raw if isinstance(raw, (int, float, str)) else str(raw)[:500],
                "attempt": attempt,
                "reference_staged": ref_staged,
            },
        )
    finally:
        logger.info("[{}] keeping sandbox {} for external cleanup",
                    task_ref, arl_session_id)


async def _run_agent(
    task_ref: str, prompt: str, arl_session_id: str,
    cfg: AleRunConfig, agent_timeout: float, attempt: int,
) -> dict[str, Any]:
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    operations_config: dict[str, Any] = {
        "attach_session": arl_session_id,
        "work_dir": cfg.workspace,
        "delete_on_shutdown": False,
    }
    loop_config = None
    if cfg.max_turns is not None:
        from agentm.core.abi.loop import LoopConfig

        loop_config = LoopConfig(max_turns=cfg.max_turns)
    session_config = AgentSessionConfig(
        cwd=os.getcwd(),
        scenario=cfg.scenario,
        model=cfg.model,
        atom_config_overrides={"operations": operations_config},
        loop_config=loop_config,
        task_class="ale",
        eval_run_id=cfg.exp_id,
        eval_task_id=task_ref,
        experiment={"harness": "ale", "attempt": attempt},
    )

    create_attempts = 0
    while True:
        try:
            session = await AgentSession.create(session_config)
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            create_attempts += 1
            transient = any(m in msg for m in _TRANSIENT_CREATE_MARKERS)
            if not transient or create_attempts >= _MAX_CREATE_ATTEMPTS:
                return {"error": msg[:2000]}
            logger.info("[{}] agent session create retry {}/{}: {}",
                        task_ref, create_attempts, _MAX_CREATE_ATTEMPTS, msg[:160])
            await asyncio.sleep(30)

    logger.info("[{}] agentm trace messages --session {} --format text",
                task_ref, session.session_id)

    timed_out = False
    try:
        coro = session.prompt(prompt)
        if agent_timeout > 0:
            try:
                await asyncio.wait_for(coro, timeout=agent_timeout)
            except TimeoutError:
                timed_out = True
                logger.warning("[{}] agent wall clock ({}s) expired",
                               task_ref, agent_timeout)
        else:
            await coro
    finally:
        await session.shutdown()
    return {"session_id": session.session_id, "timed_out": timed_out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALE_REPO_URL = "https://github.com/rdi-berkeley/agents-last-exam"
_MIRROR_REGISTRY = "pair-cn-guangzhou.cr.volces.com"
_DEFAULT_TASK_IMAGES = f"{_MIRROR_REGISTRY}/opspai"
_DEFAULT_RUNTIME_IMAGE = f"{_MIRROR_REGISTRY}/library/python:3.12"


def _default_ale_repo() -> Path:
    """Resolve ALE checkout: ``$ALE_REPO`` > ``$AGENTM_HOME/ale-repo`` > auto-clone."""
    from_env = os.environ.get("ALE_REPO")
    if from_env:
        return Path(from_env).expanduser()
    from agentm.core.lib.user_config import agentm_home_dir
    repo = agentm_home_dir() / "ale-repo"
    if repo.is_dir():
        return repo
    import subprocess
    logger.info("ale: cloning {} → {}", _ALE_REPO_URL, repo)
    subprocess.run(
        ["git", "clone", "--depth", "1", _ALE_REPO_URL, str(repo)],
        check=True,
    )
    return repo


def _default_data_root() -> Path | None:
    """Resolve ALE data root: ``$ALE_DATA_ROOT`` > ``$AGENTM_HOME/ale-data`` > None."""
    from_env = os.environ.get("ALE_DATA_ROOT")
    if from_env:
        return Path(from_env).expanduser()
    from agentm.core.lib.user_config import agentm_home_dir
    data = agentm_home_dir() / "ale-data"
    return data if data.is_dir() else None


_HF_DATA_REPO = "agents-last-exam/agents-last-exam-data"
_HF_REF_REPO = "agents-last-exam/agents-last-exam-reference"


def _hf_download(repo: str, task: str, dest: Path) -> None:
    """Download one task's tree from a HF dataset repo via the ``hf`` CLI.

    The CLI path avoids a brotli-decoder bug that intermittently corrupts
    ``snapshot_download``; per-task ``--include`` keeps the repo scan tiny.
    """
    import subprocess as _sp
    _sp.run(
        ["hf", "download", repo, "--repo-type", "dataset",
         "--include", f"tasks/{task}/**", "--local-dir", str(dest)],
        check=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=1800,
        env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "0"},
    )


def _ensure_task_data(data_root: Path, tasks: list[str]) -> None:
    """Fetch + assemble data for ``tasks`` only (idempotent, per-task).

    For each task, pull input/software (+ hidden reference) from HF into
    ``data_root/<domain>/<task>/<variant>/…``, skipping ones already assembled.
    Only called for tasks whose baked image is missing from the registry — when
    every image already exists, no data is fetched at all (self-contained run).
    """
    import shutil
    data_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        domain, _, name = task.partition("/")
        td = data_root / domain / name
        if td.is_dir() and any((v / "input").is_dir() for v in td.iterdir() if v.is_dir()):
            continue  # already assembled
        _hf_download(_HF_DATA_REPO, task, data_root / "_hf_data")
        try:
            _hf_download(_HF_REF_REPO, task, data_root / "_hf_ref")
        except Exception:
            pass  # rubric-graded tasks ship no reference
        src = data_root / "_hf_data" / "tasks" / domain / name
        if not src.is_dir():
            continue
        for vdir in sorted(src.iterdir()):
            if not (vdir / "input").is_dir():
                continue
            dest = td / vdir.name
            for sub in ("input", "software"):
                s = vdir / sub
                if s.is_dir():
                    dd = dest / sub
                    if dd.exists():
                        shutil.rmtree(dd)
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(s, dd)
            ref_src = data_root / "_hf_ref" / "tasks" / domain / name / vdir.name / "reference"
            if ref_src.is_dir():
                dr = dest / "reference"
                if dr.exists():
                    shutil.rmtree(dr)
                shutil.copytree(ref_src, dr)


def _variant_of(ale_repo: Path, domain: str, name: str, data_root: Path) -> str:
    """Variant the run stages from: load_task's, else the single assembled dir
    / base. The HF layout can store the instance under a name (e.g. tsla_fy2023)
    that differs from load_task's (e.g. base), so bake under load_task's name."""
    td = data_root / domain / name
    dirs = [p.name for p in td.iterdir() if p.is_dir() and (p / "input").is_dir()] \
        if td.is_dir() else []
    try:
        v = bridge.load_task(ale_repo, domain, name).variant_name
        if v in dirs:
            return v
    except Exception:
        pass
    if "base" in dirs:
        return "base"
    return dirs[0] if dirs else "base"


def _missing_baked_images(tasks: list[str], registry: str, tag: str) -> list[str]:
    """Tasks whose baked data image is NOT already in the registry."""
    import subprocess as _sp
    from .images import data_image_ref
    missing: list[str] = []
    for ref in tasks:
        domain, _, name = ref.partition("/")
        result = _sp.run(
            ["docker", "manifest", "inspect", data_image_ref(registry, domain, name, tag)],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            missing.append(ref)
    return missing


def _ensure_baked_images(
    tasks: list[str], data_root: Path, registry: str, tag: str, ale_repo: Path,
) -> None:
    """Build + push data images for ``tasks`` (already known missing), each with
    its real variant so non-``base`` tasks bake correctly."""
    from .images import build_data_image
    logger.info("ale: building {} missing data images", len(tasks))
    for ref in tasks:
        domain, _, name = ref.partition("/")
        variant = _variant_of(ale_repo, domain, name, data_root)
        if not (data_root / domain / name / variant / "input").is_dir():
            logger.warning("ale: no data for {}, skipping image build", ref)
            continue
        image = build_data_image(
            data_root=data_root, domain=domain, name=name,
            registry=registry, tag=tag, variant=variant, push=True,
        )
        logger.info("ale: built + pushed {}", image)


class AleAdapter:
    name = "ale"
    description = "Agents' Last Exam — real professional tasks graded in ARL sandboxes"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="ale",
            help="Run Agents' Last Exam tasks against ARL sandboxes.",
            add_completion=False,
        )

        @cli.command("list-tasks")
        def list_tasks(
            ale_repo: Annotated[Optional[Path], typer.Option(
                help="agents-last-exam checkout ($ALE_REPO or auto-clone)",
            )] = None,
            data_root: Annotated[Optional[Path], typer.Option(
                help="Task data root; when set, only tasks with local input/ data are shown",
            )] = None,
        ) -> None:
            """List available tasks (domain/task)."""
            if ale_repo is None:
                ale_repo = _default_ale_repo()
            for domain, name in bridge.discover_tasks(ale_repo):
                if data_root is not None:
                    if not (data_root / domain / name).is_dir():
                        continue
                typer.echo(f"{domain}/{name}")

        @cli.command("build-images")
        def build_images(
            task: Annotated[list[str], typer.Option(
                "--task", "-t", help="Task ref domain/name (repeatable)",
            )],
            data_root: Annotated[Optional[Path], typer.Option(
                help="Task data root ($ALE_DATA_ROOT, needed once at build time)",
            )] = None,
            registry: Annotated[str, typer.Option(
                help="Target registry prefix, e.g. docker.io namespace",
            )] = "opspai",
            tag: Annotated[str, typer.Option(help="Image tag")] = "latest",
            base_image: Annotated[str, typer.Option(
                help="Base image for the data image (needs sh+cp+find)",
            )] = "busybox:latest",
            push: Annotated[bool, typer.Option(help="Push after building")] = False,
        ) -> None:
            """Bake each task's data into ONE private image.

            Builds <registry>/ale-<domain>-<task>-data:<tag> containing
            input/ + software/ + reference/. At run time it becomes a hidden
            private container; the agent's runtime image stays shared
            (`run --task-images <registry> --image <runtime>`), so warm
            pools remain reusable and runs upload nothing.
            """
            from .images import build_data_image
            if not task:
                typer.echo("no tasks given (-t domain/name)", err=True)
                raise typer.Exit(2)
            if data_root is None:
                resolved = _default_data_root()
                if resolved is None:
                    typer.echo("no --data-root and $ALE_DATA_ROOT not set", err=True)
                    raise typer.Exit(2)
                data_root = resolved
            failed: list[str] = []
            for ref in task:
                domain, _, name = ref.partition("/")
                try:
                    image = build_data_image(
                        data_root=data_root.resolve(),
                        domain=domain, name=name,
                        registry=registry, tag=tag,
                        base_image=base_image, push=push,
                    )
                    typer.echo(f"built {ref}: {image}")
                except Exception as e:  # noqa: BLE001
                    failed.append(ref)
                    typer.echo(f"FAILED {ref}: {e}", err=True)
            if failed:
                raise typer.Exit(1)

        @cli.command()
        def run(
            task: Annotated[Optional[list[str]], typer.Option(
                "--task", "-t", help="Task ref domain/name (repeatable); omit for all",
            )] = None,
            domain: Annotated[Optional[list[str]], typer.Option(
                "--domain", "-d", help="Filter by domain (repeatable)",
            )] = None,
            ale_repo: Annotated[Optional[Path], typer.Option(
                help="agents-last-exam checkout ($ALE_REPO or auto-clone)",
            )] = None,
            task_images: Annotated[Optional[str], typer.Option(
                help="Baked mode: registry prefix of the per-task data images "
                     "built by `build-images` (zero upload; data mounts as a "
                     "hidden private container)",
            )] = _DEFAULT_TASK_IMAGES,
            images_tag: Annotated[str, typer.Option(help="Baked mode: image tag")] = "latest",
            data_root: Annotated[Optional[Path], typer.Option(
                help="Upload mode: host task-data root ($ALE_DATA_ROOT) "
                     "<root>/<domain>/<task>/<variant>/{input,software,reference}",
            )] = None,
            image: Annotated[Optional[str], typer.Option(
                help="Sandbox runtime image the agent lives in "
                     "(shared across tasks; must carry the task's system deps)",
            )] = _DEFAULT_RUNTIME_IMAGE,
            model: Annotated[Optional[str], typer.Option(help="Model profile")] = None,
            scenario: Annotated[str, typer.Option(help="AgentM scenario")] = "arl",
            profile: Annotated[str, typer.Option(help="ARL warm-pool profile")] = "default",
            concurrency: Annotated[int, typer.Option("-j", help="Parallel tasks")] = 1,
            attempts: Annotated[int, typer.Option("-n", help="Attempts per task (pass@k)")] = 1,
            agent_timeout: Annotated[float, typer.Option(
                help="Agent wall clock seconds; 0 = task card timeout",
            )] = 0.0,
            eval_timeout: Annotated[float, typer.Option(help="Grader timeout seconds")] = 7200.0,
            max_turns: Annotated[Optional[int], typer.Option(help="Agent max turns")] = None,
            keep_sandbox: Annotated[bool, typer.Option(help="Keep sandboxes after run")] = True,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
            arl_context: Annotated[Optional[str], typer.Option(
                help="ARL config context from ~/.config/arl/config.yaml "
                     "(default: the CLI's current_context)",
            )] = None,
            no_eval: Annotated[bool, typer.Option(
                "--no-eval", help="Agent phase only: solve + save output/, defer "
                "grading (use when the hidden reference isn't available yet). "
                "Grade later with `agentm eval ale grade`.",
            )] = False,
            outputs_dir: Annotated[Optional[Path], typer.Option(
                help="Where to save agent output archives for --no-eval "
                     "(default: <exp>/outputs)",
            )] = None,
        ) -> None:
            """Run one or more ALE tasks end to end (or agent-only with --no-eval)."""
            if ale_repo is None:
                ale_repo = _default_ale_repo()
            if not task:
                domains = set(domain) if domain else None
                task = [f"{d}/{n}" for d, n in bridge.discover_tasks(ale_repo)
                        if not d.startswith("demo")
                        and (domains is None or d in domains)]
            if data_root is None and task_images is None:
                data_root = _default_data_root()

            if task_images is not None:
                # Self-contained: pull pre-built data images straight from the
                # registry. Only fetch data + build for tasks whose image is
                # missing — when every image exists (the common case), no local
                # data root is needed or touched.
                missing = _missing_baked_images(list(task), task_images, images_tag)
                if missing:
                    if data_root is None:
                        from agentm.core.lib.user_config import agentm_home_dir
                        data_root = agentm_home_dir() / "ale-data"
                    logger.info("ale: {} image(s) missing → fetching data + building",
                                len(missing))
                    _ensure_task_data(data_root, missing)
                    _ensure_baked_images(
                        missing, data_root, task_images, images_tag, ale_repo,
                    )

            # Gateway/auth resolve exactly like the `arl` CLI: env vars, then
            # the active context in ~/.config/arl/config.yaml. ARL_CONTEXT is
            # read by every GatewayClient — the harness sandbox here AND the
            # agent_env attach inside the AgentM session. An explicit
            # --arl-context is authoritative: drop stale env overrides that
            # would silently win over the requested context.
            if arl_context:
                os.environ["ARL_CONTEXT"] = arl_context
                os.environ.pop("ARL_GATEWAY_URL", None)
                os.environ.pop("ARL_API_KEY", None)
            from arl.config import resolve_from_config
            gw_url, gw_key = resolve_from_config()
            typer.echo(f"ARL gateway: {gw_url} (auth: {'yes' if gw_key else 'no'})")

            # Root AgentSessions need an explicit model profile: AgentSession
            # does not fall back to config.toml default_model on its own (the
            # chat CLI resolves it before session creation — mirror that).
            if model is None:
                from agentm.core.lib.user_config import load_user_config
                model = load_user_config().default_model
                if model is None:
                    typer.echo(
                        "no --model and no default_model in config.toml", err=True,
                    )
                    raise typer.Exit(2)
                typer.echo(f"model profile: {model} (config.toml default)")

            with experiment_context(
                "ale", model=model, exp_id=exp_id,
                image=image, task_images=task_images, images_tag=images_tag,
                scenario=scenario, tasks=list(task),
                attempts=attempts, agent_timeout=agent_timeout,
            ) as exp:
                cfg = AleRunConfig(
                    ale_repo=ale_repo.resolve(),
                    data_root=data_root.resolve() if data_root else None,
                    image=image,
                    task_images=task_images,
                    images_tag=images_tag,
                    scenario=scenario,
                    model=model,
                    exp_id=exp.exp_id,
                    out_dir=exp.output_dir,
                    profile=profile,
                    agent_timeout=agent_timeout,
                    eval_timeout=eval_timeout,
                    max_turns=max_turns,
                    keep_sandbox=keep_sandbox,
                    defer_grading=no_eval,
                    outputs_dir=(outputs_dir.resolve() if outputs_dir
                                 else exp.output_dir / "outputs"),
                )
                jobs = [(ref, k) for ref in task for k in range(attempts)]
                results: list[TaskResult] = []
                if concurrency <= 1 or len(jobs) == 1:
                    for ref, k in jobs:
                        result = run_one(ref, cfg, attempt=k)
                        exp.record_result(result.to_dict())
                        results.append(result)
                        typer.echo(_fmt_line(result))
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = {
                            pool.submit(run_one, ref, cfg, k): (ref, k)
                            for ref, k in jobs
                        }
                        for fut in as_completed(futures):
                            result = fut.result()
                            exp.record_result(result.to_dict())
                            results.append(result)
                            typer.echo(_fmt_line(result))

                if no_eval:
                    solved = sum(1 for r in results if r.status == "solved")
                    errs = sum(1 for r in results if r.status == "error")
                    summary = (
                        f"ALE agent-phase: {solved}/{len(results)} solved, "
                        f"{errs} error. Outputs → {cfg.outputs_dir}. "
                        f"Grade later: agentm eval ale grade --exp-id {exp.exp_id}"
                    )
                    exp.finish(status="completed", summary={
                        "solved": solved, "error": errs, "total": len(results),
                        "phase": "agent", "outputs_dir": str(cfg.outputs_dir),
                    })
                else:
                    passed, mean_reward = _summarize(results)
                    summary = (
                        f"ALE: {passed}/{len(results)} pass, "
                        f"mean reward {mean_reward:.3f}"
                    )
                    exp.finish(status="completed", summary={
                        "pass": passed, "total": len(results),
                        "mean_reward": mean_reward,
                    })
                typer.echo(summary)

        @cli.command()
        def grade(
            task: Annotated[list[str], typer.Option(
                "--task", "-t", help="Task ref domain/name (repeatable)",
            )],
            ale_repo: Annotated[Optional[Path], typer.Option(
                help="agents-last-exam checkout ($ALE_REPO or auto-clone)",
            )] = None,
            outputs_dir: Annotated[Path, typer.Option(
                help="Dir of saved agent output archives (from `run --no-eval`)",
            )] = Path("."),
            reference_root: Annotated[Optional[Path], typer.Option(
                help="Host task-data root holding reference/ ($ALE_DATA_ROOT) "
                     "<root>/<domain>/<task>/<variant>/reference",
            )] = None,
            image: Annotated[str, typer.Option(
                help="Runtime image for the grading sandbox (match the task's deps)",
            )] = _DEFAULT_RUNTIME_IMAGE,
            data_root: Annotated[Optional[Path], typer.Option(
                help="Optional: also restage input/ (for graders that read input)",
            )] = None,
            concurrency: Annotated[int, typer.Option("-j")] = 2,
            attempts: Annotated[int, typer.Option("-n")] = 1,
            eval_timeout: Annotated[float, typer.Option()] = 7200.0,
            arl_context: Annotated[Optional[str], typer.Option()] = None,
            exp_id: Annotated[Optional[str], typer.Option()] = None,
        ) -> None:
            """Grade saved agent outputs (phase 2 of --no-eval) once reference exists."""
            if not task:
                typer.echo("no tasks given (-t domain/name)", err=True)
                raise typer.Exit(2)
            if ale_repo is None:
                ale_repo = _default_ale_repo()
            if reference_root is None:
                reference_root = _default_data_root() or Path(".")
            if arl_context:
                os.environ["ARL_CONTEXT"] = arl_context
                os.environ.pop("ARL_GATEWAY_URL", None)
                os.environ.pop("ARL_API_KEY", None)
            from arl.config import resolve_from_config
            gw_url, _ = resolve_from_config()
            typer.echo(f"ARL gateway: {gw_url}")

            with experiment_context(
                "ale-grade", exp_id=exp_id, tasks=list(task),
            ) as exp:
                cfg = AleRunConfig(
                    ale_repo=ale_repo.resolve(), scenario="", model=None,
                    exp_id=exp.exp_id, out_dir=exp.output_dir, image=image,
                    data_root=data_root.resolve() if data_root else None,
                    reference_root=reference_root.resolve(),
                    outputs_dir=outputs_dir.resolve(),
                    eval_timeout=eval_timeout,
                )
                jobs = [(ref, k) for ref in task for k in range(attempts)]
                results: list[TaskResult] = []
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {pool.submit(grade_one, ref, cfg, k): (ref, k)
                               for ref, k in jobs}
                    for fut in as_completed(futures):
                        r = fut.result()
                        exp.record_result(r.to_dict())
                        results.append(r)
                        typer.echo(_fmt_line(r))
                passed, mean = _summarize(results)
                exp.finish(status="completed", summary={
                    "pass": passed, "total": len(results), "mean_reward": mean})
                typer.echo(f"ALE grade: {passed}/{len(results)} pass, mean reward {mean:.3f}")

        return cli


def _fmt_line(r: TaskResult) -> str:
    reward = r.score.get("reward")
    reward_txt = f"{reward:.3f}" if isinstance(reward, (int, float)) else "-"
    sid = r.session_ids[0][:12] if r.session_ids else "-"
    extra = f" ({r.error[:120]})" if r.error else ""
    return f"[{r.status:>5}] {r.task_id}  reward={reward_txt}  session={sid}  {r.latency_ms/1000:.0f}s{extra}"


def _summarize(results: list[TaskResult]) -> tuple[int, float]:
    """(#pass, mean reward) over a graded result set."""
    passed = sum(1 for r in results if r.status == "pass")
    rewards = [float(r.score.get("reward", 0.0)) for r in results]
    return passed, (sum(rewards) / len(rewards) if rewards else 0.0)


register("ale", AleAdapter.description, AleAdapter)
