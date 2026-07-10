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
    "pool at maximum capacity",
    "429",
    "503",
    "no ready sandbox",
)
_MAX_CREATE_ATTEMPTS = 20


@dataclass
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
    keep_sandbox: bool = False
    remote_root: str = bridge.DEFAULT_REMOTE_ROOT
    workspace: str = bridge.DEFAULT_WORKSPACE

    @property
    def baked(self) -> bool:
        return self.task_images is not None


# ---------------------------------------------------------------------------
# Single-task pipeline
# ---------------------------------------------------------------------------

def run_one(task_ref: str, cfg: AleRunConfig, attempt: int = 0) -> TaskResult:
    """Run one ``domain/task`` end to end. Runs in a worker thread with its
    own event loop, mirroring the sandbox adapter's concurrency model."""
    started = time.monotonic()
    domain, _, name = task_ref.partition("/")
    log_path = cfg.out_dir / "artifacts" / f"{domain}__{name}.log"
    thread_id = threading.get_ident()
    sink_id = logger.add(
        str(log_path),
        filter=lambda record: record["thread"].id == thread_id,
        enqueue=True, backtrace=False, diagnose=False,
    )
    try:
        result = asyncio.run(_run_one_async(task_ref, cfg, attempt))
    except Exception as e:  # noqa: BLE001
        logger.exception("ALE task {} crashed", task_ref)
        result = TaskResult(task_id=task_ref, status="error", error=str(e)[:2000])
    finally:
        logger.remove(sink_id)
    result.latency_ms = int((time.monotonic() - started) * 1000)
    score_file = cfg.out_dir / "artifacts" / f"{domain}__{name}.score.json"
    score_file.write_text(json.dumps(result.to_dict(), indent=2))
    return result


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
        data_image = data_image_ref(cfg.task_images, domain, name, cfg.images_tag)
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
            await bridge.stage_task_input(grader, task, cfg.data_root, cfg.remote_root)
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
        if agent_result.get("error"):
            return TaskResult(
                task_id=task_ref, status="error",
                error=f"agent: {agent_result['error']}",
                session_ids=[s for s in (agent_session_id,) if s],
                metadata={"arl_session": arl_session_id},
            )

        # -- reference + grade (post-agent) --------------------------------
        if cfg.baked:
            await bridge.stage_reference_from_container(
                sandbox, task, cfg.remote_root,
            )
        else:
            await bridge.stage_task_reference(grader, task, cfg.data_root, cfg.remote_root)
        evaluate_fn = task.module.evaluate
        raw = await asyncio.wait_for(
            evaluate_fn(task.task_obj, grader), timeout=cfg.eval_timeout,
        )
        reward = bridge.extract_score(raw)
        logger.info("[{}] reward={} (raw={})", task_ref, reward, raw)

        status = "pass" if reward >= 1.0 else "fail"
        if agent_result.get("timed_out"):
            status = "fail"
        return TaskResult(
            task_id=task_ref,
            status=status,
            score={"reward": reward},
            session_ids=[s for s in (agent_session_id,) if s],
            metadata={
                "arl_session": arl_session_id,
                "agent_timed_out": bool(agent_result.get("timed_out")),
                "raw_eval": raw if isinstance(raw, (int, float, str)) else str(raw)[:500],
                "attempt": attempt,
            },
        )
    finally:
        if cfg.keep_sandbox:
            logger.info("[{}] keeping sandbox {} (--keep-sandbox)",
                        task_ref, arl_session_id)
        else:
            try:
                await asyncio.to_thread(sandbox.delete_sandbox)
            except Exception as e:  # noqa: BLE001
                logger.warning("[{}] sandbox delete failed: {}", task_ref, e)


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
    session_kwargs: dict[str, Any] = dict(
        cwd=os.getcwd(),
        scenario=cfg.scenario,
        model=cfg.model,
        atom_config_overrides={"operations": operations_config},
        task_class="ale",
        eval_run_id=cfg.exp_id,
        eval_task_id=task_ref,
        experiment={"harness": "ale", "attempt": attempt},
    )
    if cfg.max_turns is not None:
        from agentm.core.abi.loop import LoopConfig
        session_kwargs["loop_config"] = LoopConfig(max_turns=cfg.max_turns)
    session_config = AgentSessionConfig(**session_kwargs)

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

    timed_out = False
    try:
        coro = session.prompt(prompt)
        if agent_timeout > 0:
            try:
                await asyncio.wait_for(coro, timeout=agent_timeout)
            except asyncio.TimeoutError:
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
            ale_repo: Annotated[Path, typer.Option(help="agents-last-exam checkout")],
            data_root: Annotated[Optional[Path], typer.Option(
                help="Task data root; when set, only tasks with local input/ data are shown",
            )] = None,
        ) -> None:
            """List available tasks (domain/task)."""
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
            data_root: Annotated[Path, typer.Option(
                help="Task data root (needed once, at build time)",
            )],
            registry: Annotated[str, typer.Option(
                help="Target registry prefix, e.g. docker.io namespace",
            )],
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
            task: Annotated[list[str], typer.Option(
                "--task", "-t", help="Task ref domain/name (repeatable)",
            )],
            ale_repo: Annotated[Path, typer.Option(help="agents-last-exam checkout")],
            task_images: Annotated[Optional[str], typer.Option(
                help="Baked mode: registry prefix of the per-task data images "
                     "built by `build-images` (zero upload; data mounts as a "
                     "hidden private container)",
            )] = None,
            images_tag: Annotated[str, typer.Option(help="Baked mode: image tag")] = "latest",
            data_root: Annotated[Optional[Path], typer.Option(
                help="Upload mode: host task-data root "
                     "<root>/<domain>/<task>/<variant>/{input,software,reference}",
            )] = None,
            image: Annotated[Optional[str], typer.Option(
                help="Sandbox runtime image the agent lives in "
                     "(shared across tasks; must carry the task's system deps)",
            )] = None,
            model: Annotated[Optional[str], typer.Option(help="Model profile")] = None,
            scenario: Annotated[str, typer.Option(help="AgentM scenario")] = "ale:arl",
            profile: Annotated[str, typer.Option(help="ARL warm-pool profile")] = "default",
            concurrency: Annotated[int, typer.Option("-j", help="Parallel tasks")] = 1,
            attempts: Annotated[int, typer.Option("-n", help="Attempts per task (pass@k)")] = 1,
            agent_timeout: Annotated[float, typer.Option(
                help="Agent wall clock seconds; 0 = task card timeout",
            )] = 0.0,
            eval_timeout: Annotated[float, typer.Option(help="Grader timeout seconds")] = 7200.0,
            max_turns: Annotated[Optional[int], typer.Option(help="Agent max turns")] = None,
            keep_sandbox: Annotated[bool, typer.Option(help="Keep sandboxes for debugging")] = False,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
            arl_context: Annotated[Optional[str], typer.Option(
                help="ARL config context from ~/.config/arl/config.yaml "
                     "(default: the CLI's current_context)",
            )] = None,
        ) -> None:
            """Run one or more ALE tasks end to end."""
            if not task:
                typer.echo("no tasks given (-t domain/name)", err=True)
                raise typer.Exit(2)
            if image is None or (task_images is None and data_root is None):
                typer.echo(
                    "need --image <runtime> plus a data mode: "
                    "--task-images <registry-prefix> (baked data images) "
                    "or --data-root <dir> (upload)", err=True,
                )
                raise typer.Exit(2)

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

                passed = sum(1 for r in results if r.status == "pass")
                rewards = [
                    float(r.score.get("reward", 0.0)) for r in results
                ]
                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                summary = (
                    f"ALE: {passed}/{len(results)} pass, "
                    f"mean reward {mean_reward:.3f}"
                )
                exp.finish(status="completed", summary={
                    "pass": passed, "total": len(results),
                    "mean_reward": mean_reward,
                })
                typer.echo(summary)

        return cli


def _fmt_line(r: TaskResult) -> str:
    reward = r.score.get("reward")
    reward_txt = f"{reward:.3f}" if isinstance(reward, (int, float)) else "-"
    extra = f" ({r.error[:120]})" if r.error else ""
    return f"[{r.status:>5}] {r.task_id}  reward={reward_txt}  {r.latency_ms/1000:.0f}s{extra}"


register("ale", AleAdapter.description, AleAdapter)
