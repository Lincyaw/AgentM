"""Core sandbox benchmark runner — extracted from bench.py.

Contains all the run+eval logic for Docker-sandbox-based benchmarks.
No CLI commands; the SandboxAdapter in __init__.py provides those.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, cast

import typer
from loguru import logger

from .bench import TaskSpec, eval_image_name, image_name

try:
    import yaml as _yaml
except ImportError:  # noqa: S110
    _yaml = None  # type: ignore[assignment]


DEFAULT_IMAGE_REGISTRY = os.environ.get(
    "AGENTM_EVAL_IMAGE_REGISTRY", "pair-cn-guangzhou.cr.volces.com"
)
DEFAULT_REMOTE_TERMINAL_BENCH_SCENARIO = "arl"


def _resolve_adapter_repo(adapter: Any, bench: str) -> str:
    """Auto-resolve task repo from adapter defaults: $ENV > $AGENTM_HOME/bench-repos > clone."""
    env_var = getattr(adapter, "DEFAULT_REPO_ENV", None)
    if env_var:
        from_env = os.environ.get(env_var)
        if from_env:
            p = Path(from_env).expanduser().resolve()
            subdir = getattr(adapter, "DEFAULT_REPO_SUBDIR", "")
            return str(p / subdir) if subdir else str(p)

    repo_url = getattr(adapter, "DEFAULT_REPO_URL", None)
    if not repo_url:
        typer.echo(f"Error: --repo required for bench {bench!r}.", err=True)
        raise typer.Exit(1)

    from agentm.core.lib.user_config import agentm_home_dir
    slug = bench.replace("_", "-").replace(":", "-")
    repo_dir = agentm_home_dir() / "bench-repos" / slug
    if not repo_dir.is_dir():
        logger.info("bench: cloning {} → {}", repo_url, repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
        )
    subdir = getattr(adapter, "DEFAULT_REPO_SUBDIR", "")
    return str(repo_dir / subdir) if subdir else str(repo_dir)


def _filter_supported_kwargs(
    callable_obj: object, kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        sig = inspect.signature(cast(Callable[..., Any], callable_obj))
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def _safe_experiment_id(raw: str, max_len: int = 63) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    if len(value) > max_len:
        value = value[:max_len].rstrip("-")
    return value or f"exp-{uuid.uuid4().hex[:8]}"


def _default_run_id() -> str:
    return _safe_experiment_id(
        os.environ.get("AGENTM_BENCH_RUN_ID")
        or f"r{datetime.now(UTC):%m%d%H%M%S}-{uuid.uuid4().hex[:4]}",
        max_len=24,
    )


def _experiment_ids(
    prefix: str, model: str, run_id: str, task: str, attempt_idx: int | None,
) -> tuple[str, str]:
    import hashlib

    attempt = f"a{attempt_idx}" if attempt_idx is not None else "a0"
    task_tag = hashlib.sha256(task.encode("utf-8")).hexdigest()[:10]
    suffix = f"{run_id}-{attempt}-{task_tag}-{model}-{task}"
    return (
        _safe_experiment_id(f"{prefix}-{suffix}"),
        _safe_experiment_id(f"eval-{suffix}"),
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def build_base_images(base_dir: Path) -> None:
    import glob as globmod

    for pattern in ["Dockerfile.*-base", "Dockerfile.*_base"]:
        for dockerfile_path in globmod.glob(str(base_dir / pattern)):
            dockerfile = Path(dockerfile_path)
            base_name = (
                dockerfile.name.replace("Dockerfile.", "")
                .replace("-base", "").replace("_base", "")
            )
            tag = f"tb/{base_name}:v0"
            if subprocess.run(
                ["docker", "image", "inspect", tag], capture_output=True,
            ).returncode == 0:
                typer.echo(f"  Base {tag} exists, skip")
                continue
            typer.echo(f"  Building {tag} ...")
            subprocess.run(
                ["docker", "build", "-f", str(dockerfile), "-t", tag,
                 str(base_dir), "-q"],
                check=True, capture_output=True,
            )


def generate_skaffold(
    tasks: list[TaskSpec], registry: str, prefix: str, tag: str,
) -> dict:
    artifacts = []
    for task in tasks:
        img = image_name(task.name, registry, prefix, tag).rsplit(":", 1)[0]
        # Harbor-family tasks (e.g. senior-swe) keep the whole build recipe
        # under environment/ -- the Dockerfile plus any files it COPYs. Use
        # that directory as the build context so those COPYs resolve (Docker
        # resolves COPY relative to the context, not the Dockerfile). tb1 tasks
        # with a root Dockerfile keep the task dir as context.
        env_dir = Path(task.path) / "environment"
        context = str(env_dir) if (env_dir / "Dockerfile").is_file() else task.path
        artifacts.append({"image": img, "context": context})
    return {
        "apiVersion": "skaffold/v4beta11",
        "kind": "Config",
        "metadata": {"name": f"{prefix}-images"},
        "build": {
            "tagPolicy": {"envTemplate": {"template": tag}},
            "local": {"concurrency": 4, "useBuildkit": True, "push": False},
            "artifacts": artifacts,
        },
    }


def task_image(
    adapter: Any, task: TaskSpec, registry: str, prefix: str, tag: str,
    *, source_images: bool = False,
) -> str:
    if source_images:
        src = adapter.get_source_image(task) or ""
        if not src:
            raise ValueError(f"task {task.name} declares no source image")
        return f"{registry}/{src}" if registry else src
    return adapter.get_image(task, registry, prefix, tag)


def get_eval_image(
    adapter: Any, task: TaskSpec, registry: str, prefix: str, tag: str,
) -> str:
    getter = getattr(adapter, "get_eval_image", None)
    if callable(getter):
        return str(getter(task, registry, prefix, tag))
    return eval_image_name(task.name, registry, prefix, tag)


def build_eval_images(
    tasks: list[TaskSpec], *, adapter: Any,
    registry: str, prefix: str, tag: str, push: bool,
) -> None:
    build_eval_image_fn = getattr(adapter, "build_eval_image", None)
    if not callable(build_eval_image_fn):
        typer.echo(
            f"Error: {type(adapter).__name__} does not support private eval image builds.",
            err=True,
        )
        raise typer.Exit(1)
    for task in tasks:
        base = adapter.get_image(task, registry, prefix, tag)
        eval_img = get_eval_image(adapter, task, registry, prefix, tag)
        typer.echo(f"  Building eval {eval_img} <- {base}")
        try:
            build_eval_image_fn(task, base_image=base, eval_image=eval_img, push=push)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Error: failed to build eval image for {task.name}: {e}", err=True)
            raise typer.Exit(1) from e


def resolve_source(
    repo: Path | None,
    source: str | None,
    bench: str,
    adapter: Any = None,
) -> str:
    if bench.startswith("swebench"):
        if source:
            return source
        defaults = {
            "swebench-verified": "princeton-nlp/SWE-bench_Verified",
            "swebench-pro": "ScaleAI/SWE-bench_Pro",
        }
        return defaults.get(bench, "princeton-nlp/SWE-bench_Verified")
    if repo is not None:
        return str(repo.expanduser().resolve())
    if source:
        return str(Path(source).expanduser().resolve())
    if adapter is not None and hasattr(adapter, "DEFAULT_REPO_URL"):
        return _resolve_adapter_repo(adapter, bench)
    typer.echo("Error: --repo required for this bench format.", err=True)
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Core run+eval logic
# ---------------------------------------------------------------------------

_TRANSIENT_CREATE_MARKERS = (
    "pool at maximum capacity", "queued for", "context deadline exceeded",
    "Gateway error (429)", "Gateway error (500)", "Gateway error (503)",
    "Server disconnected", "Connection reset", "Connection refused",
    "ReadTimeout", "timed out",
)
_MAX_CREATE_ATTEMPTS = 20

_active_agent_interrupts: set[tuple[Any, Any]] = set()
_active_agent_lock = threading.Lock()
_active_eval_sessions: set[Any] = set()
_active_eval_lock = threading.Lock()

@dataclass(frozen=True, slots=True)
class RunEvalConfig:
    adapter: Any
    model: str
    registry: str
    prefix: str
    tag: str
    eval_timeout: int
    agent_timeout: int
    scenario: str
    run_id: str
    prompt_prefix: str
    exp_id: str = ""
    bench: str = ""
    source_images: bool = False


@dataclass(frozen=True, slots=True)
class RunEvalJob:
    task: TaskSpec
    out: Path
    attempt_slot: int
    experiment_attempt_idx: int | None


def run_agent_session(
    job: RunEvalJob, config: RunEvalConfig, image: str,
    agent_exp_id: str, prompt: str,
) -> dict[str, Any]:
    import asyncio

    async def _run() -> dict[str, Any]:
        from agentm.core.abi.session_config import AgentSessionConfig
        from agentm.core.runtime.session import AgentSession

        adapter = config.adapter
        task = job.task

        operations_config: dict[str, Any] = {
            "image": image,
            "experiment_id": agent_exp_id,
            "delete_on_shutdown": False,
        }
        eval_getter = getattr(adapter, "get_eval_image", None)
        if callable(eval_getter):
            operations_config["private_containers"] = [{
                "name": "eval",
                "image": str(eval_getter(task, config.registry, config.prefix, config.tag)),
                "mountWorkspace": True,
                "workspaceMountPath": "/app",
                "workspaceAccess": "readWrite",
            }]
        else:
            operations_config["private_containers"] = []

        # Task sidecars (e.g. an isolated game/referee server) run as
        # additional private containers in the same pod; compose-style
        # service hostnames collapse to localhost there, so adapters also
        # supply main-container env overrides pointing at localhost.
        sidecar_getter = getattr(adapter, "get_sidecar_containers", None)
        if callable(sidecar_getter):
            operations_config["private_containers"].extend(
                sidecar_getter(task, config.registry, config.prefix, config.tag)
            )
        main_env_getter = getattr(adapter, "get_main_env", None)
        main_env = main_env_getter(task) if callable(main_env_getter) else {}
        if main_env:
            operations_config["config_env"] = {"vars": main_env}
        resources_getter = getattr(adapter, "get_resources", None)
        if callable(resources_getter):
            operations_config.update(resources_getter(task))

        session_config = AgentSessionConfig(
            cwd=os.getcwd(),
            scenario=config.scenario,
            model=config.model,
            atom_config_overrides={"operations": operations_config},
            task_class=config.bench,
            eval_run_id=config.exp_id or config.run_id,
            eval_task_id=task.name,
            experiment={"harness": "bench", "attempt": job.attempt_slot},
        )
        create_attempts = 0
        while True:
            try:
                session = await AgentSession.create(session_config)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                transient = any(marker in msg for marker in _TRANSIENT_CREATE_MARKERS)
                create_attempts += 1
                if not transient or create_attempts >= _MAX_CREATE_ATTEMPTS:
                    raise
                logger.info(
                    "agent env for {} not ready (attempt {}/{}), retrying in 30s: {}",
                    task.name, create_attempts, _MAX_CREATE_ATTEMPTS, msg[:160],
                )
                await asyncio.sleep(30)

        logger.info("{}: agentm trace messages --session {} --format text",
                    job.task.name, session.session_id)

        interrupt = asyncio.Event()
        loop = asyncio.get_running_loop()
        with _active_agent_lock:
            _active_agent_interrupts.add((loop, interrupt))
        timed_out = False
        messages: list[Any] = []
        try:
            coro = session.prompt(prompt, signal=interrupt)
            if config.agent_timeout > 0:
                try:
                    messages = await asyncio.wait_for(coro, timeout=config.agent_timeout)
                except TimeoutError:
                    timed_out = True
            else:
                messages = await coro
        finally:
            with _active_agent_lock:
                _active_agent_interrupts.discard((loop, interrupt))
            await session.shutdown()

        tool_calls = sum(
            len(m.content) for m in messages if getattr(m, "role", "") == "tool_result"
        )
        return {"session_id": session.session_id, "timed_out": timed_out, "tools": str(tool_calls)}

    try:
        return asyncio.run(_run())
    except Exception as e:  # noqa: BLE001
        logger.debug("agent session for {} failed: {}", job.task.name, e)
        return {"error": str(e)}


def run_and_eval_one(job: RunEvalJob, config: RunEvalConfig) -> dict[str, Any]:
    log_file = job.out / f"{job.task.name}.log"
    thread_id = threading.get_ident()
    sink_id = logger.add(
        str(log_file),
        filter=lambda record: record["thread"].id == thread_id,
        enqueue=True, backtrace=False, diagnose=False,
    )
    try:
        return _run_and_eval_one_inner(job, config)
    finally:
        logger.remove(sink_id)


def _run_and_eval_one_inner(
    job: RunEvalJob, config: RunEvalConfig,
) -> dict[str, Any]:
    import arl

    task = job.task
    adapter = config.adapter
    name = task.name
    state_file = job.out / f"{name}.session.json"
    score_file = job.out / f"{name}.score.json"
    agent_exp_id = _experiment_ids(
        config.prefix, config.model, config.run_id,
        name, job.experiment_attempt_idx,
    )[0]

    img = task_image(
        adapter, task, config.registry, config.prefix, config.tag,
        source_images=config.source_images,
    )

    if score_file.is_file():
        cached_scores = json.loads(score_file.read_text())
        return {"task": name, "status": "done", "tools": cached_scores.get("tools", "?"), **cached_scores}

    session_id = None
    agent_timed_out = False
    tools_count = "?"
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        session_id = state.get("session_id")
        agent_timed_out = bool(state.get("agent_timed_out", False))
        tools_count = str(state.get("tools", "?"))

    if session_id is None:
        prompt = task.prompt
        if not prompt:
            return {"task": name, "status": "no_instruction"}
        if config.prompt_prefix:
            prompt = f"{config.prompt_prefix}\n\n{prompt}"

        try:
            client = arl.GatewayClient()
            client.delete_experiment(agent_exp_id)
            client.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("orphan cleanup {}: {}", agent_exp_id, e)

        agent = run_agent_session(job, config, img, agent_exp_id, prompt)
        if "error" in agent:
            return {"task": name, "status": "agent_failed", "error": agent["error"]}
        session_id = agent["session_id"]
        agent_timed_out = agent["timed_out"]
        tools_count = agent["tools"]
        state_file.write_text(json.dumps({
            "session_id": session_id, "agent_timed_out": agent_timed_out, "tools": tools_count,
        }, ensure_ascii=False))

    client = arl.GatewayClient()
    try:
        agent_arl_sessions = client.list_experiment_sessions(agent_exp_id)
    except Exception as e:
        client.close()
        return {"task": name, "status": "eval_create_failed", "tools": tools_count, "error": f"list sessions: {e}"}
    live = [s for s in agent_arl_sessions if getattr(s, "deleted_at", None) is None]
    if live:
        arl_session_id = live[0].id
    elif agent_arl_sessions:
        source = max(agent_arl_sessions, key=lambda s: str(getattr(s, "created_at", "") or ""))
        try:
            new_session = client.create_session(
                img, idle_timeout_seconds=7200, allocation_timeout_seconds=600,
            )
            replay = client.replay_from(new_session.id, source.id)
            logger.info("replay {}: {} steps from {} into {}, {} errors",
                        name, replay.stepsReplayed, source.id, new_session.id, replay.errors)
            arl_session_id = new_session.id
        except Exception as e:
            client.close()
            return {"task": name, "status": "eval_create_failed", "tools": tools_count,
                    "error": f"replay failed: {e}"}
    else:
        client.close()
        return {"task": name, "status": "eval_create_failed", "tools": tools_count,
                "error": "no session found under experiment"}

    # Long verifiers (LLM-judge pipelines, compile-and-test suites) run as a
    # single execute call; the HTTP timeout must cover the adapter's effective
    # eval timeout or the client gives up while the operation is still running.
    timeout_for = getattr(adapter, "eval_timeout_for", None)
    eval_timeout = (
        timeout_for(task, config.eval_timeout) if callable(timeout_for)
        else config.eval_timeout
    )
    # A long verifier runs as one synchronous execute POST that outlives the
    # LB/proxy in front of the gateway, which resets or times out the
    # connection. ARL execute operations are idempotent, so the SDK recovers
    # internally (recover=True) by polling the operation to completion; no
    # client-side wrapper is needed. The adapter bounds that poll via
    # recover_timeout on the long steps.
    session = arl.SandboxSession.attach(arl_session_id, timeout=eval_timeout + 120)
    with _active_eval_lock:
        _active_eval_sessions.add(session)
    eval_session = session
    scores: dict[str, Any] | None = None
    eval_error: Exception | None = None
    try:
        evaluate_fn = getattr(adapter, "evaluate_private_container", None)
        if callable(evaluate_fn):
            scores = evaluate_fn(eval_session, task, container="eval", timeout=config.eval_timeout)
        else:
            scores = adapter.evaluate(eval_session, task, timeout=config.eval_timeout)
    except Exception as e:
        eval_error = e
        typer.echo(f"  [WARN] eval {name}: {e}", err=True)
    finally:
        with _active_eval_lock:
            _active_eval_sessions.discard(session)
        try:
            client.delete_experiment(agent_exp_id)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  [WARN] cleanup {agent_exp_id}: {e}", err=True)
        client.close()

    if eval_error is not None:
        result: dict[str, Any] = {"task": name, "status": "eval_failed", "tools": tools_count, "error": str(eval_error)}
        if agent_timed_out:
            result["agent_timed_out"] = True
        return result

    if scores is None:
        scores = {"reward": 0.0, "eval_output": "", "error": "eval returned no scores"}
    if scores.get("reward") is None:
        scores["reward"] = 0.0
    scores.setdefault("eval_mode", "in_place")
    if agent_timed_out:
        scores["agent_timed_out"] = True

    score_file.write_text(json.dumps(
        {"task": name, "session_id": session_id, "tools": tools_count, **scores},
        ensure_ascii=False,
    ))
    return {"task": name, "status": "done", "tools": tools_count, "session_id": session_id, **scores}


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def print_summary_table(results: dict[str, dict], adapter: Any) -> None:
    typer.echo(f"\n{'=' * 75}")
    typer.echo(adapter.summary_header())
    for name in sorted(results):
        typer.echo(adapter.summary_row(name, results[name]))
    footer = adapter.summary_footer(results)
    if footer:
        typer.echo(footer)


def print_pass_at_k(
    all_attempts: list[dict[str, dict]], tasks: list[TaskSpec],
    out: Path, adapter: Any,
) -> None:
    k = len(all_attempts)
    task_names = [t.name for t in tasks]
    per_task: dict[str, list[dict]] = {name: [] for name in task_names}
    for attempt_results in all_attempts:
        for name in task_names:
            per_task[name].append(attempt_results.get(name, {}))

    typer.echo(f"\n{'=' * 75}")
    typer.echo(f"  pass@{k} Summary ({k} attempts)")
    typer.echo(adapter.pass_at_k_header())

    all_stats: list[dict] = []
    summary_rows = []
    for name in task_names:
        runs = per_task[name]
        line, stats = adapter.pass_at_k_row(name, runs)
        typer.echo(line)
        all_stats.append(stats)
        summary_rows.append({"task": name, **stats, "attempts": [
            {k_: v for k_, v in r.items() if k_ != "eval_output"} for r in runs
        ]})

    typer.echo(adapter.pass_at_k_footer(all_stats, len(task_names)))

    summary_file = out / "summary.json"
    pass_values = [
        float(s["pass_at_k"]) for s in all_stats
        if isinstance(s.get("pass_at_k"), (int, float))
    ]
    if len(pass_values) == len(all_stats):
        pass_at_k = sum(pass_values) / len(task_names) if task_names else 0
    else:
        pass_count = sum(1 for s in all_stats if s.get("any_pass"))
        pass_at_k = pass_count / len(task_names) if task_names else 0
    summary_file.write_text(json.dumps({
        "k": k, "pass_at_k": pass_at_k, "tasks": summary_rows,
    }, ensure_ascii=False, indent=2))
    typer.echo(f"\n  Summary written to: {summary_file}")


def doctor_sessions(all_attempt_results: list[dict[str, dict[str, Any]]]) -> None:
    from agentm.core.observability import clickhouse as ch

    url = ch.get_url()
    if url is None:
        return
    sids = sorted({
        str(r["session_id"])
        for results in all_attempt_results
        for r in results.values()
        if r.get("session_id")
    })
    if not sids:
        return
    bad = 0
    for sid in sids:
        try:
            violations = [v for v in ch.doctor(url, sid) if v["severity"] == "error"]
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  [WARN] doctor {sid}: {e}", err=True)
            continue
        if violations:
            bad += 1
            for v in violations:
                typer.echo(
                    f"  [DOCTOR] {sid} {v['check']}: "
                    f"expected {v['expected']}, got {v['actual']}"
                )
    typer.echo(f"Trace doctor: {len(sids) - bad}/{len(sids)} sessions clean")


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def kill_children() -> None:
    with _active_agent_lock:
        entries = list(_active_agent_interrupts)
    for loop, interrupt in entries:
        try:
            loop.call_soon_threadsafe(interrupt.set)
        except RuntimeError as e:
            logger.debug("interrupt signal to closed loop: {}", e)


def cleanup_experiments(
    pfx: str, mdl: str, rid: str, task_list: list[TaskSpec], n_attempts: int,
) -> None:
    try:
        import arl

        client = arl.GatewayClient()
        cleaned = 0
        for t in task_list:
            for attempt_idx in range(n_attempts):
                exp_ids = _experiment_ids(pfx, mdl, rid, t.name, attempt_idx if n_attempts > 1 else None)
                for exp_id in exp_ids:
                    try:
                        client.delete_experiment(exp_id)
                        cleaned += 1
                    except Exception as e:  # noqa: BLE001
                        typer.echo(f"  [WARN] cleanup {exp_id}: {e}", err=True)
        client.close()
        if cleaned:
            typer.echo(f"  Cleaned up {cleaned} ARL experiment(s)")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"  [WARN] experiment cleanup failed: {e}", err=True)


def cleanup_eval_sessions() -> None:
    with _active_eval_lock:
        sessions = list(_active_eval_sessions)
    if sessions:
        typer.echo(f"  Cleaning up {len(sessions)} eval sandbox(es)...")
    for s in sessions:
        try:
            s.delete_sandbox()
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  [WARN] eval sandbox cleanup: {e}", err=True)
