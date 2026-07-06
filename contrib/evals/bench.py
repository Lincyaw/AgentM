#!/usr/bin/env python3
"""Multi-format benchmark runner: Terminal Bench / Harbor / SWE-bench.

Usage:
    uv run python contrib/evals/bench.py build --repo ~/longcli-bench/tasks_long_cli --push
    uv run python contrib/evals/bench.py list --repo ~/longcli-bench/tasks_long_cli
    uv run python contrib/evals/bench.py batch --repo ~/longcli-bench/tasks_long_cli --model litellm -j 20
    uv run python contrib/evals/bench.py list --bench swebench-verified --source princeton-nlp/SWE-bench_Verified
    uv run python contrib/evals/bench.py mirror --bench harbor --repo ~/harbor-datasets/terminal-bench --registry opspai --prefix tb2 -j 8
    uv run python contrib/evals/bench.py batch --bench harbor --repo ~/harbor-datasets/terminal-bench --registry opspai --prefix tb2 -j 10
"""

from __future__ import annotations

import inspect
import json
import os
import re
import signal
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Callable, cast

import typer
from loguru import logger

from benchmarks import get_adapter
from benchmarks.base import TaskSpec, eval_image_name, image_name

try:
    import yaml as _yaml
except ImportError:  # noqa: S110
    _yaml = None  # type: ignore[assignment]


app = typer.Typer(
    name="bench",
    help="Build, push, run, and evaluate benchmark task images.",
    add_completion=False,
)

# Env-first so a general checkout is not tied to one private registry; the
# fallback keeps the historical default for existing runs.
DEFAULT_IMAGE_REGISTRY = os.environ.get(
    "AGENTM_EVAL_IMAGE_REGISTRY", "pair-diag-cn-guangzhou.cr.volces.com/pair"
)
DEFAULT_REMOTE_TERMINAL_BENCH_SCENARIO = "terminal_bench:arl"

_INFRA_RETRY_LOG_MARKERS = (
    b"404 Client Error",
    b"session not found",
    b"Source session",
    b"ImagePullBackOff",
    b"Failed to pull image",
    b"failed to pull image",
)


def _filter_supported_kwargs(
    callable_obj: object,
    kwargs: dict[str, Any],
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _eval_mode(private_eval: bool) -> str:
    return "private_container" if private_eval else "uploaded_tests"


def _experiment_ids(
    prefix: str,
    model: str,
    run_id: str,
    task: str,
    attempt_idx: int | None,
) -> tuple[str, str]:
    attempt = f"a{attempt_idx}" if attempt_idx is not None else "a0"
    suffix = f"{run_id}-{model}-{attempt}-{task}"
    return (
        _safe_experiment_id(f"{prefix}-{suffix}"),
        _safe_experiment_id(f"eval-{suffix}"),
    )


# ---------------------------------------------------------------------------
# TB1 legacy helpers (used by build command)
# ---------------------------------------------------------------------------

def _build_base_images(base_dir: Path) -> None:
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
                ["docker", "image", "inspect", tag], capture_output=True
            ).returncode == 0:
                typer.echo(f"  Base {tag} exists, skip")
                continue
            typer.echo(f"  Building {tag} ...")
            subprocess.run(
                ["docker", "build", "-f", str(dockerfile), "-t", tag,
                 str(base_dir), "-q"],
                check=True, capture_output=True,
            )


def _generate_skaffold(tasks: list[TaskSpec], registry: str, prefix: str, tag: str) -> dict:
    artifacts = []
    for task in tasks:
        img = image_name(task.name, registry, prefix, tag).rsplit(":", 1)[0]
        artifacts.append({"image": img, "context": task.path})
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


def _get_eval_image(adapter: Any, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
    getter = getattr(adapter, "get_eval_image", None)
    if callable(getter):
        return str(getter(task, registry, prefix, tag))
    return eval_image_name(task.name, registry, prefix, tag)


def _build_eval_images(
    tasks: list[TaskSpec],
    *,
    adapter: Any,
    registry: str,
    prefix: str,
    tag: str,
    push: bool,
) -> None:
    build_eval_image = getattr(adapter, "build_eval_image", None)
    if not callable(build_eval_image):
        typer.echo(
            f"Error: {type(adapter).__name__} does not support private eval image builds. "
            "Implement build_eval_image() in the benchmark adapter.",
            err=True,
        )
        raise typer.Exit(1)

    for task in tasks:
        base_image = adapter.get_image(task, registry, prefix, tag)
        eval_image = _get_eval_image(adapter, task, registry, prefix, tag)
        typer.echo(f"  Building eval {eval_image} <- {base_image}")
        try:
            build_eval_image(
                task,
                base_image=base_image,
                eval_image=eval_image,
                push=push,
            )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Error: failed to build eval image for {task.name}: {e}", err=True)
            raise typer.Exit(1) from e


def _private_eval_container(
    adapter: Any,
    task: TaskSpec,
    registry: str,
    prefix: str,
    tag: str,
    *,
    container: str,
) -> dict[str, Any]:
    image_pull_policy = os.environ.get(
        "AGENTM_BENCH_PRIVATE_EVAL_IMAGE_PULL_POLICY",
        "IfNotPresent",
    )
    spec_hook = getattr(adapter, "private_eval_container", None)
    if callable(spec_hook):
        return dict(
            spec_hook(
                task,
                registry,
                prefix,
                tag,
                container=container,
                image_pull_policy=image_pull_policy,
            )
        )
    return {
        "name": container,
        "image": _get_eval_image(adapter, task, registry, prefix, tag),
        "mountWorkspace": True,
        "workspaceMountPath": "/app",
        "workspaceAccess": "readWrite",
        "imagePullPolicy": image_pull_policy,
    }


# ---------------------------------------------------------------------------
# Core run+eval logic
# ---------------------------------------------------------------------------

_active_procs: set[subprocess.Popen] = set()  # type: ignore[type-arg]
_active_procs_lock = __import__("threading").Lock()
_active_eval_sessions: set[Any] = set()
_active_eval_lock = __import__("threading").Lock()


@dataclass(frozen=True, slots=True)
class _RunEvalConfig:
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


@dataclass(frozen=True, slots=True)
class _RunEvalJob:
    task: TaskSpec
    out: Path
    attempt_slot: int
    experiment_attempt_idx: int | None


def _terminate_process_group(
    proc: subprocess.Popen,  # type: ignore[type-arg]
    *,
    grace_seconds: float = 10.0,
    log_file: Any | None = None,
) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()

    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        if log_file is not None:
            log_file.write("[bench] agent did not terminate; killing process group\n")
            log_file.flush()

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()
    proc.wait()


def _run_and_eval_one(
    job: _RunEvalJob,
    config: _RunEvalConfig,
) -> dict[str, Any]:
    """Run agent + evaluate one task. Returns result dict."""
    import arl

    task = job.task
    adapter = config.adapter
    name = task.name
    log = job.out / f"{name}.log"
    score_file = job.out / f"{name}.score.json"
    agent_exp_id = _experiment_ids(
        config.prefix,
        config.model,
        config.run_id,
        name,
        job.experiment_attempt_idx,
    )[0]

    image = adapter.get_image(task, config.registry, config.prefix, config.tag)

    # --- Agent phase ---
    session_id = None
    agent_timed_out = False
    if log.is_file():
        raw = log.read_bytes()
        agent_timed_out = b"[bench] agent timeout" in raw
        if not any(marker in raw for marker in _INFRA_RETRY_LOG_MARKERS):
            # Only match the completion summary line (session_id=xxx at EOF),
            # NOT the startup log line (session id: xxx). The startup line
            # appears immediately; matching it on a re-invocation would skip
            # the agent phase while the subprocess is still running.
            m = re.search(rb"^session_id=(\S+)", raw, re.MULTILINE)
            if m:
                session_id = m.group(1).decode()

    if session_id is None:
        prompt = task.prompt
        if not prompt:
            return {"task": name, "status": "no_instruction"}
        if config.prompt_prefix:
            prompt = f"{config.prompt_prefix}\n\n{prompt}"

        # If a log exists without completion marker, the previous agent may
        # still be running (orphaned by a pkill). Clean up its sandbox before
        # starting a fresh attempt.
        if log.is_file():
            try:
                client = arl.GatewayClient()
                client.delete_experiment(agent_exp_id)
                client.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("orphan cleanup {}: {}", agent_exp_id, e)
            log.unlink(missing_ok=True)

        eval_image = adapter.get_eval_image(task, config.registry, config.prefix, config.tag)
        env = {
            **os.environ,
            "AGENTM_AGENT_ENV_IMAGE": image,
            "AGENTM_AGENT_ENV_EVAL_IMAGE": eval_image,
            "AGENTM_AGENT_ENV_EXPERIMENT_ID": agent_exp_id,
            "AGENTM_AGENT_ENV_DELETE_ON_SHUTDOWN": "false",
        }
        cmd = [
            "uv", "run", "agentm",
            "--scenario", config.scenario,
            "--model", config.model,
            "-p", prompt,
        ]
        with open(log, "w") as f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with _active_procs_lock:
                _active_procs.add(proc)
            try:
                try:
                    proc.wait(
                        timeout=config.agent_timeout
                        if config.agent_timeout > 0
                        else None
                    )
                except subprocess.TimeoutExpired:
                    agent_timed_out = True
                    f.write(
                        "\n[bench] agent timeout after "
                        f"{config.agent_timeout}s; terminating\n"
                    )
                    f.flush()
                    _terminate_process_group(proc, log_file=f)
            finally:
                with _active_procs_lock:
                    _active_procs.discard(proc)

        raw = log.read_bytes()
        m = re.search(rb"^session_id=(\S+)", raw, re.MULTILINE)
        if not m:
            m = re.search(rb"session id:\s*(\S+)", raw)
        if m:
            session_id = m.group(1).decode()

    if session_id is None:
        return {"task": name, "status": "agent_failed"}

    tools_m = re.search(rb"tool_calls=(\d+)", log.read_bytes())
    tools_count = tools_m.group(1).decode() if tools_m else "?"

    # --- Eval phase (in-place: reuse the agent's sandbox) ---
    if score_file.is_file():
        cached_scores = json.loads(score_file.read_text())
        return {"task": name, "status": "done", "tools": tools_count, **cached_scores}

    # Find the agent's ARL sandbox session via experiment_id.
    client = arl.GatewayClient()
    try:
        agent_arl_sessions = client.list_experiment_sessions(agent_exp_id)
    except Exception as e:
        client.close()
        return {"task": name, "status": "eval_create_failed", "tools": tools_count, "error": f"list sessions: {e}"}
    if not agent_arl_sessions:
        client.close()
        return {"task": name, "status": "eval_create_failed", "tools": tools_count, "error": "no ARL session for agent"}
    arl_session_id = agent_arl_sessions[0].id

    session = arl.SandboxSession.attach(arl_session_id)
    with _active_eval_lock:
        _active_eval_sessions.add(session)
    scores: dict[str, Any] | None = None
    eval_error: Exception | None = None
    try:
        evaluate_fn = getattr(adapter, "evaluate_private_container", None)
        if callable(evaluate_fn):
            scores = evaluate_fn(session, task, container="eval", timeout=config.eval_timeout)
        else:
            scores = adapter.evaluate(session, task, timeout=config.eval_timeout)
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
        result: dict[str, Any] = {
            "task": name,
            "status": "eval_failed",
            "tools": tools_count,
            "error": str(eval_error),
        }
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

    score_file.write_text(json.dumps({"task": name, **scores}, ensure_ascii=False))

    return {"task": name, "status": "done", "tools": tools_count, **scores}


def _print_summary_table(results: dict[str, dict], adapter: Any) -> None:
    """Print final summary table using adapter methods."""
    typer.echo(f"\n{'=' * 75}")
    typer.echo(adapter.summary_header())
    for name in sorted(results):
        typer.echo(adapter.summary_row(name, results[name]))
    footer = adapter.summary_footer(results)
    if footer:
        typer.echo(footer)


def _print_pass_at_k(
    all_attempts: list[dict[str, dict]],
    tasks: list[TaskSpec],
    out: Path,
    adapter: Any,
) -> None:
    """Compute and print pass@k metrics across multiple attempts."""
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
            {k: v for k, v in r.items() if k != "eval_output"} for r in runs
        ]})

    typer.echo(adapter.pass_at_k_footer(all_stats, len(task_names)))

    summary_file = out / "summary.json"
    pass_values = [
        float(s["pass_at_k"])
        for s in all_stats
        if isinstance(s.get("pass_at_k"), (int, float))
    ]
    if len(pass_values) == len(all_stats):
        pass_at_k = sum(pass_values) / len(task_names) if task_names else 0
    else:
        pass_count = sum(1 for s in all_stats if s.get("any_pass"))
        pass_at_k = pass_count / len(task_names) if task_names else 0
    summary_file.write_text(json.dumps({
        "k": k,
        "pass_at_k": pass_at_k,
        "tasks": summary_rows,
    }, ensure_ascii=False, indent=2))
    typer.echo(f"\n  Summary written to: {summary_file}")


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _resolve_source(
    repo: Path | None, source: str | None, bench: str
) -> str:
    """Resolve the task source from --repo or --source flags."""
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
    typer.echo("Error: --repo required for this bench format.", err=True)
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def build(
    repo: Annotated[Path, typer.Option("--repo")],
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    registry: Annotated[str, typer.Option()] = DEFAULT_IMAGE_REGISTRY,
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    push: Annotated[bool, typer.Option("--push")] = False,
    base_dir: Annotated[Path | None, typer.Option("--base-dir")] = None,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    eval_images: Annotated[
        bool,
        typer.Option(
            "--eval-images",
            help="Also build paired adapter-defined private evaluator images.",
        ),
    ] = False,
    eval_only: Annotated[
        bool,
        typer.Option(
            "--eval-only",
            help="Build only paired private evaluator images; task images must already exist.",
        ),
    ] = False,
) -> None:
    """Build (and optionally push) task environment images."""
    adapter = get_adapter(bench)
    if not eval_only and not adapter.supports_build():
        typer.echo(f"Error: {bench} uses pre-built images, nothing to build.", err=True)
        raise typer.Exit(1)

    repo = repo.expanduser().resolve()
    if base_dir:
        _build_base_images(base_dir.expanduser().resolve())

    tasks = adapter.discover_tasks(str(repo))
    if task:
        tasks = [t for t in tasks if t.name in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    if not eval_only and _yaml is None:
        typer.echo("Error: PyYAML required for skaffold config generation.", err=True)
        raise typer.Exit(1)

    if not eval_only:
        skaffold_cfg = _generate_skaffold(tasks, registry, prefix, tag)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(skaffold_cfg, f)
            skaffold_path = f.name
        try:
            cmd = ["skaffold", "build", "-f", skaffold_path]
            if push:
                cmd.append("--push")
            result = subprocess.run(cmd, env={**os.environ, "TAG": tag})
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
        finally:
            os.unlink(skaffold_path)
        typer.echo(f"Built {len(tasks)} images ({registry}/{prefix}-*:{tag})")

    if eval_images or eval_only:
        _build_eval_images(
            tasks,
            adapter=adapter,
            registry=registry,
            prefix=prefix,
            tag=tag,
            push=push,
        )
        typer.echo(f"Built {len(tasks)} eval images ({registry}/{prefix}-*-eval:{tag})")


@app.command()
def mirror(
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "harbor",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "tb2",
    tag: Annotated[str, typer.Option()] = "v0",
    concurrency: Annotated[int, typer.Option("-j")] = 4,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Pull upstream images, retag, and push to our registry.

    For benchmarks with pre-built images (Harbor, SWE-bench), mirrors
    source images to {registry}/{prefix}-{task}:{tag}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    resolved_source = _resolve_source(repo, source, bench)
    adapter = get_adapter(bench)
    tasks = adapter.discover_tasks(resolved_source)
    if task:
        tasks = [t for t in tasks if t.name in task]

    mirrorable = [(t, adapter.get_source_image(t)) for t in tasks]
    mirrorable = [(t, src) for t, src in mirrorable if src]
    if not mirrorable:
        typer.echo("No images to mirror (adapter has no source images).", err=True)
        raise typer.Exit(1)

    typer.echo(f"Mirror: {len(mirrorable)} images | {bench} → {registry}/")

    def _mirror_one(t: TaskSpec, src: str) -> tuple[str, bool, str]:
        dst = adapter.get_image(t, registry, prefix, tag)
        if dry_run:
            return t.name, True, f"{src} → {dst}"
        try:
            subprocess.run(
                ["crane", "copy", src, dst],
                check=True, capture_output=True,
            )
            return t.name, True, f"{src} → {dst}"
        except subprocess.CalledProcessError as e:
            return t.name, False, e.stderr.decode(errors="replace").strip()

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_mirror_one, t, src): t.name for t, src in mirrorable}
        for future in as_completed(futures):
            name, success, msg = future.result()
            status = "OK" if success else "FAIL"
            typer.echo(f"  [{status}] {name}: {msg}")
            if success:
                ok += 1
            else:
                fail += 1

    typer.echo(f"\n{ok} mirrored, {fail} failed")
    if fail:
        raise typer.Exit(1)


@app.command("list")
def list_tasks(
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    registry: Annotated[str, typer.Option()] = DEFAULT_IMAGE_REGISTRY,
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List discovered tasks and their image names."""
    resolved_source = _resolve_source(repo, source, bench)
    adapter = get_adapter(bench)
    tasks = adapter.discover_tasks(resolved_source)

    if json_out:
        out = []
        for t in tasks:
            d = {
                "name": t.name,
                "image": adapter.get_image(t, registry, prefix, tag),
                "difficulty": t.difficulty,
                "category": t.category,
            }
            if t.instance_id:
                d["instance_id"] = t.instance_id
            if t.repo:
                d["repo"] = t.repo
            out.append(d)
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if bench.startswith("swebench"):
        typer.echo(f"{'Instance ID':<55} {'Repo':<30} {'Image'}")
        for t in tasks:
            typer.echo(f"{t.name:<55} {t.repo:<30} "
                       f"{adapter.get_image(t, registry, prefix, tag)}")
    else:
        typer.echo(f"{'Task':<30} {'Diff':<8} {'Image'}")
        for t in tasks:
            typer.echo(f"{t.name:<30} {t.difficulty:<8} "
                       f"{adapter.get_image(t, registry, prefix, tag)}")
    typer.echo(f"\n{len(tasks)} tasks")


@app.command()
def run(
    task: Annotated[str, typer.Option("--task", "-t")],
    instruction: Annotated[str | None, typer.Option("-p")] = None,
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    model: Annotated[str, typer.Option()] = "glm47",
    gateway: Annotated[str, typer.Option()] = "http://localhost:28080",
    registry: Annotated[str, typer.Option()] = DEFAULT_IMAGE_REGISTRY,
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
) -> None:
    """Run a single task via ARL sandbox."""
    adapter = get_adapter(bench)
    resolved_source = _resolve_source(repo, source, bench)

    all_tasks = adapter.discover_tasks(resolved_source)
    matching = [t for t in all_tasks if t.name == task]
    if not matching:
        typer.echo(f"Error: task '{task}' not found in {bench} source.", err=True)
        raise typer.Exit(1)
    task_spec = matching[0]

    image = adapter.get_image(task_spec, registry, prefix, tag)
    prompt = instruction or task_spec.prompt
    if not prompt:
        typer.echo("Error: no prompt found. Provide -p or ensure task has instruction.", err=True)
        raise typer.Exit(1)

    env = {
        **os.environ,
        "AGENTM_AGENT_ENV_IMAGE": image,
        "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{task}",
    }
    result = subprocess.run([
        "uv", "run", "agentm", "--scenario", DEFAULT_REMOTE_TERMINAL_BENCH_SCENARIO,
        "--model", model, "-p", prompt,
    ], env=env)
    raise typer.Exit(result.returncode)


@app.command()
def batch(
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    model: Annotated[str, typer.Option()] = "glm47",
    registry: Annotated[str, typer.Option()] = DEFAULT_IMAGE_REGISTRY,
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    concurrency: Annotated[int, typer.Option("-j")] = 5,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("/tmp/bench-results"),
    eval_timeout: Annotated[int, typer.Option("--eval-timeout")] = 300,
    agent_timeout: Annotated[int, typer.Option("--agent-timeout")] = 0,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    attempts: Annotated[int, typer.Option("--attempts", "-n")] = 1,
    scenario: Annotated[str, typer.Option("--scenario")] = DEFAULT_REMOTE_TERMINAL_BENCH_SCENARIO,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    prompt_prefix: Annotated[str, typer.Option("--prompt-prefix")] = "",
) -> None:
    """Run all tasks in parallel, then evaluate. Results include scores.

    Eval runs in-place inside the agent's sandbox — no second pod, no
    trajectory replay. The sandbox is deleted after scoring.

    With --attempts N, runs N independent attempts per task and computes
    pass@k metrics. Results are stored in attempt_0/ .. attempt_{N-1}/.

    Ctrl-C stops submitting new tasks and waits for in-flight ones to finish.
    """
    from concurrent.futures import Future, ThreadPoolExecutor, as_completed

    from benchmarks.swebench import write_predictions

    resolved_source = _resolve_source(repo, source, bench)
    adapter = get_adapter(bench)
    tasks = adapter.discover_tasks(resolved_source)
    if task:
        tasks = [t for t in tasks if t.name in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    base_out = results_dir / f"{bench}-{model}"
    base_out.mkdir(parents=True, exist_ok=True)
    resolved_run_id = _safe_experiment_id(run_id, max_len=24) if run_id else _default_run_id()

    interrupted = False

    def _kill_children() -> None:
        with _active_procs_lock:
            procs = list(_active_procs)
        for p in procs:
            _terminate_process_group(p, grace_seconds=5.0)

    def _cleanup_experiments(
        pfx: str, mdl: str, rid: str,
        task_list: list[TaskSpec], n_attempts: int,
    ) -> None:
        try:
            import arl
            client = arl.GatewayClient()
            cleaned = 0
            for t in task_list:
                for attempt_idx in range(n_attempts):
                    exp_ids = _experiment_ids(
                        pfx,
                        mdl,
                        rid,
                        t.name,
                        attempt_idx if n_attempts > 1 else None,
                    )
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

    def _cleanup_eval_sessions() -> None:
        with _active_eval_lock:
            sessions = list(_active_eval_sessions)
        if sessions:
            typer.echo(f"  Cleaning up {len(sessions)} eval sandbox(es)...")
        for s in sessions:
            try:
                s.delete_sandbox()  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001
                typer.echo(f"  [WARN] eval sandbox cleanup: {e}", err=True)

    def _on_sigint(_sig: int, _frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            typer.echo("\nForce quit.")
            raise SystemExit(1)
        interrupted = True
        typer.echo("\nCtrl-C — terminating child processes & eval sandboxes...")
        _kill_children()
        _cleanup_eval_sessions()

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)

    # Build flat job list: task-first so same-image jobs run together (warm pool)
    all_attempt_results: list[dict[str, dict]] = [{} for _ in range(attempts)]
    jobs: list[_RunEvalJob] = []
    for attempt_idx in range(attempts):
        out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
        out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        for attempt_idx in range(attempts):
            out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
            score_path = out / f"{t.name}.score.json"
            if score_path.is_file():
                scores = json.loads(score_path.read_text())
                log_path = out / f"{t.name}.log"
                tools_count = "?"
                if log_path.is_file():
                    tools_m = re.search(rb"tool_calls=(\d+)", log_path.read_bytes())
                    tools_count = tools_m.group(1).decode() if tools_m else "?"
                all_attempt_results[attempt_idx][t.name] = {
                    "task": t.name,
                    "status": "done",
                    "tools": tools_count,
                    **scores,
                }
                continue
            jobs.append(
                _RunEvalJob(
                    task=t,
                    out=out,
                    attempt_slot=attempt_idx,
                    experiment_attempt_idx=attempt_idx if attempts > 1 else None,
                )
            )

    total_jobs = len(jobs)
    typer.echo(
        f"Batch: {len(tasks)} tasks × {attempts} attempt(s) = {total_jobs} jobs | "
        f"bench={bench} | model={model} | concurrency={concurrency}"
        + (f" | agent_timeout={agent_timeout}s" if agent_timeout > 0 else "")
        + " | in-place eval"
    )
    typer.echo(f"Run id: {resolved_run_id}")
    typer.echo(f"Results: {base_out}")
    typer.echo("")

    run_config = _RunEvalConfig(
        adapter=adapter,
        model=model,
        registry=registry,
        prefix=prefix,
        tag=tag,
        eval_timeout=eval_timeout,
        agent_timeout=agent_timeout,
        scenario=scenario,
        run_id=resolved_run_id,
        prompt_prefix=prompt_prefix,
    )

    # Collect results per attempt
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending: dict[Future[dict[str, Any]], _RunEvalJob] = {}
            for job in jobs:
                if interrupted:
                    break
                f = pool.submit(_run_and_eval_one, job, run_config)
                pending[f] = job
            for future in as_completed(pending):
                job = pending[future]
                attempt_idx = job.attempt_slot
                name = job.task.name
                completed += 1
                attempt_label = f"[a{attempt_idx}] " if attempts > 1 else ""
                try:
                    r = future.result()
                    all_attempt_results[attempt_idx][name] = r
                    line = adapter.format_score_line(r)
                    typer.echo(f"  {attempt_label}{line.strip()} ({completed}/{total_jobs})")
                except Exception as e:
                    logger.debug("Task {} failed: {}", name, e)
                    typer.echo(f"  {attempt_label}[ERROR] {name}: {e} ({completed}/{total_jobs})")
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        _cleanup_eval_sessions()
        _cleanup_experiments(prefix, model, resolved_run_id, tasks, attempts)

    # Per-attempt summaries
    for attempt_idx in range(attempts):
        results = all_attempt_results[attempt_idx]
        if not results:
            continue
        if attempts > 1:
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"Attempt {attempt_idx}")
        _print_summary_table(results, adapter)
        done = sum(1 for r in results.values() if r.get("status") == "done")
        typer.echo(f"\n{done}/{len(tasks)} completed")
        if bench.startswith("swebench"):
            out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
            write_predictions(results, out, model)

    if attempts > 1:
        _print_pass_at_k(all_attempt_results, tasks, base_out, adapter)


if __name__ == "__main__":
    app()
