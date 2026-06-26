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

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from benchmarks import get_adapter
from benchmarks.base import BenchAdapter, TaskSpec, image_name, replay_trajectory

try:
    import yaml as _yaml
except ImportError:  # noqa: S110
    _yaml = None  # type: ignore[assignment]


app = typer.Typer(
    name="bench",
    help="Build, push, run, and evaluate benchmark task images.",
    add_completion=False,
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
            "tagPolicy": {"envTemplate": {"template": "{{.IMAGE_NAME}}:" + tag}},
            "local": {"concurrency": 4, "useBuildkit": True, "push": False},
            "artifacts": artifacts,
        },
    }


# ---------------------------------------------------------------------------
# Core run+eval logic
# ---------------------------------------------------------------------------

_active_procs: set[subprocess.Popen] = set()  # type: ignore[type-arg]
_active_procs_lock = __import__("threading").Lock()

_AGENT_ENV_RESOURCE_DEFAULTS = {
    "AGENTM_AGENT_ENV_CPU_REQUEST": "4",
    "AGENTM_AGENT_ENV_MEMORY_REQUEST": "8Gi",
    "AGENTM_AGENT_ENV_CPU_LIMIT": "8",
    "AGENTM_AGENT_ENV_MEMORY_LIMIT": "16Gi",
}


def _delete_experiment(gateway: str, api_key: str, experiment_id: str) -> None:
    import arl

    client = arl.GatewayClient(base_url=gateway, api_key=api_key or None)
    try:
        client.delete_experiment(experiment_id)
    finally:
        client.close()


def _find_experiment_session(gateway: str, api_key: str, experiment_id: str) -> str | None:
    import arl

    client = arl.GatewayClient(base_url=gateway, api_key=api_key or None)
    try:
        sessions = client.list_experiment_sessions(experiment_id)
    finally:
        client.close()
    if not sessions:
        return None
    sessions = sorted(sessions, key=lambda s: str(getattr(s, "created_at", "") or ""))
    return sessions[-1].id


def _run_and_eval_one(
    task: TaskSpec, *,
    adapter: BenchAdapter,
    source: str,
    model: str, gateway: str,
    registry: str, prefix: str, tag: str,
    out: Path, eval_timeout: int,
    agent_timeout: int | None = None,
    attempt_idx: int | None = None,
    api_key: str = "",
    eval_mode: str = "same-session",
) -> dict:
    """Run agent + evaluate one task. Returns result dict."""
    import arl

    name = task.name
    log = out / f"{name}.log"
    score_file = out / f"{name}.score.json"

    image = adapter.get_image(task, registry, prefix, tag)
    run_id = f"{model}-a{attempt_idx}-{name}" if attempt_idx is not None else f"{model}-{name}"
    agent_experiment_id = f"{prefix}-{run_id}"
    eval_experiment_id = f"eval-{run_id}"

    raw_log = log.read_bytes() if log.is_file() else b""
    tools_m = re.search(rb"tool_calls=(\d+)", raw_log)
    tools_count = tools_m.group(1).decode() if tools_m else "?"

    if score_file.is_file():
        scores = json.loads(score_file.read_text())
        return {"task": name, "status": scores.get("status", "done"), "tools": tools_count, **scores}

    # --- Agent phase ---
    session_id = None
    if raw_log:
        m = re.search(rb"session_id=(\S+)", raw_log)
        if m:
            session_id = m.group(1).decode()

    if session_id is None:
        prompt = task.prompt
        if not prompt:
            return {"task": name, "status": "no_instruction"}

        env = {
            **os.environ,
            "AGENTM_AGENT_ENV_IMAGE": image,
            "AGENTM_AGENT_ENV_GATEWAY_URL": gateway,
            "AGENTM_AGENT_ENV_EXPERIMENT_ID": agent_experiment_id,
            **({"AGENTM_AGENT_ENV_API_KEY": api_key} if api_key else {}),
        }
        if eval_mode == "same-session":
            env["AGENTM_AGENT_ENV_DELETE_ON_SHUTDOWN"] = "0"
            for key, value in _AGENT_ENV_RESOURCE_DEFAULTS.items():
                env.setdefault(key, value)
        cmd = [
            "uv", "run", "agentm",
            "--scenario", "terminal_bench_arl",
            "--model", model,
            "-p", prompt,
        ]
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            with _active_procs_lock:
                _active_procs.add(proc)
            try:
                try:
                    proc.wait(timeout=agent_timeout if agent_timeout and agent_timeout > 0 else None)
                except subprocess.TimeoutExpired:
                    f.write(f"\n[bench] agent timed out after {agent_timeout}s; terminating\n")
                    f.flush()
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        f.write("[bench] agent did not exit after TERM; killing\n")
                        f.flush()
                        proc.kill()
                        proc.wait()
                    result = {
                        "task": name,
                        "status": "agent_timeout",
                        "error": f"agent exceeded timeout {agent_timeout}s",
                    }
                    if eval_mode == "same-session":
                        _delete_experiment(gateway, api_key, agent_experiment_id)
                    score_file.write_text(json.dumps(result, ensure_ascii=False))
                    return result
            finally:
                with _active_procs_lock:
                    _active_procs.discard(proc)

        raw = log.read_bytes()
        m = re.search(rb"session_id=(\S+)", raw)
        if m:
            session_id = m.group(1).decode()

    if session_id is None:
        if eval_mode == "same-session":
            _delete_experiment(gateway, api_key, agent_experiment_id)
        return {"task": name, "status": "agent_failed"}

    tools_m = re.search(rb"tool_calls=(\d+)", log.read_bytes())
    tools_count = tools_m.group(1).decode() if tools_m else "?"

    if eval_mode == "same-session":
        arl_session_id = _find_experiment_session(gateway, api_key, agent_experiment_id)
        if arl_session_id is None:
            result = {
                "task": name,
                "status": "agent_sandbox_missing",
                "error": f"no ARL session found for experiment {agent_experiment_id}",
            }
            score_file.write_text(json.dumps(result, ensure_ascii=False))
            return result
        session = arl.SandboxSession.attach(
            arl_session_id,
            gateway_url=gateway,
            timeout=max(600.0, eval_timeout * 2.0),
            keep_alive=True,
            api_key=api_key or None,
        )
        try:
            try:
                scores = adapter.evaluate(session, task, timeout=eval_timeout)
            except Exception as exc:  # noqa: BLE001
                scores = {"status": "eval_failed", "error": str(exc)}
            score_file.write_text(json.dumps({"task": name, **scores}, ensure_ascii=False))
            status = scores.get("status", "done")
            return {"task": name, "status": status, "tools": tools_count, **scores}
        finally:
            try:
                session.delete_sandbox()
            finally:
                session.close()

    from arl.session import ResourceRequirements  # type: ignore[import-not-found]
    session = arl.ManagedSession(
        image=image,
        experiment_id=eval_experiment_id,
        gateway_url=gateway,
        workspace_dir="/app",
        api_key=api_key or None,
        timeout=max(600.0, eval_timeout * 2.0),
        resources=ResourceRequirements(
            requests={"cpu": "4", "memory": "8Gi"},
            limits={"cpu": "8", "memory": "16Gi"},
        ),
    )
    try:
        session.create_sandbox()
    except Exception as e:
        return {"task": name, "status": "eval_create_failed", "tools": tools_count, "error": str(e)}

    try:
        replay_trajectory(session, session_id)
        scores = adapter.evaluate(session, task, timeout=eval_timeout)
    except Exception as e:
        return {"task": name, "status": "eval_failed", "tools": tools_count, "error": str(e)}
    finally:
        try:
            session.delete_sandbox()
        except Exception:  # noqa: S110
            pass
        try:
            session.close()
        except Exception:  # noqa: S110
            pass

    score_file.write_text(json.dumps({"task": name, **scores}, ensure_ascii=False))

    return {"task": name, "status": "done", "tools": tools_count, **scores}


def _print_summary_table(results: dict[str, dict], adapter: BenchAdapter) -> None:
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
    adapter: BenchAdapter,
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
    pass_count = sum(1 for s in all_stats if s.get("any_pass"))
    summary_file.write_text(json.dumps({
        "k": k,
        "pass_at_k": pass_count / len(task_names) if task_names else 0,
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
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    push: Annotated[bool, typer.Option("--push")] = False,
    base_dir: Annotated[Path | None, typer.Option("--base-dir")] = None,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
) -> None:
    """Build (and optionally push) task environment images."""
    adapter = get_adapter(bench)
    if not adapter.supports_build():
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

    if _yaml is None:
        typer.echo("Error: PyYAML required for skaffold config generation.", err=True)
        raise typer.Exit(1)

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

    typer.echo(f"Mirror: {len(mirrorable)} images | {bench} → {registry}/{prefix}-*:{tag}")

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
    registry: Annotated[str, typer.Option()] = "opspai",
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
    registry: Annotated[str, typer.Option()] = "opspai",
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
        "AGENTM_AGENT_ENV_GATEWAY_URL": gateway,
        "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{task}",
    }
    result = subprocess.run([
        "uv", "run", "agentm", "--scenario", "terminal_bench_arl",
        "--model", model, "-p", prompt,
    ], env=env)
    raise typer.Exit(result.returncode)


@app.command()
def batch(
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    model: Annotated[str, typer.Option()] = "glm47",
    gateway: Annotated[str, typer.Option()] = "http://localhost:28080",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    concurrency: Annotated[int, typer.Option("-j")] = 5,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("/tmp/bench-results"),
    eval_timeout: Annotated[int, typer.Option("--eval-timeout")] = 300,
    agent_timeout: Annotated[int, typer.Option("--agent-timeout")] = 0,
    eval_mode: Annotated[str, typer.Option("--eval-mode")] = "same-session",
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    api_key: Annotated[str, typer.Option("--api-key", envvar="AGENTM_AGENT_ENV_API_KEY")] = "",
    attempts: Annotated[int, typer.Option("--attempts", "-n")] = 1,
) -> None:
    """Run all tasks in parallel, then evaluate. Results include scores.

    With --attempts N, runs N independent attempts per task and computes
    pass@k metrics. Results are stored in attempt_0/ .. attempt_{N-1}/.

    Ctrl-C stops submitting new tasks and waits for in-flight ones to finish.
    """
    import signal
    from concurrent.futures import Future, ThreadPoolExecutor, as_completed

    from benchmarks.swebench import write_predictions

    resolved_source = _resolve_source(repo, source, bench)
    if eval_mode not in {"same-session", "replay"}:
        typer.echo("Error: --eval-mode must be 'same-session' or 'replay'.", err=True)
        raise typer.Exit(1)

    adapter = get_adapter(bench)
    tasks = adapter.discover_tasks(resolved_source)
    if task:
        tasks = [t for t in tasks if t.name in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    base_out = results_dir / f"{bench}-{model}"
    base_out.mkdir(parents=True, exist_ok=True)

    interrupted = False

    def _kill_children() -> None:
        with _active_procs_lock:
            procs = list(_active_procs)
        for p in procs:
            try:
                p.terminate()
            except OSError:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    def _cleanup_arl_sessions(attempt_tasks: list[TaskSpec], attempt_idx: int) -> None:
        try:
            import arl
            client = arl.GatewayClient(base_url=gateway, api_key=api_key or None)
            for t in attempt_tasks:
                for exp_id in [
                    f"{prefix}-{model}-a{attempt_idx}-{t.name}",
                    f"eval-{model}-a{attempt_idx}-{t.name}",
                ]:
                    try:
                        client.delete_experiment(exp_id)
                    except Exception:  # noqa: S110, BLE001
                        pass
            client.close()
        except Exception:  # noqa: S110, BLE001
            pass

    def _on_sigint(_sig: int, _frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            typer.echo("\nForce quit.")
            raise SystemExit(1)
        interrupted = True
        typer.echo("\nCtrl-C — terminating child processes...")
        _kill_children()
        typer.echo("Cleaning ARL sessions for submitted attempts...")
        for attempt_idx in range(attempts):
            _cleanup_arl_sessions(tasks, attempt_idx)

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)

    # Build flat job list: (attempt_idx, task) for all attempts
    jobs: list[tuple[int, TaskSpec]] = []
    for attempt_idx in range(attempts):
        out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
        out.mkdir(parents=True, exist_ok=True)
        for t in tasks:
            jobs.append((attempt_idx, t))

    total_jobs = len(jobs)
    typer.echo(
        f"Batch: {len(tasks)} tasks × {attempts} attempt(s) = {total_jobs} jobs | "
        f"bench={bench} | model={model} | concurrency={concurrency} | eval_mode={eval_mode}"
    )
    typer.echo(f"Results: {base_out}")
    typer.echo("")

    # Collect results per attempt
    all_attempt_results: list[dict[str, dict]] = [{} for _ in range(attempts)]
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending: dict[Future[dict], tuple[int, str]] = {}
            for attempt_idx, t in jobs:
                if interrupted:
                    break
                out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
                f = pool.submit(
                    _run_and_eval_one, t,
                    adapter=adapter, source=resolved_source,
                    model=model, gateway=gateway,
                    registry=registry, prefix=prefix, tag=tag,
                    out=out, eval_timeout=eval_timeout,
                    agent_timeout=agent_timeout,
                    attempt_idx=attempt_idx if attempts > 1 else None,
                    api_key=api_key,
                    eval_mode=eval_mode,
                )
                pending[f] = (attempt_idx, t.name)
            for future in as_completed(pending):
                attempt_idx, name = pending[future]
                completed += 1
                attempt_label = f"[a{attempt_idx}] " if attempts > 1 else ""
                try:
                    r = future.result()
                    all_attempt_results[attempt_idx][name] = r
                    line = adapter.format_score_line(r)
                    typer.echo(f"  {attempt_label}{line.strip()} ({completed}/{total_jobs})")
                except Exception as e:
                    typer.echo(f"  {attempt_label}[ERROR] {name}: {e} ({completed}/{total_jobs})")
    finally:
        signal.signal(signal.SIGINT, prev_handler)

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
