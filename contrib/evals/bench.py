#!/usr/bin/env python3
"""Multi-format benchmark runner: Terminal Bench / Harbor / SWE-bench.

Usage:
    uv run python contrib/evals/bench.py build --repo ~/longcli-bench/tasks_long_cli --push
    uv run python contrib/evals/bench.py list --repo ~/longcli-bench/tasks_long_cli
    uv run python contrib/evals/bench.py batch --repo ~/longcli-bench/tasks_long_cli --model litellm -j 20
    uv run python contrib/evals/bench.py list --bench swebench-verified --source princeton-nlp/SWE-bench_Verified
    uv run python contrib/evals/bench.py batch --bench harbor --repo ~/harbor-bench/tasks -j 10
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
from benchmarks.base import TaskSpec, image_name, replay_trajectory

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


def _run_and_eval_one(
    task: TaskSpec, *,
    adapter: object,
    source: str,
    model: str, gateway: str,
    registry: str, prefix: str, tag: str,
    out: Path, eval_timeout: int,
    api_key: str = "",
) -> dict:
    """Run agent + evaluate one task. Returns result dict."""
    import arl

    name = task.name
    log = out / f"{name}.log"
    score_file = out / f"{name}.score.json"

    image = adapter.get_image(task, registry, prefix, tag)  # type: ignore[union-attr]

    # --- Agent phase ---
    session_id = None
    if log.is_file():
        raw = log.read_bytes()
        m = re.search(rb"session_id=(\S+)", raw)
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
            "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{model}-{name}",
        }
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
                proc.wait()
            finally:
                with _active_procs_lock:
                    _active_procs.discard(proc)

        raw = log.read_bytes()
        m = re.search(rb"session_id=(\S+)", raw)
        if m:
            session_id = m.group(1).decode()

    if session_id is None:
        return {"task": name, "status": "agent_failed"}

    tools_m = re.search(rb"tool_calls=(\d+)", log.read_bytes())
    tools_count = tools_m.group(1).decode() if tools_m else "?"

    # --- Eval phase ---
    if score_file.is_file():
        scores = json.loads(score_file.read_text())
        return {"task": name, "status": "done", "tools": tools_count, **scores}

    from arl.session import ResourceRequirements  # type: ignore[import-not-found]
    session = arl.ManagedSession(
        image=image,
        experiment_id=f"eval-{model}-{name}",
        gateway_url=gateway,
        workspace_dir="/app",
        api_key=api_key or None,
        timeout=max(600.0, eval_timeout * 2.0),
        resources=ResourceRequirements(
            requests={"cpu": "4", "memory": "8Gi"},
            limits={"cpu": "8", "memory": "16Gi"},
        ),
    )
    session.create_sandbox()

    try:
        replay_trajectory(session, session_id)
        scores = adapter.evaluate(session, task, timeout=eval_timeout)  # type: ignore[union-attr]
    finally:
        session.delete_sandbox()

    score_file.write_text(json.dumps({"task": name, **scores}, ensure_ascii=False))

    return {"task": name, "status": "done", "tools": tools_count, **scores}


def _print_summary_table(results: dict[str, dict], adapter: object) -> None:
    """Print final summary table using adapter methods."""
    typer.echo(f"\n{'=' * 75}")
    typer.echo(adapter.summary_header())  # type: ignore[union-attr]
    for name in sorted(results):
        typer.echo(adapter.summary_row(name, results[name]))  # type: ignore[union-attr]
    footer = adapter.summary_footer(results)  # type: ignore[union-attr]
    if footer:
        typer.echo(footer)


def _print_pass_at_k(
    all_attempts: list[dict[str, dict]],
    tasks: list[TaskSpec],
    out: Path,
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
    typer.echo(f"  {'-' * 70}")
    typer.echo(
        f"  {'Task':<25} {'Best F2P':<10} {'Any pass':<10} "
        f"{'Avg F2P step':<14} {'Avg P2P step'}"
    )
    typer.echo(f"  {'-' * 70}")

    any_pass_count = 0
    f2p_steps_all: list[float] = []
    p2p_steps_all: list[float] = []

    summary_rows = []
    for name in task_names:
        runs = per_task[name]
        f2p_steps: list[float] = []
        p2p_steps: list[float] = []
        any_all_pass = False

        for r in runs:
            f2p = r.get("f2p")
            p2p = r.get("p2p")
            f2p_s = f2p.get("step_score") if isinstance(f2p, dict) else None
            if isinstance(f2p_s, (int, float)):
                f2p_steps.append(f2p_s)
            p2p_total = p2p.get("total", 0) if isinstance(p2p, dict) else 0
            p2p_passed = p2p.get("passed", 0) if isinstance(p2p, dict) else 0
            if p2p_total > 0:
                p2p_steps.append(p2p_passed / p2p_total)

            f2p_pass = isinstance(f2p_s, (int, float)) and f2p_s >= 1.0
            p2p_pass = p2p_total > 0 and p2p_passed == p2p_total
            if f2p_pass and p2p_pass:
                any_all_pass = True

        best_f2p = max(f2p_steps) if f2p_steps else None
        avg_f2p = sum(f2p_steps) / len(f2p_steps) if f2p_steps else None
        avg_p2p = sum(p2p_steps) / len(p2p_steps) if p2p_steps else None

        if any_all_pass:
            any_pass_count += 1
        if avg_f2p is not None:
            f2p_steps_all.append(avg_f2p)
        if avg_p2p is not None:
            p2p_steps_all.append(avg_p2p)

        best_str = f"{best_f2p:.1%}" if best_f2p is not None else "-"
        pass_str = "YES" if any_all_pass else "no"
        avg_f2p_str = f"{avg_f2p:.1%}" if avg_f2p is not None else "-"
        avg_p2p_str = f"{avg_p2p:.1%}" if avg_p2p is not None else "-"

        typer.echo(f"  {name:<25} {best_str:<10} {pass_str:<10} {avg_f2p_str:<14} {avg_p2p_str}")
        summary_rows.append({
            "task": name,
            "best_f2p_step": best_f2p,
            "any_all_pass": any_all_pass,
            "avg_f2p_step": avg_f2p,
            "avg_p2p_step": avg_p2p,
            "attempts": [{
                "f2p": r.get("f2p"),
                "p2p": r.get("p2p"),
                "tools": r.get("tools"),
                "status": r.get("status"),
            } for r in runs],
        })

    n_tasks = len(task_names)
    typer.echo(f"\n  Overall pass@{k}:     {any_pass_count}/{n_tasks} = {any_pass_count / n_tasks:.1%}")
    if f2p_steps_all:
        typer.echo(f"  Avg F2P Step Score:  {sum(f2p_steps_all) / len(f2p_steps_all):.1%}")
    if p2p_steps_all:
        typer.echo(f"  Avg P2P Step Score:  {sum(p2p_steps_all) / len(p2p_steps_all):.1%}")

    summary_file = out / "summary.json"
    summary_file.write_text(json.dumps({
        "k": k,
        "pass_at_k": any_pass_count / n_tasks if n_tasks else 0,
        "avg_f2p_step": sum(f2p_steps_all) / len(f2p_steps_all) if f2p_steps_all else None,
        "avg_p2p_step": sum(p2p_steps_all) / len(p2p_steps_all) if p2p_steps_all else None,
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)

    all_attempt_results: list[dict[str, dict]] = []

    try:
        for attempt_idx in range(attempts):
            if interrupted:
                break

            out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
            out.mkdir(parents=True, exist_ok=True)

            typer.echo(f"{'=' * 60}")
            if attempts > 1:
                typer.echo(f"Attempt {attempt_idx + 1}/{attempts}")
            typer.echo(
                f"Batch: {len(tasks)} tasks | bench={bench} | "
                f"model={model} | concurrency={concurrency}"
            )
            typer.echo(f"Results: {out}")
            typer.echo("")

            results: dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending: dict[object, str] = {}
                for t in tasks:
                    if interrupted:
                        break
                    f = pool.submit(
                        _run_and_eval_one, t,
                        adapter=adapter, source=resolved_source,
                        model=model, gateway=gateway,
                        registry=registry, prefix=prefix, tag=tag,
                        out=out, eval_timeout=eval_timeout,
                        api_key=api_key,
                    )
                    pending[f] = t.name
                for future in as_completed(pending):
                    name = pending[future]
                    try:
                        r = future.result()
                        results[name] = r
                        typer.echo(adapter.format_score_line(r))
                    except Exception as e:
                        typer.echo(f"  [ERROR] {name}: {e}")

            all_attempt_results.append(results)
            _print_summary_table(results, adapter)

            done = sum(1 for r in results.values() if r.get("status") == "done")
            typer.echo(f"\n{done}/{len(tasks)} completed")

            if bench.startswith("swebench"):
                write_predictions(results, out, model)
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if attempts > 1 and all_attempt_results:
        _print_pass_at_k(all_attempt_results, tasks, base_out)


if __name__ == "__main__":
    app()
