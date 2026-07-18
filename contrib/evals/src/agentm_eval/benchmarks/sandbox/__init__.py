"""Sandbox benchmark adapter — Docker-sandbox-based benchmarks (TB1, Harbor, SWE-bench).

Wraps the runner module with experiment lifecycle management.
"""

from __future__ import annotations

import json
import os
import signal
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from loguru import logger

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult

from .formats import get_format_adapter
from .runner import (
    DEFAULT_IMAGE_REGISTRY,
    DEFAULT_REMOTE_TERMINAL_BENCH_SCENARIO,
    RunEvalConfig,
    RunEvalJob,
    build_base_images,
    build_eval_images,
    cleanup_eval_sessions,
    cleanup_experiments,
    doctor_sessions,
    generate_skaffold,
    kill_children,
    print_pass_at_k,
    print_summary_table,
    resolve_source,
    run_and_eval_one,
    task_image,
)


def _resolve_defaults(
    adapter: Any,
    registry: str | None,
    prefix: str | None,
    tag: str | None,
) -> tuple[str, str, str]:
    """Fill unset CLI flags from adapter class defaults, then global defaults."""
    if registry is None:
        registry = getattr(adapter, "DEFAULT_REGISTRY", DEFAULT_IMAGE_REGISTRY)
    if prefix is None:
        prefix = getattr(adapter, "DEFAULT_PREFIX", "")
    if tag is None:
        tag = getattr(adapter, "DEFAULT_TAG", "v0")
    return registry, prefix, tag


class SandboxAdapter:
    name = "sandbox"
    description = "Docker-sandbox benchmarks (TB1, Harbor, SWE-bench)"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="sandbox",
            help="Build, run, and evaluate Docker-sandbox benchmark tasks.",
            add_completion=False,
        )

        @cli.command()
        def build(
            repo: Annotated[Path, typer.Option("--repo")],
            bench: Annotated[str, typer.Option("--bench")] = "tb1",
            registry: Annotated[Optional[str], typer.Option()] = None,
            prefix: Annotated[Optional[str], typer.Option()] = None,
            tag: Annotated[Optional[str], typer.Option()] = None,
            push: Annotated[bool, typer.Option("--push")] = False,
            base_dir: Annotated[Path | None, typer.Option("--base-dir")] = None,
            task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
            eval_images: Annotated[bool, typer.Option("--eval-images")] = False,
            eval_only: Annotated[bool, typer.Option("--eval-only")] = False,
        ) -> None:
            """Build (and optionally push) task environment images."""
            import tempfile

            try:
                import yaml as _yaml
            except ImportError:
                _yaml = None  # type: ignore[assignment]

            adapter = get_format_adapter(bench)
            registry, prefix, tag = _resolve_defaults(adapter, registry, prefix, tag)
            if not eval_only and not adapter.supports_build():
                typer.echo(f"Error: {bench} uses pre-built images, nothing to build.", err=True)
                raise typer.Exit(1)

            repo = repo.expanduser().resolve()
            if base_dir:
                build_base_images(base_dir.expanduser().resolve())

            tasks = adapter.discover_tasks(str(repo))
            if task:
                tasks = [t for t in tasks if t.name in task]
            if not tasks:
                typer.echo("No tasks found.", err=True)
                raise typer.Exit(1)

            if not eval_only:
                if _yaml is None:
                    typer.echo("Error: PyYAML required for skaffold config generation.", err=True)
                    raise typer.Exit(1)
                skaffold_cfg = generate_skaffold(tasks, registry, prefix, tag)
                import subprocess

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
                build_eval_images(
                    tasks, adapter=adapter, registry=registry,
                    prefix=prefix, tag=tag, push=push,
                )

        @cli.command()
        def mirror(
            repo: Annotated[Path | None, typer.Option("--repo")] = None,
            source: Annotated[str | None, typer.Option("--source")] = None,
            bench: Annotated[str, typer.Option("--bench")] = "harbor",
            registry: Annotated[Optional[str], typer.Option()] = None,
            prefix: Annotated[Optional[str], typer.Option()] = None,
            tag: Annotated[Optional[str], typer.Option()] = None,
            concurrency: Annotated[int, typer.Option("-j")] = 4,
            task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
            dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
        ) -> None:
            """Mirror upstream images to our registry."""
            import subprocess

            adapter = get_format_adapter(bench)
            resolved_source = resolve_source(repo, source, bench, adapter)
            registry, prefix, tag = _resolve_defaults(adapter, registry, prefix, tag)
            tasks = adapter.discover_tasks(resolved_source)
            if task:
                tasks = [t for t in tasks if t.name in task]

            mirrorable: list[tuple[Any, str]] = [
                (t, src) for t in tasks if (src := adapter.get_source_image(t))
            ]
            if not mirrorable:
                typer.echo("No images to mirror.", err=True)
                raise typer.Exit(1)

            typer.echo(f"Mirror: {len(mirrorable)} images | {bench} → {registry}/")

            def _mirror_one(t: Any, src: str) -> tuple[str, bool, str]:
                dst = adapter.get_image(t, registry, prefix, tag)
                if dry_run:
                    return t.name, True, f"{src} → {dst}"
                try:
                    subprocess.run(["crane", "copy", src, dst], check=True, capture_output=True)
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

        @cli.command("list")
        def list_tasks(
            repo: Annotated[Path | None, typer.Option("--repo")] = None,
            source: Annotated[str | None, typer.Option("--source")] = None,
            bench: Annotated[str, typer.Option("--bench")] = "tb1",
            registry: Annotated[Optional[str], typer.Option()] = None,
            prefix: Annotated[Optional[str], typer.Option()] = None,
            tag: Annotated[Optional[str], typer.Option()] = None,
            json_out: Annotated[bool, typer.Option("--json")] = False,
            source_images: Annotated[bool, typer.Option("--source-images")] = False,
        ) -> None:
            """List discovered tasks and their image names."""
            adapter = get_format_adapter(bench)
            resolved_source = resolve_source(repo, source, bench, adapter)
            registry, prefix, tag = _resolve_defaults(adapter, registry, prefix, tag)
            if not source_images:
                source_images = getattr(adapter, "DEFAULT_SOURCE_IMAGES", False)
            tasks = adapter.discover_tasks(resolved_source)

            if json_out:
                out = []
                for t in tasks:
                    d: dict[str, Any] = {
                        "name": t.name,
                        "image": task_image(adapter, t, registry, prefix, tag, source_images=source_images),
                        "difficulty": t.difficulty, "category": t.category,
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
                    img = task_image(adapter, t, registry, prefix, tag, source_images=source_images)
                    typer.echo(f"{t.name:<55} {t.repo:<30} {img}")
            else:
                typer.echo(f"{'Task':<30} {'Diff':<8} {'Image'}")
                for t in tasks:
                    img = task_image(adapter, t, registry, prefix, tag, source_images=source_images)
                    typer.echo(f"{t.name:<30} {t.difficulty:<8} {img}")
            typer.echo(f"\n{len(tasks)} tasks")

        @cli.command()
        def batch(
            repo: Annotated[Path | None, typer.Option("--repo")] = None,
            source: Annotated[str | None, typer.Option("--source")] = None,
            bench: Annotated[str, typer.Option("--bench")] = "tb1",
            model: Annotated[str, typer.Option()] = "glm47",
            registry: Annotated[Optional[str], typer.Option()] = None,
            prefix: Annotated[Optional[str], typer.Option()] = None,
            tag: Annotated[Optional[str], typer.Option()] = None,
            concurrency: Annotated[int, typer.Option("-j")] = 5,
            eval_timeout: Annotated[int, typer.Option("--eval-timeout")] = 300,
            agent_timeout: Annotated[int, typer.Option("--agent-timeout")] = 0,
            task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
            attempts: Annotated[int, typer.Option("--attempts", "-n")] = 1,
            scenario: Annotated[str, typer.Option("--scenario")] = DEFAULT_REMOTE_TERMINAL_BENCH_SCENARIO,
            prompt_prefix: Annotated[str, typer.Option("--prompt-prefix")] = "",
            source_images: Annotated[bool, typer.Option("--source-images")] = False,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Run all tasks in parallel, then evaluate."""
            adapter = get_format_adapter(bench)
            if getattr(adapter, "USE_HARBOR_RUNNER", False):
                typer.echo(
                    f"Harbor-format benchmark '{bench}' is now driven by the "
                    f"ExternalAgentMAgent.\n"
                    f"Use harbor run directly:\n\n"
                    f"  harbor run --agent agentm_harbor:ExternalAgentMAgent "
                    f"-m {model} --concurrency {concurrency}\n\n"
                    f"The 'build', 'mirror', and 'list' commands still work here.",
                    err=True,
                )
                raise typer.Exit(1)

            from agentm.env import autoload_dotenv
            from .swebench import write_predictions

            autoload_dotenv(Path.cwd())

            resolved_source = resolve_source(repo, source, bench, adapter)
            registry, prefix, tag = _resolve_defaults(adapter, registry, prefix, tag)
            if not source_images:
                source_images = getattr(adapter, "DEFAULT_SOURCE_IMAGES", False)
            tasks = adapter.discover_tasks(resolved_source)
            if task:
                tasks = [t for t in tasks if t.name in task]
            if not tasks:
                typer.echo("No tasks found.", err=True)
                raise typer.Exit(1)

            with experiment_context(
                "sandbox", model=model, exp_id=exp_id,
                bench=bench, attempts=attempts,
            ) as exp:
                base_out = exp.artifacts_dir / f"{bench}-{model}"
                base_out.mkdir(parents=True, exist_ok=True)

                interrupted = False

                def _on_sigint(_sig: int, _frame: object) -> None:
                    nonlocal interrupted
                    if interrupted:
                        typer.echo("\nForce quit.")
                        os._exit(130)
                    interrupted = True
                    typer.echo("\nCtrl-C — terminating child processes & eval sandboxes...")
                    kill_children()
                    cleanup_eval_sessions()

                prev_handler = signal.signal(signal.SIGINT, _on_sigint)

                all_attempt_results: list[dict[str, dict]] = [{} for _ in range(attempts)]
                jobs: list[RunEvalJob] = []
                for attempt_idx in range(attempts):
                    out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
                    out.mkdir(parents=True, exist_ok=True)
                for t in tasks:
                    for attempt_idx in range(attempts):
                        out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
                        score_path = out / f"{t.name}.score.json"
                        if score_path.is_file():
                            scores = json.loads(score_path.read_text())
                            all_attempt_results[attempt_idx][t.name] = {
                                "task": t.name, "status": "done",
                                "tools": scores.get("tools", "?"), **scores,
                            }
                            continue
                        jobs.append(RunEvalJob(
                            task=t, out=out, attempt_slot=attempt_idx,
                            experiment_attempt_idx=attempt_idx if attempts > 1 else None,
                        ))

                total_jobs = len(jobs)
                typer.echo(
                    f"Batch: {len(tasks)} tasks × {attempts} attempt(s) = {total_jobs} jobs | "
                    f"bench={bench} | model={model} | concurrency={concurrency}"
                    + (f" | agent_timeout={agent_timeout}s" if agent_timeout > 0 else "")
                )
                typer.echo(f"Experiment: {exp.exp_id}")
                typer.echo(f"Results: {base_out}")
                typer.echo("")

                run_config = RunEvalConfig(
                    adapter=adapter, model=model, registry=registry,
                    prefix=prefix, tag=tag, eval_timeout=eval_timeout,
                    agent_timeout=agent_timeout, scenario=scenario,
                    # Unique part of exp_id (timestamp + uuid) is at the TAIL;
                    # a head slice collides across runs and re-attaches evals
                    # to a previous run's (deleted) sandboxes.
                    run_id=exp.exp_id[-24:], exp_id=exp.exp_id,
                    prompt_prefix=prompt_prefix, bench=bench,
                    source_images=source_images,
                )

                completed = 0
                try:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        pending: dict[Future[dict[str, Any]], RunEvalJob] = {}
                        for job in jobs:
                            if interrupted:
                                break
                            f = pool.submit(run_and_eval_one, job, run_config)
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

                                is_pass = adapter.is_pass(r) if r.get("status") == "done" else False
                                exp.record_result(TaskResult(
                                    task_id=name,
                                    status="pass" if is_pass else ("error" if r.get("error") else "fail"),
                                    score={k: v for k, v in r.items()
                                           if k in ("reward", "f2p", "p2p", "score")},
                                    session_ids=[r["session_id"]] if r.get("session_id") else [],
                                    error=r.get("error"),
                                    metadata={"attempt": attempt_idx, "tools": r.get("tools")},
                                ).to_dict())
                            except Exception as e:
                                logger.debug("Task {} failed: {}", name, e)
                                typer.echo(f"  {attempt_label}[ERROR] {name}: {e} ({completed}/{total_jobs})")
                finally:
                    signal.signal(signal.SIGINT, prev_handler)
                    cleanup_eval_sessions()
                    cleanup_experiments(prefix, model, exp.exp_id[-24:], tasks, attempts)

                for attempt_idx in range(attempts):
                    results = all_attempt_results[attempt_idx]
                    if not results:
                        continue
                    if attempts > 1:
                        typer.echo(f"\n{'=' * 60}")
                        typer.echo(f"Attempt {attempt_idx}")
                    print_summary_table(results, adapter)
                    done = sum(1 for r in results.values() if r.get("status") == "done")
                    typer.echo(f"\n{done}/{len(tasks)} completed")
                    if bench.startswith("swebench"):
                        out = base_out / f"attempt_{attempt_idx}" if attempts > 1 else base_out
                        write_predictions(results, out, model)

                if attempts > 1:
                    print_pass_at_k(all_attempt_results, tasks, base_out, adapter)

                doctor_sessions(all_attempt_results)

                n_pass = sum(
                    1 for results in all_attempt_results
                    for r in results.values()
                    if r.get("status") == "done" and adapter.is_pass(r)
                )
                n_total = sum(len(r) for r in all_attempt_results)
                exp.finish(summary={
                    "n_tasks": len(tasks), "attempts": attempts,
                    "pass": n_pass, "total": n_total,
                })

        return cli


register("sandbox", SandboxAdapter.description, SandboxAdapter)
