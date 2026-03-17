"""CLI eval subcommand — batch LLM evaluation via rcabench-platform v3."""

from __future__ import annotations

import asyncio
import os

from rcabench_platform.v3.sdk.llm_eval.config import ConfigLoader, EvalConfig
from rcabench_platform.v3.sdk.llm_eval.eval.benchmarks.base_benchmark import (
    BaseBenchmark,
    RolloutResult,
)
from rcabench_platform.v3.sdk.llm_eval.eval.data import EvaluationSample

from rich.console import Console

from agentm.cli.run import run_investigation_headless
from agentm.config.loader import load_scenario_config

console = Console()


async def run_eval(
    config_path: str,
    scenario_dir: str,
    system_config_path: str,
    exp_id_override: str | None,
    judge_only: bool,
    stat_only: bool,
    max_steps: int,
    timeout: float,
    dashboard: bool = False,
    dashboard_port: int = 8765,
    dashboard_host: str = "0.0.0.0",
) -> None:
    """Orchestrate preprocess -> rollout -> judge -> stat pipeline."""

    eval_config: EvalConfig = ConfigLoader.load_eval_config(config_path)

    if exp_id_override:
        eval_config.exp_id = exp_id_override

    # Auto-populate model_name from scenario config if not explicitly set.
    if not getattr(eval_config, "model_name", None):
        scenario_cfg = load_scenario_config(f"{scenario_dir}/scenario.yaml")
        eval_config.model_name = scenario_cfg.orchestrator.model

    if eval_config.db_url:
        os.environ["LLM_EVAL_DB_URL"] = eval_config.db_url

    benchmark = BaseBenchmark(
        eval_config,
        source_path_fn=lambda source: os.path.join(
            "/mnt/jfs/rcabench_dataset/", source, "converted"
        ),
    )

    if stat_only:
        console.print("[bold]Running stat only...[/]")
        await benchmark.stat()
        return

    if judge_only:
        console.print("[bold]Running judge + stat...[/]")
        await benchmark.judge()
        await benchmark.stat()
        return

    console.print(
        f"[bold]Eval:[/] exp_id=[cyan]{eval_config.exp_id}[/]  "
        f"concurrency=[cyan]{eval_config.concurrency}[/]"
    )

    # -- EvalTracker + optional dashboard --------------------------------------
    tracker = None
    dashboard_server_task = None

    if dashboard:
        from agentm.server.eval_tracker import EvalTracker
        from agentm.server.app import Broadcaster, create_dashboard_app

        import uvicorn

        tracker = EvalTracker(trajectory_dir="./trajectories")

        bc = Broadcaster()
        app = create_dashboard_app(
            eval_tracker=tracker,
            broadcaster=bc,
        )

        # Wire tracker -> WebSocket broadcaster
        # The tracker's _notify runs under threading.Lock, possibly from
        # the main event-loop thread.  Schedule the async broadcast safely.
        _loop = asyncio.get_running_loop()

        def _tracker_to_ws(event: dict) -> None:
            try:
                _loop.call_soon_threadsafe(_loop.create_task, bc.broadcast(event))
            except RuntimeError:
                pass

        tracker.add_listener(_tracker_to_ws)

        uvi_config = uvicorn.Config(
            app, host=dashboard_host, port=dashboard_port, log_level="warning"
        )
        server = uvicorn.Server(uvi_config)
        dashboard_server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)
        console.print(
            f"Dashboard: [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link]"
        )

    console.print("[bold]Phase 1:[/] preprocess")
    benchmark.preprocess()

    console.print("[bold]Phase 2:[/] rollout")

    async def runner(sample: EvaluationSample):
        sample_id = str(sample.id)
        incident = (sample.augmented_question or sample.raw_question or "").strip()
        meta = sample.meta if isinstance(sample.meta, dict) else {}
        data_dir: str = meta.get("path", "")

        if not data_dir or not incident:
            reasons = []
            if not data_dir:
                reasons.append("meta.path is empty")
            if not incident:
                reasons.append("augmented_question is empty")
            console.print(
                f"  [yellow]SKIP[/] sample id={sample_id} idx={sample.dataset_index}: "
                f"{', '.join(reasons)} "
                f"(meta_type={type(sample.meta).__name__}, "
                f"meta_keys={list(meta.keys())[:10] if meta else 'N/A'})"
            )
            if tracker:
                tracker.register_sample(sample_id, sample.dataset_index, data_dir)
                tracker.mark_skipped(sample_id, ", ".join(reasons))
            return RolloutResult()

        console.print(
            f"  [blue]START[/] sample id={sample_id} idx={sample.dataset_index} "
            f"data_dir={data_dir}"
        )
        if tracker:
            tracker.register_sample(sample_id, sample.dataset_index, data_dir)
            tracker.mark_running(sample_id, f"eval-{sample_id}")

        def _on_headless_start(_run_id: str, traj_path: str | None) -> None:
            if tracker and traj_path:
                tracker.update_trajectory_path(sample_id, traj_path)

        try:
            (
                response_json,
                trajectory_json,
                run_id,
                traj_file_path,
            ) = await run_investigation_headless(
                data_dir=data_dir,
                incident=incident,
                scenario_dir=scenario_dir,
                config_path=system_config_path,
                max_steps=max_steps,
                timeout=timeout,
                on_start=_on_headless_start,
            )
        except Exception as e:
            console.print(
                f"  [red]FAIL[/] sample id={sample_id} idx={sample.dataset_index}: {e}"
            )
            if tracker:
                tracker.mark_failed(sample_id, str(e))
            return RolloutResult()

        status = "[green]OK[/]" if response_json else "[red]EMPTY[/]"
        console.print(f"  {status} sample id={sample_id} idx={sample.dataset_index}")
        if tracker:
            if response_json:
                tracker.mark_completed(sample_id)
            else:
                tracker.mark_failed(sample_id, "empty response")
        return RolloutResult(
            response=response_json or "",
            trajectory_json=trajectory_json,
        )

    ok_count, fail_count = await benchmark.rollout(
        runner, max_samples=eval_config.max_samples
    )
    console.print(f"  [green]{ok_count} ok[/] / [red]{fail_count} failed[/]")

    console.print("[bold]Phase 3:[/] judge")
    await benchmark.judge()

    console.print("[bold]Phase 4:[/] stat")
    await benchmark.stat()

    # Keep dashboard alive after eval completes
    if dashboard_server_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] -- press Ctrl+C to stop."
        )
        try:
            await dashboard_server_task
        except asyncio.CancelledError:
            pass
