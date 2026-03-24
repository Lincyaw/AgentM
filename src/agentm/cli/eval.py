"""CLI eval subcommand — thin wrapper around ``rca llm-eval run --agent agentm``.

All heavy lifting is done by rcabench-platform's CLI.  This wrapper adds
AgentM-specific conveniences:

* Auto-populates ``model_name`` from the scenario config when not set.
* Defaults ``--scenario`` and ``--system-config`` to AgentM paths.
"""

from __future__ import annotations

import asyncio
import os

from rich.console import Console

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
    """Orchestrate preprocess -> rollout -> judge -> stat via rcabench-platform."""
    from rcabench_platform.v3.sdk.llm_eval.config import ConfigLoader, EvalConfig
    from rcabench_platform.v3.sdk.llm_eval.eval import BaseBenchmark

    eval_config: EvalConfig = ConfigLoader.load_eval_config(config_path)

    if exp_id_override:
        eval_config.exp_id = exp_id_override

    # Auto-populate model_name from scenario config if not explicitly set.
    if not getattr(eval_config, "model_name", None):
        from agentm.config.loader import load_scenario_config

        scenario_cfg = load_scenario_config(f"{scenario_dir}/scenario.yaml")
        eval_config.model_name = scenario_cfg.orchestrator.model

    if eval_config.db_url:
        os.environ["LLM_EVAL_DB_URL"] = eval_config.db_url

    # Resolve source_path: config field → env → default
    resolved_source_path = eval_config.source_path or os.environ.get(
        "RCABENCH_SOURCE_PATH", "/mnt/jfs/rcabench_dataset"
    )

    def _resolve_source(source: str) -> str:
        return os.path.join(resolved_source_path, source, "converted")

    benchmark = BaseBenchmark(eval_config, source_path_fn=_resolve_source)

    # -- Judge-only / stat-only shortcuts ------------------------------------
    if stat_only:
        console.print("[bold]Running stat only...[/]")
        await benchmark.stat()
        return

    if judge_only:
        console.print("[bold]Running judge + stat...[/]")
        await benchmark.judge()
        await benchmark.stat()
        return

    # -- Full pipeline -------------------------------------------------------
    console.print(
        f"[bold]Eval:[/] exp_id=[cyan]{eval_config.exp_id}[/]  "
        f"concurrency=[cyan]{eval_config.concurrency}[/]"
    )

    # EvalTracker + optional dashboard — use the framework's tracker/dashboard
    from rcabench_platform.v3.sdk.llm_eval.eval.tracker import EvalTracker

    tracker: EvalTracker | None = None
    dashboard_server_task = None

    if dashboard:
        tracker = EvalTracker(trajectory_dir="./trajectories")

        try:
            import uvicorn

            from rcabench_platform.v3.sdk.llm_eval.eval.dashboard import (
                Broadcaster,
                create_eval_dashboard,
            )

            bc = Broadcaster()
            app = create_eval_dashboard(eval_tracker=tracker, broadcaster=bc)

            _loop = asyncio.get_running_loop()

            def _tracker_to_ws(event: dict) -> None:
                try:
                    asyncio.run_coroutine_threadsafe(bc.broadcast(event), _loop)
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
        except ImportError:
            console.print("[yellow]Dashboard requires uvicorn and fastapi[/]")

    console.print("[bold]Phase 1:[/] preprocess")
    benchmark.preprocess()

    console.print("[bold]Phase 2:[/] rollout")

    from agentm.agents.eval_agent import AgentMAgent

    agent = AgentMAgent(
        scenario_dir=scenario_dir,
        config_path=system_config_path,
        max_steps=max_steps,
        timeout=timeout,
    )

    # Bridge RunContext events → EvalTracker + console output
    def on_event(sample_id: str, event: dict) -> None:
        evt_type = event.get("type", "")
        sample = event.get("sample")

        if evt_type == "started":
            data_dir = event.get("data_dir", "")
            idx = sample.dataset_index if sample else "?"
            console.print(
                f"  [blue]START[/] sample id={sample_id} idx={idx} data_dir={data_dir}"
            )
            if tracker and sample:
                tracker.register_sample(sample_id, sample.dataset_index, data_dir)

        elif evt_type == "running":
            run_id = event.get("run_id", "")
            if tracker:
                tracker.mark_running(sample_id, run_id)

        elif evt_type == "trajectory_update":
            traj_path = event.get("path", "")
            if tracker and traj_path:
                tracker.update_trajectory_path(sample_id, traj_path)

        elif evt_type == "completed":
            idx = sample.dataset_index if sample else "?"
            console.print(f"  [green]OK[/] sample id={sample_id} idx={idx}")
            if tracker:
                tracker.mark_completed(sample_id)

        elif evt_type == "failed":
            idx = sample.dataset_index if sample else "?"
            error = event.get("error", "empty response")
            console.print(f"  [red]FAIL[/] sample id={sample_id} idx={idx}: {error}")
            if tracker:
                tracker.mark_failed(sample_id, error)

        elif evt_type == "skipped":
            idx = sample.dataset_index if sample else "?"
            console.print(f"  [yellow]SKIP[/] sample id={sample_id} idx={idx}")
            if tracker and sample:
                meta = sample.meta if isinstance(sample.meta, dict) else {}
                tracker.register_sample(
                    sample_id, sample.dataset_index, meta.get("path", "")
                )
                tracker.mark_skipped(sample_id, "missing incident or data_dir")

    ok_count, fail_count = await benchmark.rollout(
        agent,
        max_samples=eval_config.max_samples,
        on_event=on_event,
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
