"""CLI eval subcommand — batch LLM evaluation via rcabench-platform v3."""

from __future__ import annotations

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
    dataset_path: str | None,
) -> None:
    """Orchestrate preprocess → rollout → judge → stat pipeline."""

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
        # source_path_fn=lambda source: os.path.join(dataset_path, "data", source),
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

    console.print("[bold]Phase 1:[/] preprocess")
    benchmark.preprocess()

    console.print("[bold]Phase 2:[/] rollout")

    async def runner(sample: EvaluationSample):
        incident = (sample.augmented_question or sample.raw_question or "").strip()
        data_dir: str = (
            (sample.meta or {}).get("path", "") if isinstance(sample.meta, dict) else ""
        )

        if not data_dir or not incident:
            return RolloutResult()

        response_json, trajectory_json = await run_investigation_headless(
            data_dir=data_dir,
            incident=incident,
            scenario_dir=scenario_dir,
            config_path=system_config_path,
            max_steps=max_steps,
            timeout=timeout,
        )
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
