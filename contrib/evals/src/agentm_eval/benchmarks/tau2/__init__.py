"""tau2-bench evaluation adapter.

Wraps tau2-bench's evaluation pipeline with AgentM model profiles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult


def ensure_tau2_path() -> None:
    """Add tau2-bench's ``src`` dir to ``sys.path`` so ``import tau2`` works."""
    tau2_dir = os.environ.get(
        "TAU2_BENCH_DIR",
        str(Path.home() / "AoyangSpace" / "tau2-bench"),
    )
    tau2_src = str(Path(tau2_dir) / "src")
    if tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)


def _resolve_model(
    agentm_profile: str | None, raw_llm: str | None,
) -> tuple[str, dict]:
    if agentm_profile and raw_llm:
        raise typer.BadParameter("Pass --model OR --raw-llm, not both.")
    if raw_llm:
        return raw_llm, {}
    if agentm_profile:
        from agentm.core.lib import resolve_model_profile

        profile = resolve_model_profile(agentm_profile)
        if profile is None:
            raise KeyError(f"Profile {agentm_profile!r} not found in config.toml")
        result: dict[str, Any] = {}
        if profile.base_url:
            result["api_base"] = profile.base_url
        if profile.api_key:
            result["api_key"] = profile.api_key
        if profile.max_output_tokens:
            result["max_tokens"] = profile.max_output_tokens
        if profile.extra_body:
            result["extra_body"] = profile.extra_body
        return f"openai/{profile.model}", result
    raise typer.BadParameter("Provide --model (AgentM profile) or --raw-llm.")


class Tau2Adapter:
    name = "tau2"
    description = "tau2-bench conversational agent evaluation"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="tau2",
            help="Run tau2-bench evaluations with AgentM model profiles.",
            add_completion=False,
        )

        @cli.command()
        def run(
            model: Annotated[Optional[str], typer.Option("--model", "-m", help="AgentM model profile")] = None,
            raw_llm: Annotated[Optional[str], typer.Option("--raw-llm", help="Raw litellm model string")] = None,
            user_model: Annotated[Optional[str], typer.Option("--user-model", help="AgentM profile for user simulator")] = None,
            user_raw_llm: Annotated[Optional[str], typer.Option("--user-raw-llm", help="Raw litellm model for user")] = None,
            domain: Annotated[str, typer.Option("--domain", "-d")] = "mock",
            num_tasks: Annotated[Optional[int], typer.Option("--num-tasks", "-n")] = None,
            task_ids: Annotated[Optional[str], typer.Option("--task-ids", help="Comma-separated")] = None,
            num_trials: Annotated[int, typer.Option("--num-trials")] = 1,
            max_concurrency: Annotated[int, typer.Option("-j")] = 1,
            seed: Annotated[int, typer.Option("--seed")] = 42,
            max_steps: Annotated[int, typer.Option("--max-steps", help="Max conversation turns per task")] = 200,
            max_errors: Annotated[int, typer.Option("--max-errors", help="Max consecutive tool errors before abort")] = 10,
            max_retries: Annotated[int, typer.Option("--max-retries", help="LLM call retry count on transient errors")] = 3,
            retry_delay: Annotated[float, typer.Option("--retry-delay", help="Seconds between LLM retries")] = 1.0,
            auto_resume: Annotated[bool, typer.Option("--auto-resume/--no-auto-resume", help="Resume from existing save file")] = False,
            retrieval_config: Annotated[Optional[str], typer.Option("--retrieval-config", help="Knowledge retrieval config (banking_knowledge only)")] = None,
            save_to: Annotated[Optional[str], typer.Option("--save-to", help="Override tau2 save directory name")] = None,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
            rerun_failed: Annotated[Optional[str], typer.Option("--rerun-failed", help="Rerun infra_error tasks from a previous tau2 results.json")] = None,
        ) -> None:
            """Run tau2-bench evaluation."""
            ensure_tau2_path()

            agent_llm, agent_llm_args = _resolve_model(model, raw_llm)
            if user_model or user_raw_llm:
                user_llm, user_llm_args = _resolve_model(user_model, user_raw_llm)
            else:
                user_llm, user_llm_args = agent_llm, dict(agent_llm_args)

            # tau2's NL assertion evaluator calls litellm with the default
            # openai provider (gpt-4.1). When using a custom endpoint, set
            # OPENAI_API_BASE/KEY so the evaluator routes through it too.
            if agent_llm_args.get("api_base") and not os.environ.get("OPENAI_API_BASE"):
                os.environ["OPENAI_API_BASE"] = agent_llm_args["api_base"]
            if agent_llm_args.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = agent_llm_args["api_key"]

            from tau2.data_model.simulation import TextRunConfig
            from tau2.registry import registry
            from tau2.runner import get_tasks, run_domain, run_single_task

            from agentm_eval.benchmarks.tau2.agent import create_agentm_agent

            try:
                registry.register_agent_factory(create_agentm_agent, "agentm_agent")
            except ValueError:
                pass

            # --rerun-failed: extract infra_error task IDs from a previous run
            task_id_list = task_ids.split(",") if task_ids else None
            if rerun_failed:
                import json as _json

                prev = _json.loads(Path(rerun_failed).read_text())
                failed_ids = [
                    s["task_id"]
                    for s in prev.get("simulations", [])
                    if s.get("termination_reason") == "infrastructure_error"
                ]
                if not failed_ids:
                    typer.echo("No infrastructure_error tasks found in the previous run.")
                    raise typer.Exit(0)
                typer.echo(f"Rerunning {len(failed_ids)} failed tasks from previous run")
                task_id_list = [str(tid) for tid in failed_ids]

            config = TextRunConfig(
                domain=domain, agent="agentm_agent",
                llm_agent=agent_llm, llm_args_agent=agent_llm_args,
                llm_user=user_llm, llm_args_user=user_llm_args,
                num_trials=num_trials, max_concurrency=max_concurrency,
                task_ids=task_id_list, num_tasks=num_tasks,
                max_steps=max_steps, max_errors=max_errors,
                max_retries=max_retries, retry_delay=retry_delay,
                auto_resume=auto_resume,
                retrieval_config=retrieval_config,
                save_to=save_to,
            )

            with experiment_context(
                "tau2", model=model or raw_llm, exp_id=exp_id,
                domain=domain, num_tasks=num_tasks, num_trials=num_trials,
            ) as exp:
                if num_tasks == 1 and num_trials == 1 and not task_id_list:
                    tasks = get_tasks(domain)
                    result = run_single_task(config, tasks[0], seed=seed)
                    sims = [result]
                elif task_id_list and len(task_id_list) == 1 and num_trials == 1:
                    tasks = get_tasks(domain, task_ids=task_id_list)
                    result = run_single_task(config, tasks[0], seed=seed)
                    sims = [result]
                else:
                    results = run_domain(config)
                    sims = getattr(results, "simulations", [])

                for s in sims:
                    reward = s.reward_info.reward if s.reward_info else 0.0
                    exp.record_result(TaskResult(
                        task_id=s.task_id,
                        status="pass" if reward >= 1.0 else "fail",
                        score={"reward": reward},
                        metadata={
                            "n_messages": len(s.messages),
                            "db_match": (
                                s.reward_info.db_check.db_match
                                if s.reward_info and s.reward_info.db_check else None
                            ),
                        },
                    ).to_dict())

                rewards = [s.reward_info.reward for s in sims if s.reward_info]
                if rewards:
                    avg = sum(rewards) / len(rewards)
                    passed = sum(1 for r in rewards if r >= 1.0)
                    typer.echo(f"\ntasks: {len(rewards)}")
                    typer.echo(f"avg_reward: {avg:.4f}")
                    typer.echo(f"pass_rate: {passed}/{len(rewards)} = {passed/len(rewards):.2%}")
                    for s in sims:
                        r = s.reward_info.reward if s.reward_info else "N/A"
                        typer.echo(f"  {s.task_id:<30} reward={r}")

                    exp.finish(summary={
                        "avg_reward": avg,
                        "pass_rate": passed / len(rewards),
                        "n_tasks": len(rewards),
                    })

        @cli.command("list-profiles")
        def list_profiles() -> None:
            """List available AgentM model profiles."""
            from agentm.core.lib.user_config import load_user_config

            config = load_user_config()
            if not config.models:
                typer.echo("No AgentM profiles found.")
                return
            typer.echo(f"\n{'Name':<25} {'Model':<30} {'Base URL':<40}")
            typer.echo("-" * 95)
            for name, p in sorted(config.models.items()):
                typer.echo(f"{name:<25} {p.model:<30} {p.base_url or '(default)':<40}")

        @cli.command("list-domains")
        def list_domains() -> None:
            """List available tau2-bench domains."""
            ensure_tau2_path()
            from tau2.registry import registry

            info = registry.get_info()
            typer.echo(f"Domains: {', '.join(info.domains)}")

        from agentm_eval.benchmarks.tau2 import session as _session
        _session.register_commands(cli)

        return cli


register("tau2", Tau2Adapter.description, Tau2Adapter)
