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


def _ensure_tau2_importable() -> None:
    tau2_dir = os.environ.get(
        "TAU2_BENCH_DIR",
        str(Path.home() / "AoyangSpace" / "tau2-bench"),
    )
    tau2_src = str(Path(tau2_dir) / "src")
    if tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)


def _agentm_config_path() -> Path:
    home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(home) / "config.toml"


def _load_agentm_profiles() -> dict[str, dict[str, Any]]:
    import tomllib

    cfg_path = _agentm_config_path()
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "rb") as f:
        return tomllib.load(f).get("models", {})


def _resolve_profile(profile_name: str) -> dict[str, Any]:
    profiles = _load_agentm_profiles()
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise KeyError(f"Profile {profile_name!r} not found. Available: {available}")
    p = profiles[profile_name]
    result: dict[str, Any] = {"model": f"openai/{p['model']}", "api_base": p.get("base_url"), "api_key": p.get("api_key")}
    if p.get("max_output_tokens"):
        result["max_tokens"] = p["max_output_tokens"]
    if p.get("extra_body"):
        result["extra_body"] = p["extra_body"]
    return {k: v for k, v in result.items() if v is not None}


def _resolve_model(
    agentm_profile: str | None, raw_llm: str | None,
) -> tuple[str, dict]:
    if agentm_profile and raw_llm:
        raise typer.BadParameter("Pass --model OR --raw-llm, not both.")
    if raw_llm:
        return raw_llm, {}
    if agentm_profile:
        resolved = _resolve_profile(agentm_profile)
        model = resolved.pop("model")
        return model, resolved
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
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Run tau2-bench evaluation."""
            _ensure_tau2_importable()

            agent_llm, agent_llm_args = _resolve_model(model, raw_llm)
            if user_model or user_raw_llm:
                user_llm, user_llm_args = _resolve_model(user_model, user_raw_llm)
            else:
                user_llm, user_llm_args = agent_llm, dict(agent_llm_args)

            from tau2.data_model.simulation import TextRunConfig
            from tau2.registry import registry
            from tau2.runner import get_tasks, run_domain, run_single_task

            from agentm_eval.adapters.tau2_agent import create_agentm_agent

            try:
                registry.register_agent_factory(create_agentm_agent, "agentm_agent")
            except ValueError:
                pass

            task_id_list = task_ids.split(",") if task_ids else None

            config = TextRunConfig(
                domain=domain, agent="agentm_agent",
                llm_agent=agent_llm, llm_args_agent=agent_llm_args,
                llm_user=user_llm, llm_args_user=user_llm_args,
                num_trials=num_trials, max_concurrency=max_concurrency,
                task_ids=task_id_list, num_tasks=num_tasks,
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
            profiles = _load_agentm_profiles()
            if not profiles:
                typer.echo("No AgentM profiles found.")
                return
            typer.echo(f"\n{'Name':<25} {'Model':<30} {'Base URL':<40}")
            typer.echo("-" * 95)
            for name, p in sorted(profiles.items()):
                typer.echo(f"{name:<25} {p.get('model', '?'):<30} {p.get('base_url', '(default)'):<40}")

        @cli.command("list-domains")
        def list_domains() -> None:
            """List available tau2-bench domains."""
            _ensure_tau2_importable()
            from tau2.registry import registry

            info = registry.get_info()
            typer.echo(f"Domains: {', '.join(info.domains)}")

        return cli


register("tau2", Tau2Adapter.description, Tau2Adapter)
