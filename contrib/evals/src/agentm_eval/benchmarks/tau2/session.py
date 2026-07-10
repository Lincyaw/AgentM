"""tau2-bench evaluation through full AgentM sessions.

Unlike the direct-litellm mode which calls litellm directly, this module
creates a real AgentM session per task with the tau2_gym atom installed.
The atom bridges tau2's Gym environment into AgentM's tool loop, so the
full atom stack (retry, compaction, system prompt, etc.) is exercised.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import warnings
from typing import Any

import typer
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium")


def _ensure_tau2():
    tau2_dir = os.environ.get(
        "TAU2_BENCH_DIR",
        os.path.expanduser("~/AoyangSpace/tau2-bench"),
    )
    tau2_src = os.path.join(tau2_dir, "src")
    if tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)


def _resolve_user_llm(
    agent_model: str, user_llm: str | None, user_llm_args_json: str | None,
) -> tuple[str, dict]:
    """Derive user simulator LLM from agent profile when not explicitly set."""
    if user_llm:
        ulargs = json.loads(user_llm_args_json) if user_llm_args_json else {}
        return user_llm, ulargs

    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(agent_model)
    if profile is None:
        return "openai/gpt-4.1-mini", {}
    result: dict[str, Any] = {}
    if profile.base_url:
        result["api_base"] = profile.base_url
    if profile.api_key:
        result["api_key"] = profile.api_key
    return f"openai/{profile.model}", result


def _set_env_from_profile(model: str) -> None:
    """Set OPENAI_API_BASE/KEY and TAU2_LLM_* env vars from the AgentM model
    profile so tau2's user simulator and NL evaluator use the same endpoint."""
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(model)
    if profile is None:
        return
    if profile.base_url and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = profile.base_url
    if profile.api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = profile.api_key

    for key in ("TAU2_LLM_NL_ASSERTIONS", "TAU2_LLM_ENV_INTERFACE"):
        if not os.environ.get(key):
            os.environ[key] = profile.model


async def _run_one_task(
    model: str,
    domain: str,
    task_id: str,
    scenario: str | None,
    user_llm: str,
    user_llm_args: dict,
    max_steps: int,
    eval_run_id: str | None = None,
) -> dict[str, Any]:
    from agentm.core.abi.loop import LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    config = AgentSessionConfig(
        cwd=os.getcwd(),
        model=model,
        scenario="tau2_eval",
        no_skills=True,
        auto_commit=False,
        log_trace_command=True,
        extra_extensions=[
            ("contrib.extensions.tau2_gym", {
                "domain": domain,
                "task_id": task_id,
                "user_llm": user_llm,
                "user_llm_args": user_llm_args,
                "max_steps": max_steps,
            }),
        ],
        loop_config=LoopConfig(max_turns=max_steps),
        purpose=f"tau2-eval:{domain}:{task_id}",
        eval_run_id=eval_run_id,
        eval_task_id=str(task_id),
        task_class=f"tau2:{domain}",
    )

    session = await AgentSession.create(config)
    try:
        bridge = session.get_service("tau2_bridge")
        initial_obs = bridge.last_obs if bridge else ""

        if initial_obs:
            prompt = (
                "A customer has contacted you. Here is the conversation so far:\n\n"
                f"{initial_obs}\n\n"
                "Help the customer using the available tools."
            )
        else:
            prompt = "A customer has contacted you. Greet them and ask how you can help."
        await session.prompt(prompt)
        await session.idle(timeout=600.0)

        if bridge and not bridge.terminated:
            await bridge._step("done()")

        result = {
            "task_id": task_id,
            "domain": domain,
            "session_id": session.session_id,
            "reward": bridge.reward if bridge else 0.0,
            "reward_info": bridge.reward_info if bridge else {},
            "terminated": bridge.terminated if bridge else False,
        }
        return result
    finally:
        await session.shutdown()


def run(
    model: str = typer.Option(..., "--model", "-m", help="AgentM model profile"),
    domain: str = typer.Option("mock", "--domain", "-d"),
    task_id: str = typer.Option(..., "--task-id", "-t"),
    scenario: str | None = typer.Option(None, "--scenario", help="AgentM scenario (default: minimal)"),
    user_llm: str | None = typer.Option(None, "--user-llm", help="User simulator LLM (default: same as agent)"),
    user_llm_args: str | None = typer.Option(None, "--user-llm-args", help="JSON dict"),
    max_steps: int = typer.Option(100, "--max-steps"),
):
    """Run a single tau2-bench task through an AgentM session."""
    _ensure_tau2()
    _set_env_from_profile(model)

    resolved_user_llm, ulargs = _resolve_user_llm(model, user_llm, user_llm_args)

    result = asyncio.run(
        _run_one_task(model, domain, task_id, scenario, resolved_user_llm, ulargs, max_steps)
    )

    print(f"\n{'='*60}")
    print(f"Task:       {result['task_id']}")
    print(f"Domain:     {result['domain']}")
    print(f"Session:    {result['session_id']}")
    print(f"Reward:     {result['reward']}")
    print(f"Terminated: {result['terminated']}")
    print(f"{'='*60}")

    if result["reward_info"]:
        print(f"\nReward info: {json.dumps(result['reward_info'], indent=2)[:500]}")


def batch(
    model: str = typer.Option(..., "--model", "-m"),
    domain: str = typer.Option("mock", "--domain", "-d"),
    num_tasks: int | None = typer.Option(None, "--num-tasks", "-n"),
    task_ids: str | None = typer.Option(None, "--task-ids", help="Comma-separated"),
    scenario: str | None = typer.Option(None, "--scenario"),
    user_llm: str | None = typer.Option(None, "--user-llm", help="User simulator LLM (default: same as agent)"),
    max_steps: int = typer.Option(100, "--max-steps"),
    concurrency: int = typer.Option(4, "-j"),
    exp_id: str | None = typer.Option(None, "--exp-id", help="Override experiment ID"),
):
    """Run multiple tau2-bench tasks through AgentM sessions."""
    _ensure_tau2()
    _set_env_from_profile(model)
    resolved_user_llm, user_llm_args = _resolve_user_llm(model, user_llm, None)

    from agentm_eval.experiment import experiment_context
    from agentm_eval.result import TaskResult
    from tau2.runner import get_tasks

    if task_ids:
        tid_list = task_ids.split(",")
        tasks = get_tasks(domain, task_ids=tid_list)
    else:
        tasks = get_tasks(domain)
        if num_tasks:
            tasks = tasks[:num_tasks]

    logger.info("Running {} tasks on domain={} with model={}", len(tasks), domain, model)

    with experiment_context(
        "tau2-session", model=model, exp_id=exp_id,
        domain=domain, num_tasks=len(tasks), concurrency=concurrency,
    ) as exp:
        log_file = exp.output_dir / "run.log"
        full_log = exp.output_dir / "full.log"
        logger.add(
            log_file,
            format="{time:HH:mm:ss} | {level:<7} | {message}",
            level="INFO",
            filter=lambda record: record["name"] == "__main__",
        )
        logger.add(
            full_log,
            format="{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}",
            level="DEBUG",
        )
        logger.info("Experiment {} started: domain={}, model={}, tasks={}, j={}",
                     exp.exp_id, domain, model, len(tasks), concurrency)

        results: list[dict] = []
        sem = asyncio.Semaphore(concurrency)
        total = len(tasks)
        t0 = time.monotonic()
        n_pass = 0
        n_fail = 0
        n_err = 0

        def _record(r: dict) -> None:
            nonlocal n_pass, n_fail, n_err
            results.append(r)
            reward = r.get("reward", 0.0)
            tid = str(r["task_id"])
            if r.get("error"):
                n_err += 1
            elif reward >= 1.0:
                n_pass += 1
            else:
                n_fail += 1
            exp.record_result(TaskResult(
                task_id=tid,
                status="pass" if reward >= 1.0 else ("error" if r.get("error") else "fail"),
                score={"reward": reward},
                session_ids=[r["session_id"]] if r.get("session_id") else [],
                error=r.get("error"),
                metadata={
                    "domain": domain,
                    "terminated": r.get("terminated"),
                    "reward_info": r.get("reward_info", {}),
                },
            ).to_dict())
            done = n_pass + n_fail + n_err
            elapsed = time.monotonic() - t0
            m, s = divmod(int(elapsed), 60)
            rate = f"{n_pass / done:.0%}" if done else "-"
            status = "PASS" if reward >= 1.0 else ("ERR" if r.get("error") else "FAIL")
            logger.info("[{}/{}] {} {}  pass={} ({}/{})  elapsed={}:{:02d}",
                        done, total, status, tid, rate, n_pass, done, m, s)

        async def run_with_sem(tid: str) -> None:
            async with sem:
                try:
                    r = await _run_one_task(
                        model, domain, tid, scenario, resolved_user_llm,
                        user_llm_args, max_steps, eval_run_id=exp.exp_id,
                    )
                except Exception as e:
                    import traceback
                    logger.error("Task {} failed: {}: {}", tid,
                                 type(e).__name__, e)
                    logger.debug("Task {} traceback:\n{}", tid, traceback.format_exc())
                    r = {"task_id": tid, "reward": 0.0, "error": f"{type(e).__name__}: {e}"}
                _record(r)

        async def run_all():
            await asyncio.gather(*[run_with_sem(str(t.id)) for t in tasks])

        asyncio.run(run_all())

        rewards = [r["reward"] for r in results if "reward" in r]
        passed = sum(1 for r in rewards if r >= 1.0)
        avg = sum(rewards) / len(rewards) if rewards else 0

        exp.finish(summary={
            "avg_reward": avg,
            "pass_rate": passed / len(rewards) if rewards else 0,
            "n_tasks": len(rewards),
            "n_pass": passed,
        })

    print(f"\n{'='*60}")
    print(f"Experiment: {exp.exp_id}")
    print(f"Domain: {domain}, Model: {model}")
    print(f"Tasks: {len(results)}, Avg Reward: {avg:.4f}, Pass: {passed}/{len(rewards)} ({passed/len(rewards):.1%})")
    print(f"Log: {log_file}")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r.get("reward", 0) >= 1.0 else "✗"
        sid = r.get("session_id", "")
        print(f"  {status} {r['task_id']:<20} reward={r.get('reward', 'ERR'):<6} session={sid}")


def register_commands(cli: typer.Typer) -> None:
    """Mount session-mode commands onto a parent CLI."""
    cli.command("session-run")(run)
    cli.command("session-batch")(batch)
