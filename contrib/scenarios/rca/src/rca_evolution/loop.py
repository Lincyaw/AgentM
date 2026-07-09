"""Self-evolution loop orchestrator.

Thin orchestration over existing infrastructure:
- ``rca llm-eval run`` for case execution (handles concurrency, retries, judge)
- ``eval.db`` for results (correctness, response, trajectory)
- AgentM sessions for observer/distiller LLM calls
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from rca_evolution.distiller import DistilledSkill, distill_skill
from rca_evolution.observer import DivergenceReport, observe_case

@dataclass(slots=True)
class IterationResult:
    iteration: int
    skill: DistilledSkill | None
    baseline_accuracy: float
    skill_accuracy: float
    accepted: bool
    reports: list[DivergenceReport] = field(default_factory=list)

@dataclass(slots=True)
class EvolutionResult:
    iterations: list[IterationResult] = field(default_factory=list)
    accepted_skills: list[DistilledSkill] = field(default_factory=list)
    initial_accuracy: float = 0.0
    final_accuracy: float = 0.0

def _resolve_provider(model_profile: str) -> tuple[str, dict[str, Any]]:
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import (
        apply_reasoning_effort,
        resolve_provider_model,
    )

    provider_id, _model_id, profile = resolve_provider_model(model_flag=model_profile)
    build_config = profile.to_build_config() if profile else {"model": _model_id}
    apply_reasoning_effort(build_config, None)
    return DEFAULT_PROVIDER_REGISTRY.build(provider_id, dict(build_config))


def _eval_observability_dir() -> Path:
    """Resolve the observability directory used by the eval subprocess."""
    from agentm.core.lib.observability_dir import resolve_observability_dir

    return resolve_observability_dir()


def _effective_env(env_override: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment passed to eval subprocesses."""
    return {**os.environ, **(env_override or {})}


def _run_eval(
    *,
    config_path: str,
    exp_id: str,
    scenario: str,
    concurrency: int,
    limit: int | None = None,
    max_turns: int = 100,
    extra_ak: dict[str, str] | None = None,
    env_override: dict[str, str] | None = None,
) -> None:
    """Shell out to ``rca llm-eval run``.

    Renders ``${VAR}`` placeholders in the config via ``envsubst`` first.
    """
    env = _effective_env(env_override)

    rendered = subprocess.run(
        ["envsubst"], input=Path(config_path).read_text(),
        capture_output=True, text=True, env=env,
    )
    if rendered.returncode != 0:
        raise RuntimeError(f"envsubst failed: {rendered.stderr}")

    fd, rendered_path = tempfile.mkstemp(suffix=".yaml", prefix="evo-eval-")
    os.write(fd, rendered.stdout.encode())
    os.close(fd)

    try:
        cmd = [
            "uv", "run", "rca", "llm-eval", "run", rendered_path,
            "-a", "agentm",
            "--ak", f"scenario={scenario}",
            "--ak", f"max_turns={max_turns}",
            "-n", str(concurrency),
            "--exp-id", exp_id,
        ]
        for k, v in (extra_ak or {}).items():
            cmd.extend(["--ak", f"{k}={v}"])
        if limit is not None:
            cmd.extend(["-l", str(limit)])

        logger.info(f'Running eval: exp={exp_id} concurrency={concurrency} limit={limit} cmd={" ".join(cmd)}')
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
        except subprocess.TimeoutExpired:
            logger.warning(f"Eval timed out for {exp_id}; using partial results")
            return
        if result.stderr:
            logger.debug(f"Eval stderr: {result.stderr[-1000:]}")
        if result.returncode != 0:
            logger.error(f"Eval failed (exit {result.returncode}): {result.stderr[-500:]}")
            raise RuntimeError(f"rca llm-eval run failed: {result.stderr[-200:]}")
        logger.info(f"Eval complete: {exp_id}")
    finally:
        os.unlink(rendered_path)

def _query_results(db_path: str, exp_id: str) -> list[dict[str, Any]]:
    """Read judged results from eval.db for an experiment."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT source, correct, response, trace_id, meta "
        "FROM evaluation_data WHERE exp_id = ? AND stage = 'judged'",
        (exp_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]

def _accuracy(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct")) / len(results)

def _write_eval_config(
    *,
    template_path: str,
    exp_id: str,
    tags: list[str],
    env_vars: dict[str, str],
) -> str:
    """Render an eval config YAML with envsubst and return the temp path."""
    env = _effective_env(env_vars)
    result = subprocess.run(
        ["envsubst"],
        input=Path(template_path).read_text(),
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"envsubst failed: {result.stderr}")

    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="evolution-eval-")
    os.write(fd, result.stdout.encode())
    os.close(fd)
    return path

async def run_evolution_loop(
    *,
    eval_config: str,
    db_path: str,
    data_root: str,
    skill_output_dir: str,
    scenario: str = "rca:baseline",
    model_profile: str = "litellm-dsv4flash-nothink",
    exp_id_prefix: str = "evolution",
    concurrency: int = 5,
    train_limit: int | None = 20,
    test_limit: int | None = 10,
    max_iterations: int = 3,
    env_vars: dict[str, str] | None = None,
) -> EvolutionResult:
    """Run the self-evolution loop.

    Steps:
    1. ``rca llm-eval run`` baseline on test cases → eval.db
    2. ``rca llm-eval run`` on train cases → eval.db
    3. Query eval.db for failed train cases
    4. Observer: analyze each failure (AgentM session)
    5. Distiller: cluster → SKILL.md (AgentM session)
    6. ``rca llm-eval run`` with skill on test cases → eval.db
    7. Compare accuracy, accept/reject
    """
    provider_tuple = _resolve_provider(model_profile)
    _, provider_cfg = provider_tuple

    model_name = provider_cfg.get("model", model_profile)
    eval_env: dict[str, str] = {
        "MODEL_NAME": model_name,
        "JUDGE_MODEL": model_name,
        "AGENTM_MODEL": model_name,
        "RCA_DATASET_ROOT": data_root,
        "AGENTM_PROVIDER": "openai",
        "OPENAI_API_KEY": provider_cfg.get("api_key", ""),
        "OPENAI_VERIFY_SSL": "false",
    }
    if provider_cfg.get("base_url"):
        eval_env["OPENAI_BASE_URL"] = provider_cfg["base_url"]
    eval_env.update(env_vars or {})
    eval_obs_dir = _eval_observability_dir()

    result = EvolutionResult()

    eval_ak = {"model": model_profile}

    # Step 1: Baseline on test set
    baseline_exp = f"{exp_id_prefix}-baseline"
    logger.info(f"=== Step 1: baseline eval on test set (exp={baseline_exp}) ===")
    _run_eval(
        config_path=eval_config, exp_id=baseline_exp,
        scenario=scenario, concurrency=concurrency,
        limit=test_limit, extra_ak=eval_ak, env_override=eval_env,
    )
    baseline_results = _query_results(db_path, baseline_exp)
    baseline_acc = _accuracy(baseline_results)
    result.initial_accuracy = baseline_acc
    logger.info(f'Baseline accuracy: {baseline_acc * 100:.1f}% ({sum(1 for r in baseline_results if r.get("correct"))}/{len(baseline_results)})')

    for iteration in range(max_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")

        # Step 2: Run train cases
        train_exp = f"{exp_id_prefix}-train-{iteration}"
        logger.info(f"Running train cases (exp={train_exp})...")
        _run_eval(
            config_path=eval_config, exp_id=train_exp,
            scenario=scenario, concurrency=concurrency,
            limit=train_limit, extra_ak=eval_ak, env_override=eval_env,
        )
        train_results = _query_results(db_path, train_exp)
        train_acc = _accuracy(train_results)
        logger.info(f'Train accuracy: {train_acc * 100:.1f}% ({sum(1 for r in train_results if r.get("correct"))}/{len(train_results)})')

        # Step 3: Find failures
        failed = [r for r in train_results if not r.get("correct")]
        if not failed:
            logger.info("No failures in training set. Stopping.")
            result.iterations.append(IterationResult(
                iteration=iteration + 1, skill=None,
                baseline_accuracy=baseline_acc, skill_accuracy=baseline_acc,
                accepted=False,
            ))
            break

        # Step 4: Observe failures
        logger.info(f"Observing {len(failed)} failures...")
        reports: list[DivergenceReport] = []
        for row in failed:
            source = row["source"]
            case_dir = os.path.join(data_root, source)
            meta = json.loads(row.get("meta") or "{}")
            session_log_id = meta.get("session_log_id", "")
            traj_path = (
                str(eval_obs_dir / f"{session_log_id}.jsonl")
                if session_log_id
                else ""
            )

            report = await observe_case(
                case_id=source,
                data_dir=case_dir,
                agent_response=row.get("response", ""),
                trajectory_path=traj_path,
                provider_tuple=provider_tuple,
            )
            reports.append(report)
            logger.info(f"  {source}: {len(report.divergence_points)} divergence points, lesson={report.key_lesson[:80]}")

        # Step 5: Distill
        logger.info("Distilling failure patterns...")
        skill = await distill_skill(
            reports=reports, provider_tuple=provider_tuple,
            skill_output_dir=skill_output_dir,
        )

        if skill is None:
            logger.info("Could not distill a skill. Stopping.")
            result.iterations.append(IterationResult(
                iteration=iteration + 1, skill=None,
                baseline_accuracy=baseline_acc, skill_accuracy=baseline_acc,
                accepted=False, reports=reports,
            ))
            break

        logger.info(f"Distilled: {skill.name} action={skill.action} (pattern={skill.pattern_category}, freq={skill.pattern_frequency})")

        # Handle retire
        if skill.action == "retire":
            retire_dir = Path(skill_output_dir) / skill.name
            if (retire_dir / "SKILL.md").exists():
                (retire_dir / "SKILL.md").unlink()
                if retire_dir.exists():
                    retire_dir.rmdir()
                logger.info(f"Retired skill: {skill.name} ({skill.reason})")
            result.iterations.append(IterationResult(
                iteration=iteration + 1, skill=skill,
                baseline_accuracy=result.initial_accuracy,
                skill_accuracy=baseline_acc,
                accepted=True, reports=reports,
            ))
            continue

        # Install skill (create or update)
        skill_dir = Path(skill_output_dir) / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill.content)
        logger.info(f'Wrote skill ({skill.action}): {skill_dir / "SKILL.md"}')

        # Step 6: Backtest with skill on test set
        skill_exp = f"{exp_id_prefix}-skill-{iteration}"
        logger.info(f"Backtesting with skill (exp={skill_exp})...")
        _run_eval(
            config_path=eval_config, exp_id=skill_exp,
            scenario=scenario, concurrency=concurrency,
            limit=test_limit, extra_ak=eval_ak, env_override=eval_env,
        )
        skill_results = _query_results(db_path, skill_exp)
        skill_acc = _accuracy(skill_results)
        logger.info(f"Skill accuracy: {skill_acc * 100:.1f}% (baseline: {baseline_acc * 100:.1f}%)")

        # Step 7: Accept or reject
        accepted = skill_acc > baseline_acc
        if accepted:
            logger.info(f"ACCEPTED {skill.name} ({baseline_acc * 100:.1f}% → {skill_acc * 100:.1f}%)")
            result.accepted_skills.append(skill)
            baseline_acc = skill_acc
        else:
            logger.info(f"REJECTED {skill.name} ({skill_acc * 100:.1f}% vs {baseline_acc * 100:.1f}%)")
            (skill_dir / "SKILL.md").unlink(missing_ok=True)
            if skill_dir.exists():
                skill_dir.rmdir()

        result.iterations.append(IterationResult(
            iteration=iteration + 1, skill=skill,
            baseline_accuracy=result.initial_accuracy,
            skill_accuracy=skill_acc,
            accepted=accepted, reports=reports,
        ))

    result.final_accuracy = baseline_acc
    logger.info(f"=== Evolution complete: {result.initial_accuracy * 100:.1f}% → {result.final_accuracy * 100:.1f}%, {len(result.accepted_skills)} skills accepted ===")
    return result
