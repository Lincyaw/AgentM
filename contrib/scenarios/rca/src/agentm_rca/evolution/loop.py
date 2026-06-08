"""Self-evolution loop orchestrator.

Thin orchestration over existing infrastructure:
- ``rca llm-eval run`` for case execution (handles concurrency, retries, judge)
- ``eval.db`` for results (correctness, response, trajectory)
- AgentM sessions for observer/distiller LLM calls
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm_rca.evolution.distiller import DistilledSkill, distill_skill
from agentm_rca.evolution.observer import DivergenceReport, observe_case

_logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    iteration: int
    skill: DistilledSkill | None
    baseline_accuracy: float
    skill_accuracy: float
    accepted: bool
    reports: list[DivergenceReport] = field(default_factory=list)


@dataclass
class EvolutionResult:
    iterations: list[IterationResult] = field(default_factory=list)
    accepted_skills: list[DistilledSkill] = field(default_factory=list)
    initial_accuracy: float = 0.0
    final_accuracy: float = 0.0


def _resolve_provider(model_profile: str) -> tuple[str, dict[str, Any]]:
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import resolve_model_profile

    profile = resolve_model_profile(model_profile)
    if profile is None:
        raise ValueError(f"no ~/.agentm/config.toml profile: {model_profile!r}")
    return DEFAULT_PROVIDER_REGISTRY.build(profile.provider, profile.to_build_config())


def _run_eval(
    *,
    config_path: str,
    exp_id: str,
    scenario: str,
    concurrency: int,
    limit: int | None = None,
) -> None:
    """Shell out to ``rca llm-eval run`` — the existing eval pipeline."""
    cmd = [
        "uv", "run", "--no-sync", "rca", "llm-eval", "run", config_path,
        "-a", "agentm",
        "--ak", f"scenario={scenario}",
        "-n", str(concurrency),
        "--exp-id", exp_id,
    ]
    if limit is not None:
        cmd.extend(["-l", str(limit)])

    _logger.info("Running eval: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        _logger.error("Eval failed (exit %d): %s", result.returncode, result.stderr[-500:])
        raise RuntimeError(f"rca llm-eval run failed: {result.stderr[-200:]}")
    _logger.info("Eval complete: %s", exp_id)


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
    env = {**os.environ, **env_vars}
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
    _env = env_vars or {}

    result = EvolutionResult()

    # Step 1: Baseline on test set
    baseline_exp = f"{exp_id_prefix}-baseline"
    _logger.info("=== Step 1: baseline eval on test set (exp=%s) ===", baseline_exp)
    _run_eval(
        config_path=eval_config, exp_id=baseline_exp,
        scenario=scenario, concurrency=concurrency,
        limit=test_limit,
    )
    baseline_results = _query_results(db_path, baseline_exp)
    baseline_acc = _accuracy(baseline_results)
    result.initial_accuracy = baseline_acc
    _logger.info("Baseline accuracy: %.1f%% (%d/%d)",
                 baseline_acc * 100,
                 sum(1 for r in baseline_results if r.get("correct")),
                 len(baseline_results))

    for iteration in range(max_iterations):
        _logger.info("=== Iteration %d/%d ===", iteration + 1, max_iterations)

        # Step 2: Run train cases
        train_exp = f"{exp_id_prefix}-train-{iteration}"
        _logger.info("Running train cases (exp=%s)...", train_exp)
        _run_eval(
            config_path=eval_config, exp_id=train_exp,
            scenario=scenario, concurrency=concurrency,
            limit=train_limit,
        )
        train_results = _query_results(db_path, train_exp)
        train_acc = _accuracy(train_results)
        _logger.info("Train accuracy: %.1f%% (%d/%d)",
                     train_acc * 100,
                     sum(1 for r in train_results if r.get("correct")),
                     len(train_results))

        # Step 3: Find failures
        failed = [r for r in train_results if not r.get("correct")]
        if not failed:
            _logger.info("No failures in training set. Stopping.")
            result.iterations.append(IterationResult(
                iteration=iteration + 1, skill=None,
                baseline_accuracy=baseline_acc, skill_accuracy=baseline_acc,
                accepted=False,
            ))
            break

        # Step 4: Observe failures
        _logger.info("Observing %d failures...", len(failed))
        reports: list[DivergenceReport] = []
        for row in failed:
            source = row["source"]
            case_dir = os.path.join(data_root, source)
            meta = json.loads(row.get("meta") or "{}")
            session_log_id = meta.get("session_log_id", "")
            traj_path = os.path.join(
                os.getcwd(), ".agentm", "observability",
                f"{session_log_id}.jsonl",
            ) if session_log_id else ""

            report = await observe_case(
                case_id=source,
                data_dir=case_dir,
                agent_response=row.get("response", ""),
                trajectory_path=traj_path,
                provider_tuple=provider_tuple,
            )
            reports.append(report)
            _logger.info("  %s: %d divergence points, lesson=%s",
                         source, len(report.divergence_points), report.key_lesson[:80])

        # Step 5: Distill
        _logger.info("Distilling failure patterns...")
        skill = await distill_skill(reports=reports, provider_tuple=provider_tuple)

        if skill is None:
            _logger.info("Could not distill a skill. Stopping.")
            result.iterations.append(IterationResult(
                iteration=iteration + 1, skill=None,
                baseline_accuracy=baseline_acc, skill_accuracy=baseline_acc,
                accepted=False, reports=reports,
            ))
            break

        _logger.info("Distilled: %s (pattern=%s, freq=%d)",
                     skill.name, skill.pattern_category, skill.pattern_frequency)

        # Install skill
        skill_dir = Path(skill_output_dir) / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill.content)
        _logger.info("Wrote skill: %s", skill_dir / "SKILL.md")

        # Step 6: Backtest with skill on test set
        skill_exp = f"{exp_id_prefix}-skill-{iteration}"
        _logger.info("Backtesting with skill (exp=%s)...", skill_exp)
        _run_eval(
            config_path=eval_config, exp_id=skill_exp,
            scenario=scenario, concurrency=concurrency,
            limit=test_limit,
        )
        skill_results = _query_results(db_path, skill_exp)
        skill_acc = _accuracy(skill_results)
        _logger.info("Skill accuracy: %.1f%% (baseline: %.1f%%)",
                     skill_acc * 100, baseline_acc * 100)

        # Step 7: Accept or reject
        accepted = skill_acc > baseline_acc
        if accepted:
            _logger.info("ACCEPTED %s (%.1f%% → %.1f%%)",
                         skill.name, baseline_acc * 100, skill_acc * 100)
            result.accepted_skills.append(skill)
            baseline_acc = skill_acc
        else:
            _logger.info("REJECTED %s (%.1f%% vs %.1f%%)",
                         skill.name, skill_acc * 100, baseline_acc * 100)
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
    _logger.info("=== Evolution complete: %.1f%% → %.1f%%, %d skills accepted ===",
                 result.initial_accuracy * 100, result.final_accuracy * 100,
                 len(result.accepted_skills))
    return result
