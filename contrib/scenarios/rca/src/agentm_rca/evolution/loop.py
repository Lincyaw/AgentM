"""Main evolution loop orchestrator.

Ties together observe -> distill -> backtest -> select into an iterative
self-improvement loop that produces validated SKILL.md files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm_rca.evolution.distiller import DistilledSkill, distill_skill
from agentm_rca.evolution.observer import DivergenceReport, observe_case

_logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case_id: str
    response: str  # raw JSON response
    correct: bool
    data_dir: str
    trajectory_path: str = ""


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


async def _run_case(
    case_id: str,
    data_dir: str,
    scenario: str,
    provider_tuple: tuple[str, dict[str, Any]],
) -> CaseResult:
    """Run one RCA case through AgentM and return the result."""
    from agentm_rca.eval.agent import AgentMAgent

    agent = AgentMAgent(scenario=scenario, provider_tuple=provider_tuple)
    try:
        result = await agent.run(
            incident="Investigate the root cause of the SLO violation.",
            data_dir=data_dir,
        )
        response = result.response
        metadata = result.metadata or {}

        # Extract trajectory path from metadata
        trajectory_path = ""
        trace_id = result.trace_id or ""
        if trace_id:
            obs_dir = os.path.join(os.getcwd(), ".agentm", "observability")
            # The session JSONL is named by session_log_id
            session_log_id = metadata.get("session_log_id", "")
            if session_log_id:
                trajectory_path = os.path.join(obs_dir, f"{session_log_id}.jsonl")

        # Check correctness by comparing against GT
        causal_graph_path = Path(data_dir) / "causal_graph.json"
        injection_path = Path(data_dir) / "injection.json"
        correct = False

        if causal_graph_path.exists() and injection_path.exists():
            with open(injection_path) as f:
                injection = json.load(f)
            injected_apps = {
                e["app"].lower()
                for e in injection.get("engine_config_summary", [])
            }

            try:
                resp_data = json.loads(response)
                agent_services = {
                    rc.get("service", "").lower()
                    for rc in resp_data.get("root_causes", [])
                    if isinstance(rc, dict) and rc.get("service")
                }
                correct = bool(injected_apps) and injected_apps.issubset(agent_services)
            except (json.JSONDecodeError, TypeError):
                correct = False

        return CaseResult(
            case_id=case_id,
            response=response,
            correct=correct,
            data_dir=data_dir,
            trajectory_path=trajectory_path,
        )
    except Exception as exc:
        _logger.error("Case %s failed: %s", case_id, exc)
        return CaseResult(
            case_id=case_id,
            response=json.dumps({"root_causes": [], "propagation": []}),
            correct=False,
            data_dir=data_dir,
        )


async def _run_cases(
    case_ids: list[str],
    data_root: str,
    scenario: str,
    provider_tuple: tuple[str, dict[str, Any]],
    concurrency: int = 3,
) -> list[CaseResult]:
    """Run multiple cases with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[CaseResult] = []

    async def _bounded_run(case_id: str) -> CaseResult:
        async with semaphore:
            data_dir = os.path.join(data_root, case_id)
            _logger.info("Running case: %s", case_id)
            result = await _run_case(case_id, data_dir, scenario, provider_tuple)
            _logger.info(
                "Case %s: %s",
                case_id,
                "CORRECT" if result.correct else "INCORRECT",
            )
            return result

    tasks = [_bounded_run(cid) for cid in case_ids]
    results = await asyncio.gather(*tasks)
    return list(results)


async def _observe_failures(
    case_results: list[CaseResult],
    provider_config: dict[str, Any],
    concurrency: int = 5,
) -> list[DivergenceReport]:
    """Run the observer on all failed cases."""
    failed = [r for r in case_results if not r.correct]
    if not failed:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    reports: list[DivergenceReport] = []

    async def _bounded_observe(cr: CaseResult) -> DivergenceReport:
        async with semaphore:
            return await observe_case(
                trajectory_path=cr.trajectory_path,
                data_dir=cr.data_dir,
                agent_response=cr.response,
                provider_config=provider_config,
            )

    tasks = [_bounded_observe(cr) for cr in failed]
    reports = await asyncio.gather(*tasks)
    return list(reports)


def _install_skill_to_scenario(skill: DistilledSkill, skill_output_dir: str) -> str:
    """Write a distilled skill to disk as a SKILL.md in a named directory."""
    skill_dir = Path(skill_output_dir) / skill.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(skill.content)
    _logger.info("Wrote skill: %s", skill_path)
    return str(skill_path)


async def run_evolution_loop(
    *,
    train_cases: list[str],
    test_cases: list[str],
    data_root: str,
    skill_output_dir: str,
    scenario: str = "rca:baseline",
    model: str = "DeepSeek-V4-pro",
    base_url: str = "http://100.114.89.62:8088/v1",
    api_key: str = "sk-DLbuXPx8tzeb29atiigWEIXoU9P0xkh_2amQNqJHpMk",
    max_iterations: int = 3,
    concurrency: int = 3,
) -> EvolutionResult:
    """Run the full self-evolution loop.

    1. Run baseline eval on train_cases (no skills) -> record results
    2. Collect DivergenceReports for failed cases
    3. Distill reports into a candidate skill
    4. Run eval on test_cases WITH the skill vs WITHOUT
    5. If accuracy improves -> accept skill, write to skill_output_dir
    6. Repeat for max_iterations (each iteration learns from remaining failures)

    Args:
        train_cases: Case IDs for learning (observe + distill).
        test_cases: Case IDs for validation (backtest).
        data_root: Path to datasets/ops-lite/cases/.
        skill_output_dir: Where to write evolved skills.
        scenario: Which scenario variant to use for eval.
        model: LLM model name for analysis.
        base_url: LLM endpoint URL.
        api_key: LLM API key.
        max_iterations: Maximum evolution iterations.
        concurrency: Max concurrent case runs.

    Returns:
        EvolutionResult with iteration details and accepted skills.
    """
    from agentm_rca.eval.agent import _build_provider

    provider_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }

    # Build provider tuple for running the agent
    # Set env vars so _build_provider can find them
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    provider_tuple = _build_provider("openai", model)

    result = EvolutionResult()

    # Step 1: Baseline eval on test cases (for comparison)
    _logger.info("=== Running baseline eval on %d test cases ===", len(test_cases))
    baseline_test_results = await _run_cases(
        test_cases, data_root, scenario, provider_tuple, concurrency=concurrency
    )
    baseline_accuracy = (
        sum(1 for r in baseline_test_results if r.correct) / len(baseline_test_results)
        if baseline_test_results
        else 0.0
    )
    result.initial_accuracy = baseline_accuracy
    _logger.info("Baseline test accuracy: %.1f%% (%d/%d)",
                 baseline_accuracy * 100,
                 sum(1 for r in baseline_test_results if r.correct),
                 len(baseline_test_results))

    for iteration in range(max_iterations):
        _logger.info("=== Evolution iteration %d/%d ===", iteration + 1, max_iterations)

        # Step 2: Run train cases and observe failures
        _logger.info("Running %d train cases...", len(train_cases))
        train_results = await _run_cases(
            train_cases, data_root, scenario, provider_tuple, concurrency=concurrency
        )
        train_accuracy = (
            sum(1 for r in train_results if r.correct) / len(train_results)
            if train_results
            else 0.0
        )
        _logger.info("Train accuracy: %.1f%% (%d/%d)",
                     train_accuracy * 100,
                     sum(1 for r in train_results if r.correct),
                     len(train_results))

        failed_results = [r for r in train_results if not r.correct]
        if not failed_results:
            _logger.info("All train cases correct; no failures to learn from.")
            result.iterations.append(IterationResult(
                iteration=iteration + 1,
                skill=None,
                baseline_accuracy=baseline_accuracy,
                skill_accuracy=baseline_accuracy,
                accepted=False,
            ))
            break

        # Step 3: Observe failures
        _logger.info("Observing %d failures...", len(failed_results))
        reports = await _observe_failures(
            failed_results, provider_config, concurrency=concurrency
        )

        # Step 4: Distill into a skill
        _logger.info("Distilling failure patterns...")
        skill = await distill_skill(reports=reports, provider_config=provider_config)

        if skill is None:
            _logger.info("Could not distill a clear pattern. Stopping.")
            result.iterations.append(IterationResult(
                iteration=iteration + 1,
                skill=None,
                baseline_accuracy=baseline_accuracy,
                skill_accuracy=baseline_accuracy,
                accepted=False,
                reports=reports,
            ))
            break

        _logger.info("Distilled skill: %s (pattern: %s, frequency: %d)",
                     skill.name, skill.pattern_category, skill.pattern_frequency)

        # Step 5: Backtest — install skill and re-run test cases
        _install_skill_to_scenario(skill, skill_output_dir)

        _logger.info("Backtesting with skill on %d test cases...", len(test_cases))
        skill_test_results = await _run_cases(
            test_cases, data_root, scenario, provider_tuple, concurrency=concurrency
        )
        skill_accuracy = (
            sum(1 for r in skill_test_results if r.correct) / len(skill_test_results)
            if skill_test_results
            else 0.0
        )
        _logger.info("Skill test accuracy: %.1f%% (baseline: %.1f%%)",
                     skill_accuracy * 100, baseline_accuracy * 100)

        # Step 6: Accept or reject
        accepted = skill_accuracy > baseline_accuracy
        if accepted:
            _logger.info("ACCEPTED skill '%s' (%.1f%% -> %.1f%%)",
                         skill.name, baseline_accuracy * 100, skill_accuracy * 100)
            result.accepted_skills.append(skill)
            # Update baseline for next iteration
            baseline_accuracy = skill_accuracy
        else:
            _logger.info("REJECTED skill '%s' (no improvement: %.1f%% vs %.1f%%)",
                         skill.name, skill_accuracy * 100, baseline_accuracy * 100)
            # Remove the rejected skill from disk
            skill_path = Path(skill_output_dir) / skill.name / "SKILL.md"
            if skill_path.exists():
                skill_path.unlink()
                skill_path.parent.rmdir()

        result.iterations.append(IterationResult(
            iteration=iteration + 1,
            skill=skill,
            baseline_accuracy=baseline_accuracy if not accepted else result.initial_accuracy,
            skill_accuracy=skill_accuracy,
            accepted=accepted,
            reports=reports,
        ))

    result.final_accuracy = baseline_accuracy
    _logger.info("=== Evolution complete ===")
    _logger.info("Initial accuracy: %.1f%%, Final accuracy: %.1f%%",
                 result.initial_accuracy * 100, result.final_accuracy * 100)
    _logger.info("Accepted skills: %d", len(result.accepted_skills))
    for s in result.accepted_skills:
        _logger.info("  - %s (%s)", s.name, s.pattern_category)

    return result
