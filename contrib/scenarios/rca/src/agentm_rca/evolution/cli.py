"""CLI for the self-evolution loop.

Usage:
    uv run python -m agentm_rca.evolution.cli \
        --data-root datasets/ops-lite/cases \
        --model litellm-dsv4flash-nothink \
        --train-split 20 \
        --output contrib/scenarios/rca/skills/evolved/

Uses ~/.agentm/config.toml model profiles — no manual URL/key needed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys

_logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-evolving skill loop for RCA.",
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Path to datasets/ops-lite/cases/ directory.",
    )
    parser.add_argument(
        "--output", default="contrib/scenarios/rca/skills/evolved/",
        help="Directory to write evolved skills into.",
    )
    parser.add_argument(
        "--scenario", default="rca:baseline",
        help="Scenario variant to use for eval runs.",
    )
    parser.add_argument(
        "--model", default="litellm-dsv4flash-nothink",
        help="~/.agentm/config.toml profile name.",
    )
    parser.add_argument(
        "--train-split", type=int, default=20,
        help="Number of cases for training.",
    )
    parser.add_argument(
        "--test-split", type=int, default=10,
        help="Number of cases for testing.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _discover_cases(data_root: str) -> list[str]:
    cases: list[str] = []
    if not os.path.isdir(data_root):
        _logger.error("Data root does not exist: %s", data_root)
        return cases
    for entry in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, entry)
        if os.path.isdir(case_dir) and os.path.exists(
            os.path.join(case_dir, "causal_graph.json")
        ):
            cases.append(entry)
    return cases


async def _main(args: argparse.Namespace) -> int:
    from agentm_rca.evolution.loop import run_evolution_loop

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output)

    all_cases = _discover_cases(data_root)
    if not all_cases:
        _logger.error("No valid cases found in %s", data_root)
        return 1

    _logger.info("Found %d cases in %s", len(all_cases), data_root)

    rng = random.Random(args.seed)
    rng.shuffle(all_cases)

    total_needed = args.train_split + args.test_split
    if total_needed > len(all_cases):
        train_count = min(args.train_split, len(all_cases) * 2 // 3)
        test_count = min(args.test_split, len(all_cases) - train_count)
    else:
        train_count = args.train_split
        test_count = args.test_split

    train_cases = all_cases[:train_count]
    test_cases = all_cases[train_count:train_count + test_count]

    _logger.info("Train cases: %d, Test cases: %d", len(train_cases), len(test_cases))

    result = await run_evolution_loop(
        train_cases=train_cases,
        test_cases=test_cases,
        data_root=data_root,
        skill_output_dir=output_dir,
        scenario=args.scenario,
        model_profile=args.model,
        max_iterations=args.max_iterations,
        concurrency=args.concurrency,
    )

    print("\n" + "=" * 60)
    print("EVOLUTION RESULTS")
    print("=" * 60)
    print(f"Initial accuracy: {result.initial_accuracy:.1%}")
    print(f"Final accuracy:   {result.final_accuracy:.1%}")
    print(f"Iterations run:   {len(result.iterations)}")
    print(f"Skills accepted:  {len(result.accepted_skills)}")

    for it in result.iterations:
        status = "ACCEPTED" if it.accepted else "REJECTED"
        skill_name = it.skill.name if it.skill else "(none)"
        print(f"  Iteration {it.iteration}: {status} skill={skill_name} accuracy={it.skill_accuracy:.1%}")

    if result.accepted_skills:
        print(f"\nAccepted skills written to: {output_dir}")
        for s in result.accepted_skills:
            print(f"  - {s.name}/SKILL.md")

    return 0


def main() -> None:
    args = _parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    exit_code = asyncio.run(_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
