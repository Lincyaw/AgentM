"""CLI for the self-evolution loop.

Usage:
    uv run python -m agentm_rca.evolution.cli \
        --data-root datasets/ops-lite/cases \
        --train-split 20 \
        --scenario rca:baseline \
        --output contrib/scenarios/rca/skills/evolved/

Splits available cases into train/test sets, runs the evolution loop,
and reports results.
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to datasets/ops-lite/cases/ directory.",
    )
    parser.add_argument(
        "--output",
        default="contrib/scenarios/rca/skills/evolved/",
        help="Directory to write evolved skills into.",
    )
    parser.add_argument(
        "--scenario",
        default="rca:baseline",
        help="Scenario variant to use for eval runs.",
    )
    parser.add_argument(
        "--model",
        default="DeepSeek-V4-pro",
        help="LLM model for the RCA agent.",
    )
    parser.add_argument(
        "--base-url",
        default="http://100.114.89.62:8088/v1",
        help="LLM endpoint base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="sk-DLbuXPx8tzeb29atiigWEIXoU9P0xkh_2amQNqJHpMk",
        help="LLM API key.",
    )
    parser.add_argument(
        "--train-split",
        type=int,
        default=20,
        help="Number of cases for training (learning failures).",
    )
    parser.add_argument(
        "--test-split",
        type=int,
        default=10,
        help="Number of cases for testing (backtest validation).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum evolution iterations.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent case runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for case selection.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def _discover_cases(data_root: str) -> list[str]:
    """Find all valid case directories (those with causal_graph.json)."""
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

    # Discover available cases
    all_cases = _discover_cases(data_root)
    if not all_cases:
        _logger.error("No valid cases found in %s", data_root)
        return 1

    _logger.info("Found %d cases in %s", len(all_cases), data_root)

    # Split into train and test
    rng = random.Random(args.seed)
    rng.shuffle(all_cases)

    total_needed = args.train_split + args.test_split
    if total_needed > len(all_cases):
        _logger.warning(
            "Requested %d total cases but only %d available. Adjusting.",
            total_needed,
            len(all_cases),
        )
        train_count = min(args.train_split, len(all_cases) * 2 // 3)
        test_count = min(args.test_split, len(all_cases) - train_count)
    else:
        train_count = args.train_split
        test_count = args.test_split

    train_cases = all_cases[:train_count]
    test_cases = all_cases[train_count:train_count + test_count]

    _logger.info("Train cases: %d, Test cases: %d", len(train_cases), len(test_cases))

    # Run the loop
    result = await run_evolution_loop(
        train_cases=train_cases,
        test_cases=test_cases,
        data_root=data_root,
        skill_output_dir=output_dir,
        scenario=args.scenario,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_iterations=args.max_iterations,
        concurrency=args.concurrency,
    )

    # Report
    print("\n" + "=" * 60)
    print("EVOLUTION RESULTS")
    print("=" * 60)
    print(f"Initial accuracy: {result.initial_accuracy:.1%}")
    print(f"Final accuracy:   {result.final_accuracy:.1%}")
    print(f"Iterations run:   {len(result.iterations)}")
    print(f"Skills accepted:  {len(result.accepted_skills)}")
    print()

    for it in result.iterations:
        status = "ACCEPTED" if it.accepted else "REJECTED"
        skill_name = it.skill.name if it.skill else "(none)"
        print(
            f"  Iteration {it.iteration}: {status} "
            f"skill={skill_name} "
            f"accuracy={it.skill_accuracy:.1%}"
        )

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
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    exit_code = asyncio.run(_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
