"""Self-evolution CLI.

Usage::

    uv run python -m rca_evolution.cli run \\
        --eval-config contrib/evals/src/agentm_eval/benchmarks/rca/configs/ops-lite-fixed-50.yaml \\
        --model litellm-dsv4flash-nothink \\
        --train-limit 20 --test-limit 10

Uses ``rca llm-eval run`` for case execution and ``eval.db`` for results.
Model is a ``$AGENTM_HOME/config.toml`` profile name (default
``~/.agentm/config.toml``).
"""

from __future__ import annotations

import asyncio
import logging as _stdlib_logging
import os
import sys
from pathlib import Path
from types import FrameType
from typing import Annotated

from loguru import logger

import typer

app = typer.Typer(
    name="evolution",
    help="Self-evolving skill loop for RCA.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command()
def run(
    eval_config: Annotated[
        Path, typer.Option("--eval-config", help="Eval config YAML (e.g. config.ops-lite-fixed-50.yaml)")
    ],
    model: Annotated[
        str, typer.Option(help="$AGENTM_HOME/config.toml profile name")
    ] = "litellm-dsv4flash-nothink",
    scenario: Annotated[
        str, typer.Option(help="Scenario variant for eval runs")
    ] = "rca:baseline",
    data_root: Annotated[
        str, typer.Option(help="Path to dataset cases directory")
    ] = "datasets/ops-lite/cases",
    db: Annotated[
        Path, typer.Option(help="Path to eval.db")
    ] = Path("eval.db"),
    output: Annotated[
        Path, typer.Option(help="Directory to write evolved skills")
    ] = Path("contrib/scenarios/rca/skills/evolved"),
    exp_prefix: Annotated[
        str, typer.Option(help="Experiment ID prefix in eval.db")
    ] = "evolution",
    concurrency: Annotated[int, typer.Option()] = 5,
    train_limit: Annotated[int, typer.Option(help="Max train cases")] = 20,
    test_limit: Annotated[int, typer.Option(help="Max test cases")] = 10,
    max_iterations: Annotated[int, typer.Option()] = 3,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the self-evolution loop."""
    class _InterceptHandler(_stdlib_logging.Handler):
        def emit(self, record: _stdlib_logging.LogRecord) -> None:
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame: FrameType | None = _stdlib_logging.currentframe()
            depth = 2
            while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stdout, level=log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{file}:{line}</cyan> <level>{message}</level>")
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    _stdlib_logging.getLogger("httpx").setLevel(_stdlib_logging.WARNING)
    _stdlib_logging.getLogger("httpcore").setLevel(_stdlib_logging.WARNING)
    _stdlib_logging.getLogger("openai").setLevel(_stdlib_logging.WARNING)

    from rca_evolution.loop import run_evolution_loop

    result = asyncio.run(run_evolution_loop(
        eval_config=str(eval_config.resolve()),
        db_path=str(db.resolve()),
        data_root=os.path.abspath(data_root),
        skill_output_dir=str(output.resolve()),
        scenario=scenario,
        model_profile=model,
        exp_id_prefix=exp_prefix,
        concurrency=concurrency,
        train_limit=train_limit,
        test_limit=test_limit,
        max_iterations=max_iterations,
    ))

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("EVOLUTION RESULTS")
    typer.echo("=" * 60)
    typer.echo(f"Initial accuracy: {result.initial_accuracy:.1%}")
    typer.echo(f"Final accuracy:   {result.final_accuracy:.1%}")
    typer.echo(f"Iterations:       {len(result.iterations)}")
    typer.echo(f"Skills accepted:  {len(result.accepted_skills)}")

    for it in result.iterations:
        status = "ACCEPTED" if it.accepted else "REJECTED"
        name = it.skill.name if it.skill else "(none)"
        typer.echo(f"  Iter {it.iteration}: {status} {name} acc={it.skill_accuracy:.1%}")

    if result.accepted_skills:
        typer.echo(f"\nSkills written to: {output}")
        for s in result.accepted_skills:
            typer.echo(f"  - {s.name}/SKILL.md")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
