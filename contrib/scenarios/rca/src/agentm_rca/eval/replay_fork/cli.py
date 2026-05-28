"""``replay-fork`` CLI -- re-audit recorded baselines and fork on surface.

Example::

    # harness=glm5.1 on Ark, agent=Doubao via .env / OPENAI_* env
    set -a; . ./.env; set +a
    uv run python -m agentm_rca.eval.replay_fork.cli run \\
        --source-exp agentm-ab100-baseline-0525-0847 \\
        --harness-model ark-glm51 \\
        --agent-model Doubao-Seed-2.0-pro \\
        --scenario rca:baseline \\
        --max-depth 3 \\
        --out runs/glm51-replay/results.jsonl

The harness model is a ``~/.agentm/config.toml`` profile (its endpoint /
key travel in the profile). The agent model is an id resolved against the
ambient ``OPENAI_*`` env the way the rca eval driver builds its provider,
so the continuation hits the same endpoint the recorded baseline did.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="replay-fork",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _root() -> None:
    """Replay-fork experiment commands (keeps ``run`` an explicit subcommand)."""


@app.command()
def run(
    source_exp: Annotated[
        str, typer.Option("--source-exp", help="eval.db exp_id holding the recorded baselines")
    ],
    db: Annotated[Path, typer.Option("--db", help="Path to eval.db")] = Path("eval.db"),
    harness_model: Annotated[
        str, typer.Option("--harness-model", help="config.toml profile for extractor+auditor")
    ] = "ark-glm51",
    agent_model: Annotated[
        str, typer.Option("--agent-model", help="model id for the continuation agent")
    ] = "Doubao-Seed-2.0-pro",
    agent_provider: Annotated[
        str, typer.Option("--agent-provider", help="provider id for the continuation agent")
    ] = "openai",
    scenario: Annotated[
        str, typer.Option("--scenario", help="scenario for the continuation agent")
    ] = "rca:baseline",
    max_depth: Annotated[
        int, typer.Option("--max-depth", help="max stacked interventions (greedy spine depth)")
    ] = 3,
    max_turns: Annotated[
        int, typer.Option("--max-turns", help="max turns per continuation rollout")
    ] = 60,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", help="cases to process in parallel")
    ] = 8,
    limit: Annotated[
        int | None, typer.Option("--limit", help="only the first N cases")
    ] = None,
    case_ids: Annotated[
        str | None, typer.Option("--case-ids", help="comma-separated case ids to restrict to")
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="results JSONL path")
    ] = Path("runs/replay-fork/results.jsonl"),
    sidecar_dir: Annotated[
        Path | None,
        typer.Option("--sidecar-dir", help="dir for per-case fork-tree replay sidecars"),
    ] = None,
) -> None:
    """Run the replay-fork experiment over a recorded baseline exp."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from agentm_rca.eval.agent import AgentMAgent

    from .case_source import EvalDbCaseSource
    from .driver import JsonlResultSink, ReplayForkDriver
    from .judge import RcabenchJudge
    from .providers import build_profile_provider

    ids = [c.strip() for c in case_ids.split(",") if c.strip()] if case_ids else None

    harness_provider = build_profile_provider(harness_model)
    typer.echo(
        f"# harness: {harness_provider[0]} model={harness_provider[1].get('model')} "
        f"base_url={harness_provider[1].get('base_url')}"
    )
    typer.echo(f"# agent:   provider={agent_provider} model={agent_model} scenario={scenario}")

    agent = AgentMAgent(
        scenario=scenario,
        model=agent_model,
        provider=agent_provider,
        max_turns=max_turns,
    )
    source = EvalDbCaseSource(db, source_exp, limit=limit, case_ids=ids)
    driver = ReplayForkDriver(
        agent=agent,
        harness_provider=harness_provider,
        judge=RcabenchJudge(),
        scenario=scenario,
        max_depth=max_depth,
        sidecar_dir=sidecar_dir,
    )
    sink = JsonlResultSink(out)
    try:
        summary = asyncio.run(driver.run(source, sink, max_concurrency=max_concurrency))
    finally:
        sink.close()

    typer.echo("\n=== replay-fork summary ===")
    typer.echo(summary.format())
    typer.echo(f"# results: {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
