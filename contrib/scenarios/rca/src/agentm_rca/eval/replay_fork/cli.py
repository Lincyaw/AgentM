"""``replay-fork`` CLI -- re-audit recorded baselines and fork on surface.

Example::

    uv run python -m agentm_rca.eval.replay_fork.cli run \\
        --source-exp agentm-ab100-baseline-0525-0847 \\
        --harness-model ark-glm51 \\
        --agent-model litellm \\
        --scenario rca:baseline \\
        --max-depth 3 \\
        --out runs/glm51-replay/results.jsonl

Both ``--harness-model`` and ``--agent-model`` are
``~/.agentm/config.toml`` profile names. The profile carries the
endpoint, api_key, and model id, so no ambient ``OPENAI_*`` env vars
are needed (or consulted).
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
        str | None,
        typer.Option("--source-exp", help="eval.db exp_id holding the recorded baselines"),
    ] = None,
    source_file: Annotated[
        list[Path] | None,
        typer.Option(
            "--source-file",
            help="OTLP/JSON session file(s) to replay (repeatable)",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="case data directory (required with --source-file)",
        ),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="Path to eval.db")] = Path("eval.db"),
    harness_model: Annotated[
        str, typer.Option("--harness-model", help="config.toml profile for extractor+auditor")
    ] = "ark-glm51",
    agent_model: Annotated[
        str, typer.Option("--agent-model", help="config.toml profile for the continuation agent")
    ] = "litellm",
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
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="skip cases already present in --out and append (restart-safe)",
        ),
    ] = False,
    skip_extractor: Annotated[
        bool,
        typer.Option(
            "--skip-extractor",
            help="bypass the extractor and feed raw trajectory directly to the auditor",
        ),
    ] = False,
    upper_bound: Annotated[
        bool,
        typer.Option(
            "--upper-bound",
            help=(
                "ceiling test: skip the auditor pipeline entirely, fork just "
                "before submission with a fixed reflection prompt"
            ),
        ),
    ] = False,
    auditor_prompt: Annotated[
        str | None,
        typer.Option(
            "--auditor-prompt",
            help="auditor prompt variant name (e.g. 'trajectory_coverage') or absolute path",
        ),
    ] = None,
) -> None:
    """Run the replay-fork experiment over a recorded baseline exp."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    import json as _json

    from agentm_rca.eval.agent import AgentMAgent

    from .case_source import EvalDbCaseSource, SessionFileCaseSource
    from .driver import JsonlResultSink, ReplayForkDriver
    from .judge import RcabenchJudge
    from .providers import build_profile_provider
    from .strategy import (
        _SUBMISSION_TOOL_NAMES,
        FixedInjectionStrategy,
        ForkStrategy,
        HarnessStrategy,
        UPPER_BOUND_REFLECTION,
        after_submission,
    )

    # -- Validate source flags --
    have_exp = source_exp is not None
    have_file = source_file is not None and len(source_file) > 0
    if have_exp == have_file:
        raise typer.BadParameter(
            "exactly one of --source-exp or --source-file must be provided"
        )
    if have_file and data_dir is None:
        raise typer.BadParameter("--data-dir is required when using --source-file")

    ids = [c.strip() for c in case_ids.split(",") if c.strip()] if case_ids else None

    # Resume: read case_ids already written to --out and skip them, appending
    # new results rather than truncating. Lets a killed run pick up where it
    # left off without re-running the cases it already finished.
    skip_ids: list[str] = []
    if resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                skip_ids.append(str(_json.loads(line)["case_id"]))
            except (ValueError, KeyError):
                continue
        typer.echo(f"# resume: skipping {len(skip_ids)} cases already in {out}")

    harness_provider = build_profile_provider(harness_model)
    agent_provider = build_profile_provider(agent_model)

    # -- Build trigger registry: cadence + on-submission --
    from llmharness.audit.triggers import TriggerRegistry
    from llmharness.extensions.trigger_cadence import _CadenceTrigger
    from llmharness.extensions.trigger_on_submission import _OnSubmissionTrigger

    trigger_registry = TriggerRegistry()
    trigger_registry.register_trigger(_CadenceTrigger(interval=5))
    trigger_registry.register_trigger(
        _OnSubmissionTrigger(tool_names=_SUBMISSION_TOOL_NAMES)
    )

    # -- Build the fork strategy from CLI flags --
    strategy: ForkStrategy
    if upper_bound:
        strategy = FixedInjectionStrategy(
            reminder=UPPER_BOUND_REFLECTION,
            turn_selector=after_submission,
        )
        typer.echo(f"# strategy: {strategy.label} (upper-bound ceiling test)")
    else:
        strategy = HarnessStrategy(
            harness_provider=harness_provider,
            max_depth=max_depth,
            sidecar_dir=sidecar_dir,
            skip_extractor=skip_extractor,
            trigger_registry=trigger_registry,
            auditor_prompt=auditor_prompt,
        )
        typer.echo(
            f"# strategy: {strategy.label}\n"
            f"# harness: {harness_provider[0]} model={harness_provider[1].get('model')} "
            f"base_url={harness_provider[1].get('base_url')}"
        )

    typer.echo(
        f"# agent:   {agent_provider[0]} model={agent_provider[1].get('model')} "
        f"base_url={agent_provider[1].get('base_url')} scenario={scenario}"
    )

    agent = AgentMAgent(
        scenario=scenario,
        provider_tuple=agent_provider,
        max_turns=max_turns,
    )
    source: EvalDbCaseSource | SessionFileCaseSource
    if have_file:
        assert source_file is not None
        assert data_dir is not None
        resolved_data_dir = str(data_dir.resolve())
        source = SessionFileCaseSource(
            file_paths=source_file,
            data_dir=resolved_data_dir,
        )
        typer.echo(f"# source:  {len(source_file)} session file(s), data_dir={resolved_data_dir}")
    else:
        assert source_exp is not None
        source = EvalDbCaseSource(
            db, source_exp, limit=limit, case_ids=ids, skip_case_ids=skip_ids or None
        )
    driver = ReplayForkDriver(
        agent=agent,
        strategy=strategy,
        judge=RcabenchJudge(),
        scenario=scenario,
    )
    sink = JsonlResultSink(out, append=resume)
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
