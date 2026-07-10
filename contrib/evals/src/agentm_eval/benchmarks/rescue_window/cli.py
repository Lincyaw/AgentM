"""CLI for the measurement-first rescue-window harness (DESIGN §3, §5)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from .analysis import (
    aggregate,
    build_report,
    estimate_windows,
    ladder_gaps,
    write_report,
)
from .harness import (
    PRESETS,
    PrefixSampler,
    RolloutConfig,
    SamplingPolicy,
    StrongModelOracle,
    TreatmentFactory,
    build_profile_provider,
    default_store,
    load_adapter,
    load_corpus,
    load_corpus_from_eval_db,
    load_trajectory_messages,
    run_landscape,
)
from .model import ContentLevel, EvalUnit, EvalUnitStore

app = typer.Typer(
    name="rescue-window",
    help=__doc__,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main() -> None:
    """Load AgentM's shared environment before any command runs."""

    from agentm.cli import autoload_dotenv

    autoload_dotenv(Path.cwd())


def _resolve_session_store(obs_dir: Path | None):  # type: ignore[no-untyped-def]
    if obs_dir is not None:
        from agentm.core.runtime.session_manager import JsonlSessionStore

        return JsonlSessionStore(session_dir=obs_dir.resolve())
    return default_store()


def _make_sampler(
    progress: str, random_n: int, dense: bool, min_turn: int, final_tool: str | None
) -> PrefixSampler:
    fractions = tuple(float(x) for x in progress.split(",") if x.strip()) if progress else ()
    return PrefixSampler(
        SamplingPolicy(
            relative_progress=fractions,
            n_random=random_n,
            dense=dense,
            min_turn=min_turn,
        ),
        final_tool=final_tool,
    )


def _resolve_corpus(
    corpus: Path | None,
    exp_id: str | None,
    db: str,
    data_root: Path | None,
) -> list:
    """Resolve corpus from either a manifest file or eval.db exp_id."""
    if corpus is not None:
        return load_corpus(corpus, data_root=data_root)
    if exp_id is not None:
        refs = load_corpus_from_eval_db(exp_id, db_path=db, data_root=data_root)
        if not refs:
            raise typer.BadParameter(f"no rollout sessions found for exp_id={exp_id!r} in {db}")
        typer.echo(f"Loaded {len(refs)} trajectories from eval.db exp_id={exp_id!r}")
        return refs
    raise typer.BadParameter("provide --corpus or --exp-id")


@app.command()
def sample(
    corpus: Annotated[Path, typer.Option("--corpus", help="Corpus manifest (YAML/JSON/CSV).")],
    data_root: Annotated[Path | None, typer.Option("--data-root")] = None,
    adapter: Annotated[str, typer.Option("--adapter", help="Scenario adapter name.")] = "rca",
    progress: Annotated[str, typer.Option("--progress")] = "0.2,0.4,0.6,0.8",
    random_n: Annotated[int, typer.Option("--random")] = 0,
    dense: Annotated[bool, typer.Option("--dense/--no-dense")] = False,
    min_turn: Annotated[int, typer.Option("--min-turn", min=1)] = 1,
    obs_dir: Annotated[Path | None, typer.Option("--obs-dir")] = None,
) -> None:
    """Dry-run: print the prefixes the sampler would draw."""

    refs = load_corpus(corpus, data_root=data_root)
    session_store = _resolve_session_store(obs_dir)
    scenario = load_adapter(adapter)
    sampler = _make_sampler(progress, random_n, dense, min_turn, scenario.final_tool)
    rows = []
    for ref in refs:
        messages = load_trajectory_messages(ref, store=session_store)
        for point in sampler.sample(ref, messages):
            rows.append(
                {
                    "prefix_id": point.prefix_id,
                    "case_id": point.case_id,
                    "turn_index": point.turn_index,
                    "progress": point.progress,
                    "stratum": point.stratum,
                }
            )
    typer.echo(json.dumps({"prefixes": len(rows), "rows": rows}, ensure_ascii=False, indent=2))


@app.command()
def run(
    corpus: Annotated[Path | None, typer.Option("--corpus", help="Corpus manifest.")] = None,
    exp_id: Annotated[str | None, typer.Option("--exp-id", help="Load corpus from eval.db by experiment ID.")] = None,
    db: Annotated[str, typer.Option("--db", help="Path to eval.db.")] = "eval.db",
    out: Annotated[Path, typer.Option("--out", help="EvalUnit store JSONL path.")] = Path("runs/rescue-window/latest.jsonl"),
    data_root: Annotated[Path | None, typer.Option("--data-root")] = None,
    adapter: Annotated[str, typer.Option("--adapter", help="Scenario adapter name.")] = "rca",
    preset: Annotated[
        str,
        typer.Option("--preset", help="Condition preset: oracle-landscape | content-ladder."),
    ] = "oracle-landscape",
    oracle_model: Annotated[
        str | None,
        typer.Option("--oracle-model", help="config.toml profile for ORACLE_GROUNDED."),
    ] = None,
    actor_model: Annotated[
        str | None,
        typer.Option(
            "--actor-model",
            help="config.toml profile overriding the baseline's stored provider "
            "(use when the recorded endpoint is unreachable).",
        ),
    ] = None,
    k: Annotated[int, typer.Option("--k", min=1, help="Rollouts per intervention cell.")] = 3,
    progress: Annotated[str, typer.Option("--progress")] = "0.2,0.4,0.6,0.8",
    random_n: Annotated[int, typer.Option("--random")] = 0,
    dense: Annotated[bool, typer.Option("--dense/--no-dense")] = False,
    min_turn: Annotated[int, typer.Option("--min-turn", min=1)] = 1,
    max_turns: Annotated[int, typer.Option("--max-turns", min=1)] = 60,
    concurrency: Annotated[int, typer.Option("--concurrency", "-j", min=1)] = 1,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    obs_dir: Annotated[Path | None, typer.Option("--obs-dir")] = None,
) -> None:
    """Run the oracle landscape (E1): fork prefixes, roll out the content ladder."""

    refs = _resolve_corpus(corpus, exp_id, db, data_root)
    if limit is not None:
        refs = refs[:limit]
    scenario = load_adapter(adapter)
    conditions = PRESETS.get(preset)
    if conditions is None:
        raise typer.BadParameter(f"unknown preset {preset!r}; choices: {sorted(PRESETS)}")
    oracle = None
    if any(level is ContentLevel.ORACLE_GROUNDED for level, _ in conditions):
        if not oracle_model:
            raise typer.BadParameter("--oracle-model is required for a preset with ORACLE_GROUNDED")
        oracle = StrongModelOracle(provider=build_profile_provider(oracle_model))
    factory = TreatmentFactory(conditions=conditions, oracle=oracle)
    sampler = _make_sampler(progress, random_n, dense, min_turn, scenario.final_tool)
    provider_override = build_profile_provider(actor_model) if actor_model else None
    store = EvalUnitStore(out)
    session_store = _resolve_session_store(obs_dir)

    def _on(unit: EvalUnit) -> None:
        typer.echo(
            f"  {unit.status.upper()} {unit.prefix_id} {unit.treatment_id} "
            f"seed={unit.branch_seed} score={unit.normalized_score}"
        )

    written = asyncio.run(
        run_landscape(
            refs,
            sampler=sampler,
            factory=factory,
            store=store,
            session_store=session_store,
            adapter=scenario,
            config=RolloutConfig(max_turns=max_turns),
            k=k,
            concurrency=concurrency,
            provider_override=provider_override,
            on_result=_on,
        )
    )
    typer.echo(json.dumps({"rows_written": written, "store": str(out)}, indent=2))


@app.command()
def aggregate_cmd(
    store: Annotated[Path, typer.Argument(help="EvalUnit store JSONL.")],
    out_prefix: Annotated[Path, typer.Option("--out-prefix")] = Path("runs/rescue-window/report"),
    epsilon: Annotated[float, typer.Option("--epsilon")] = 0.0,
    gamma: Annotated[float, typer.Option("--gamma")] = 0.0,
) -> None:
    """Aggregate a store into metrics + Rescue Window + report files."""

    rows = EvalUnitStore(store).read_all()
    if not rows:
        raise typer.BadParameter(f"no rows in {store}")
    result = aggregate(rows, epsilon=epsilon)
    windows = estimate_windows(result, gamma=gamma)
    ladders = ladder_gaps(result)
    report = build_report(rows, result, windows, ladders=ladders)
    paths = write_report(report, out_prefix)
    typer.echo(json.dumps({**report["summary"], "out": paths}, ensure_ascii=False, indent=2))


# typer uses the function name as the command name; expose as "aggregate".
app.command("aggregate")(aggregate_cmd)


if __name__ == "__main__":
    app()
