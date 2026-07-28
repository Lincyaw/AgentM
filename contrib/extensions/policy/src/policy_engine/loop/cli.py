"""``python -m policy_engine loop <stage>``: six stages, each a file in and a
file out.

Every stage reads an artifact and writes the next, so one can be re-run alone
against its predecessor's output or against a file edited by hand. ``loop run``
chains the same functions and lands the same artifacts; there is no second code
path to keep in step.

``--bench`` picks the adapter. Everything benchmark-specific -- where attempts
live, how to grade a re-run, what a metric is called -- is behind it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import typer

from policy_engine.shared.pg_query import PgQuerySource

from .abstract import abstract
from .align import align, tally
from .benches import senior_swe
from .collect import collect
from .compile_gate import check_precondition, compile_candidates
from .contracts import CompiledCandidate, write_artifact
from .diagnose import diagnose
from .notes import apply_notes, notes
from .pipeline import (
    DEFAULT_VOCABULARY,
    LoopConfig,
    LoopPaths,
    load_candidates,
    load_cases,
    load_compiled,
    load_diagnoses,
    load_measurements,
    load_notes,
    run,
    summarise,
)
from .protocols import CaseSource, ReplayBackend
from .replay import replay
from .select import select

app = typer.Typer(
    name="loop",
    add_completion=False,
    no_args_is_help=True,
    help="The evolution loop: collect, diagnose, abstract, compile, replay, select.",
)

_BENCH = typer.Option(senior_swe.NAME, "--bench")
_DSN = typer.Option("", "--dsn", envvar="AGENTM_TRAJECTORY_DSN")
_SCHEMA = typer.Option("", "--schema", envvar="AGENTM_TRAJECTORY_SCHEMA")
_PROVIDER = typer.Option("", "--provider", help="model profile for loop agents")
_USER_CONFIG = typer.Option("", "--user-config", help="config.toml for the profile")
_ONLY = typer.Option("", "--only", help="substring of the task name")
_CONCURRENCY = typer.Option(4, "--concurrency")
_BENCH_ROOT = typer.Option(str(senior_swe.DEFAULT_BENCH_ROOT), "--bench-root")
_GATEWAY = typer.Option("", "--gateway-url", envvar="ARL_GATEWAY_URL")
_REGISTRY = typer.Option("", "--image-registry")
_IMAGE_TAG = typer.Option("v2026.06", "--image-tag")
_JUDGE_MODEL = typer.Option("openai/DeepSeek-V4-pro", "--judge-model")
_JUDGE_PROFILE = typer.Option("litellm-dsv4pro", "--judge-profile")
_MODEL_NAME = typer.Option("azure-gpt", "--model-name")
_OUT = typer.Option(..., "-o", "--out")


def _source(
    bench: str,
    *,
    bench_root: str,
    dsn: str,
    schema: str,
    gateway_url: str,
    image_registry: str = "",
    image_tag: str = "",
) -> CaseSource:
    if bench != senior_swe.NAME:
        raise typer.BadParameter(f"unknown bench {bench!r}")
    return senior_swe.SeniorSweSource(
        bench_root=Path(bench_root),
        trajectory_dsn=dsn,
        trajectory_schema=schema,
        gateway_url=gateway_url,
        image_registry=image_registry,
        image_tag=image_tag,
    )


def _backend(
    bench: str,
    *,
    gateway_url: str,
    image_registry: str,
    image_tag: str,
    dsn: str,
    schema: str,
    model_name: str,
    judge_model: str,
    judge_profile: str,
    user_config: str,
    work_dir: str,
) -> ReplayBackend:
    if bench != senior_swe.NAME:
        raise typer.BadParameter(f"unknown bench {bench!r}")
    key, base_url = senior_swe.judge_credentials(judge_profile, user_config)
    return senior_swe.HarborReplay(
        gateway_url=gateway_url,
        image_registry=image_registry,
        image_tag=image_tag,
        agentm_home=senior_swe.default_agentm_home(),
        trajectory_dsn=dsn,
        trajectory_schema=schema,
        model_name=model_name,
        judge_model=judge_model,
        judge_api_key=key,
        judge_base_url=base_url,
        work_dir=work_dir,
    )


@app.command("collect")
def cmd_collect(
    batch: str = typer.Option(..., "--batch"),
    out: str = _OUT,
    bench: str = _BENCH,
    bench_root: str = _BENCH_ROOT,
    dsn: str = _DSN,
    schema: str = _SCHEMA,
    only: str = _ONLY,
) -> None:
    """Graded failures in a batch become cases."""
    source = _source(
        bench, bench_root=bench_root, dsn=dsn, schema=schema, gateway_url=""
    )
    cases = collect(source, batch, only=only)
    write_artifact(Path(out), cases)
    typer.echo(f"{len(cases)} case(s) -> {out}")


@app.command("diagnose")
def cmd_diagnose(
    cases: str = typer.Option(..., "--cases"),
    out: str = _OUT,
    bench: str = _BENCH,
    bench_root: str = _BENCH_ROOT,
    dsn: str = _DSN,
    schema: str = _SCHEMA,
    gateway_url: str = _GATEWAY,
    image_registry: str = _REGISTRY,
    image_tag: str = _IMAGE_TAG,
    provider: str = _PROVIDER,
    user_config: str = _USER_CONFIG,
    concurrency: int = _CONCURRENCY,
    only: str = _ONLY,
) -> None:
    """Why each case's work diverged. One agent per case, in parallel.

    Each agent works on a machine with the repository on it: the attempt's own,
    forked, while that still exists, and a fresh one from the task's image after
    it expires. ``--gateway-url`` is what makes either possible and
    ``--image-registry`` the second; without them the agents see the artefacts
    and not the code, which is enough for some causes and not others.
    """
    source = _source(
        bench,
        bench_root=bench_root,
        dsn=dsn,
        schema=schema,
        gateway_url=gateway_url,
        image_registry=image_registry,
        image_tag=image_tag,
    )
    loaded = [c for c in load_cases(Path(cases)) if not only or only in c.task_name]
    found = asyncio.run(
        diagnose(
            loaded,
            provider=provider,
            user_config=user_config,
            concurrency=concurrency,
            source=source,
        )
    )
    write_artifact(Path(out), found)
    for item in found:
        typer.echo(
            f"  {item.task_name[:34]:36s} {item.cause_class:24s} "
            f"reachable={item.reachable}"
        )
    typer.echo(f"{len(found)}/{len(loaded)} diagnosed -> {out}")


@app.command("abstract")
def cmd_abstract(
    diagnoses: str = typer.Option(..., "--diagnoses"),
    out: str = _OUT,
    provider: str = _PROVIDER,
    user_config: str = _USER_CONFIG,
    concurrency: int = _CONCURRENCY,
    only: str = _ONLY,
) -> None:
    """Diagnoses become general checks. Reflective checks are dropped here."""
    loaded = [
        d for d in load_diagnoses(Path(diagnoses)) if not only or only in d.task_name
    ]
    found = asyncio.run(
        abstract(
            loaded,
            provider=provider,
            user_config=user_config,
            concurrency=concurrency,
        )
    )
    write_artifact(Path(out), found)
    for item in found:
        typer.echo(f"  {item.candidate_id}  [{item.checkpoint}] {item.check[:90]}")
    typer.echo(f"{len(found)} candidate(s) -> {out}")


@app.command("notes")
def cmd_notes(
    diagnoses: str = typer.Option(..., "--diagnoses"),
    out: str = _OUT,
    existing: str = typer.Option("", "--existing", help="notes to merge into"),
    apply_to: str = typer.Option(
        "",
        "--apply-to",
        help="bench task root; writes review-notes.md beside each task",
    ),
    provider: str = _PROVIDER,
    user_config: str = _USER_CONFIG,
    concurrency: int = _CONCURRENCY,
) -> None:
    """Diagnoses become notes for whoever reviews this repository next.

    A different artefact from a candidate, for a different reader: a candidate
    goes to the agent doing the work, a note to the one reviewing it, and a note
    holds for the repository rather than for a moment.
    """
    loaded = load_diagnoses(Path(diagnoses))
    held = load_notes(Path(existing)) if existing else []
    produced = asyncio.run(
        notes(
            loaded,
            held,
            provider=provider,
            user_config=user_config,
            concurrency=concurrency,
        )
    )
    write_artifact(Path(out), produced)
    for item in produced:
        typer.echo(f"  [{item.repository}] {item.situation[:88]}")
    typer.echo(f"{len(produced)} note(s) -> {out}")
    if apply_to:
        written = apply_notes(produced, Path(apply_to))
        for path in written:
            typer.echo(f"  applied -> {path}")


@app.command("align")
def cmd_align(
    measurements: str = typer.Option(..., "--measurements"),
    cases: str = typer.Option(..., "--cases"),
    out: str = _OUT,
    provider: str = _PROVIDER,
    user_config: str = _USER_CONFIG,
    concurrency: int = _CONCURRENCY,
) -> None:
    """Did each run's review find what grading punished?

    The signal the score cannot give. A score moves only when the review found
    the right defect and the agent then fixed it; this asks the first half on
    its own, and it has an answer every time.
    """
    by_case = {case.case_id: case for case in load_cases(Path(cases))}
    pairs = []
    for measurement in load_measurements(Path(measurements)):
        case = by_case.get(measurement.case_id)
        if case is None:
            typer.echo(f"  no case for {measurement.case_id}; skipping")
            continue
        pairs.append((case, measurement.review_report, measurement.candidate_id))

    verdicts = asyncio.run(
        align(
            pairs, provider=provider, user_config=user_config, concurrency=concurrency
        )
    )
    write_artifact(Path(out), verdicts)
    for item in verdicts:
        typer.echo(f"  {item.verdict:<10} {item.case_id[:44]}  {item.finding[:60]}")
    counts = tally(verdicts)
    typer.echo(" ".join(f"{name}={count}" for name, count in counts.items()))
    typer.echo(f"{len(verdicts)} alignment(s) -> {out}")


@app.command("compile")
def cmd_compile(
    candidates: str = typer.Option(..., "--candidates"),
    out: str = _OUT,
    vocab: str = typer.Option("", "--vocab"),
    provider: str = _PROVIDER,
    user_config: str = _USER_CONFIG,
    concurrency: int = _CONCURRENCY,
) -> None:
    """Plain-language conditions become a tag trigger plus a fact precondition."""
    compiled = asyncio.run(
        compile_candidates(
            load_candidates(Path(candidates)),
            vocabulary_path=Path(vocab) if vocab else DEFAULT_VOCABULARY,
            provider=provider,
            user_config=user_config,
            concurrency=concurrency,
        )
    )
    write_artifact(Path(out), compiled)
    for item in compiled:
        typer.echo(f"  {item.item_id}  trigger={item.trigger}")
        if item.precondition:
            typer.echo(f"      precondition: {item.precondition[:120]}")
    typer.echo(f"{len(compiled)} gate(s) -> {out}")


@app.command("check-gates")
def cmd_check_gates(
    compiled: str = typer.Option(..., "--compiled"),
    sessions: str = typer.Option(
        ..., "--sessions", help="comma-separated trajectory session ids"
    ),
    dsn: str = _DSN,
    schema: str = _SCHEMA,
) -> None:
    """Run each precondition against recorded sessions.

    Worth doing before spending a replay: a precondition true nowhere makes the
    item inert, one true everywhere is not gating anything, and a word-boundary
    pattern for "test" matching neither ``pytest`` nor ``--group testing`` is
    the kind of mistake this catches in a second.
    """
    ids = [s.strip() for s in sessions.split(",") if s.strip()]
    source = PgQuerySource(dsn, "")
    try:
        _report_gates(load_compiled(Path(compiled)), ids, schema=schema, source=source)
    finally:
        source.close()


def _report_gates(
    items: Sequence[CompiledCandidate],
    session_ids: Sequence[str],
    *,
    schema: str,
    source: PgQuerySource,
) -> None:
    for item in items:
        outcome, error = check_precondition(
            item.precondition, schema=schema, session_ids=session_ids, source=source
        )
        held = sum(1 for v in outcome.values() if v)
        if error:
            note = f"  ERROR {error}"
        elif not item.precondition.strip():
            note = "  (unconditional)"
        elif held == 0:
            note = "  INERT: holds for no recorded session"
        elif held == len(outcome):
            note = "  NOT GATING: holds for every recorded session"
        else:
            note = ""
        typer.echo(f"  {item.item_id}: {held}/{len(outcome)}{note}")


@app.command("replay")
def cmd_replay(
    compiled: str = typer.Option(..., "--compiled"),
    cases: str = typer.Option(..., "--cases"),
    out: str = _OUT,
    bench: str = _BENCH,
    gateway_url: str = _GATEWAY,
    image_registry: str = _REGISTRY,
    image_tag: str = _IMAGE_TAG,
    dsn: str = _DSN,
    schema: str = _SCHEMA,
    model_name: str = _MODEL_NAME,
    judge_model: str = _JUDGE_MODEL,
    judge_profile: str = _JUDGE_PROFILE,
    user_config: str = _USER_CONFIG,
    work_dir: str = typer.Option(".policy-loop-work", "--work-dir"),
    arms: str = typer.Option("placebo,candidate", "--arms"),
) -> None:
    """The counterfactual: same attempt, same decision point, with and without."""
    backend = _backend(
        bench,
        gateway_url=gateway_url,
        image_registry=image_registry,
        image_tag=image_tag,
        dsn=dsn,
        schema=schema,
        model_name=model_name,
        judge_model=judge_model,
        judge_profile=judge_profile,
        user_config=user_config,
        work_dir=work_dir,
    )
    measurements = replay(
        backend,
        load_compiled(Path(compiled)),
        load_cases(Path(cases)),
        arms=tuple(a.strip() for a in arms.split(",") if a.strip()),
        out_path=Path(out),
    )
    for m in measurements:
        suffix = f" ({m.lost_reason})" if m.lost_reason else ""
        typer.echo(f"  [{m.arm:9s}] {m.case_id[:44]:46s} {m.outcome}{suffix}")
    typer.echo(f"{len(measurements)} measurement(s) -> {out}")


@app.command("select")
def cmd_select(
    measurements: str = typer.Option(..., "--measurements"),
    out: str = _OUT,
) -> None:
    """Which candidates survived. Losses are no evidence, never a zero."""
    verdicts = select(load_measurements(Path(measurements)))
    write_artifact(Path(out), verdicts)
    typer.echo(summarise(verdicts))


@app.command("run")
def cmd_run(
    batch: str = typer.Option(..., "--batch"),
    bench: str = _BENCH,
    bench_root: str = _BENCH_ROOT,
    gateway_url: str = _GATEWAY,
    image_registry: str = _REGISTRY,
    image_tag: str = _IMAGE_TAG,
    root: str = typer.Option(".policy-loop", "--root"),
    run_id: str = typer.Option("", "--run-id"),
    dsn: str = _DSN,
    schema: str = _SCHEMA,
    provider: str = _PROVIDER,
    user_config: str = _USER_CONFIG,
    model_name: str = _MODEL_NAME,
    judge_model: str = _JUDGE_MODEL,
    judge_profile: str = _JUDGE_PROFILE,
    concurrency: int = _CONCURRENCY,
    only: str = _ONLY,
) -> None:
    """All six stages in order, writing every intermediate.

    Worth starting inside the batch window: replay does not need it, but
    diagnose can fork the attempt's own machine while it lives, and only
    reconstruct from the image after.
    """
    paths = LoopPaths.for_run(Path(root), run_id)
    verdicts = asyncio.run(
        run(
            LoopConfig(
                batch=batch,
                paths=paths,
                source=_source(
                    bench,
                    bench_root=bench_root,
                    dsn=dsn,
                    schema=schema,
                    gateway_url=gateway_url,
                    # Passed here as `loop diagnose` passes them: without the
                    # image, diagnosis loses the repository the moment the
                    # attempt's own sandbox expires, and a chained run would
                    # then see less than the same stage run by hand.
                    image_registry=image_registry,
                    image_tag=image_tag,
                ),
                backend=_backend(
                    bench,
                    gateway_url=gateway_url,
                    image_registry=image_registry,
                    image_tag=image_tag,
                    dsn=dsn,
                    schema=schema,
                    model_name=model_name,
                    judge_model=judge_model,
                    judge_profile=judge_profile,
                    user_config=user_config,
                    work_dir=str(paths.root / "work"),
                ),
                provider=provider,
                user_config=user_config,
                concurrency=concurrency,
                only=only,
            )
        )
    )
    typer.echo(summarise(verdicts))
    typer.echo(f"artifacts in {paths.root}")


@app.command("show")
def cmd_show(artifact: str = typer.Argument(..., help="any stage artifact")) -> None:
    """Print an artifact readably. The protocol is JSON; this is for eyes."""
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


__all__ = ["app"]
