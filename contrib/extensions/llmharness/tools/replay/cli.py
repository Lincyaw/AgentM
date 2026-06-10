"""``llmharness-replay`` — replay one captured phase invocation.

Subcommands
-----------
* ``extractor`` — rerun an extractor record; prints submit_events args.
* ``auditor``  — rerun an auditor record; prints submit_verdict args.
* ``list``     — index a sidecar file by phase / turn / status / latency.
* ``chain``    — bulk-replay every record in order; flag outputs that
  differ from the recorded baseline.

Each replay subcommand supports ``--provider`` to swap LLM,
``--prompt-override`` to swap system prompt, and (for single-record
replay) ``--diff`` to show the recorded output alongside — the three
knobs we want when bisecting an auditor regression.

CLI style: ``typer`` with ``Annotated`` option metadata, matching the
``agentm`` CLI convention.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from .chain import ChainResult, PhaseFilter, chain_replay_sync
from .engine import PhaseResult
from .prefix_replay import PrefixReplayError, make_plan
from llmharness.replay.record import Phase, ReplayRecord, iter_records
from .runner import replay_auditor_record, replay_extractor_record

app = typer.Typer(
    name="llmharness-replay",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


# --- shared option helpers --------------------------------------------------


def resolve_default_provider_spec(
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Build a default provider tuple from ``AGENTM_*`` / ``OPENAI_*`` env vars.

    Mirrors the precedence ``agentm`` CLI uses in
    :func:`agentm.cli._resolve_provider_model_cwd` + ``ProviderRegistry.build``:
    pulls ``AGENTM_PROVIDER`` (default = registry default), ``AGENTM_MODEL``
    (default = provider's ``default_model``), then routes through the
    descriptor so ``OPENAI_BASE_URL`` / ``OPENAI_VERIFY_SSL`` /
    ``WARPGATE_TICKET`` etc. become entries in the returned config dict.

    Returns ``None`` (deferring to runner defaults) when the registry has no
    extension module for the resolved provider, e.g. ``amazon-bedrock``
    which is currently descriptor-only.
    """
    # Late import: agentm is an optional dependency for the replay package
    # (rca-autorl installs llmharness without the full agentm runtime).
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY

    source = os.environ if env is None else env
    registry = DEFAULT_PROVIDER_REGISTRY
    provider_id = source.get("AGENTM_PROVIDER") or registry.default_provider().id
    try:
        descriptor = registry.resolve(provider_id)
    except KeyError:
        return None
    if descriptor.extension_module is None:
        return None
    model = source.get("AGENTM_MODEL") or descriptor.default_model
    if not isinstance(model, str) or not model:
        return None
    try:
        return registry.build(provider_id, {"model": model}, env=source)
    except KeyError:
        return None


def _parse_provider(spec: str | None) -> tuple[str, dict[str, Any]] | None:
    """Coerce ``--provider`` to a ``(module_path, config)`` tuple.

    Forms:

    * ``None`` (flag omitted) — pull ``AGENTM_PROVIDER`` / ``AGENTM_MODEL``
      from env and apply the same ``ProviderRegistry`` enrichment the
      ``agentm`` CLI uses, so ``OPENAI_BASE_URL`` etc. flow through
      automatically.
    * ``"module:{json_cfg}"`` — explicit config wins; no env bridging.
    * ``"name"`` — bare provider id: look it up in
      :data:`agentm.ai.DEFAULT_PROVIDER_REGISTRY` to env-bridge (matching
      ``agentm --provider <name>`` behaviour); if it isn't a known
      provider, treat as a bare module path with empty config (legacy).
    """
    if not spec:
        return resolve_default_provider_spec()
    if ":" in spec:
        module, payload = spec.split(":", 1)
        try:
            cfg = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"--provider config is not valid JSON: {exc}") from exc
        if not isinstance(cfg, dict):
            raise typer.BadParameter("--provider config JSON must be an object")
        return module.strip(), cfg
    bare = spec.strip()
    # Try registry resolve so ``--provider openai`` env-bridges identically
    # to ``agentm --provider openai``; fall back to "bare module path" for
    # backward compat with third-party providers not in the registry.
    try:
        from agentm.ai import DEFAULT_PROVIDER_REGISTRY

        registry = DEFAULT_PROVIDER_REGISTRY
        descriptor = registry.resolve(bare)
        if descriptor.extension_module is None:
            return bare, {}
        model = os.environ.get("AGENTM_MODEL") or descriptor.default_model
        if not isinstance(model, str) or not model:
            return bare, {}
        return registry.build(bare, {"model": model})
    except KeyError:
        return bare, {}


def _load_prompt_override(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"--prompt-override file not found: {p}")
    return p.read_text(encoding="utf-8")


def _pick_record(path: Path, *, phase: Phase, turn: int | None, index: int | None) -> ReplayRecord:
    # When --turn is given we only need the last match — stream-iterate
    # to avoid loading the entire sidecar (tens of MB on a 50-case run).
    if turn is not None:
        latest: ReplayRecord | None = None
        for rec in iter_records(path):
            if rec.phase == phase and rec.turn_index == turn:
                latest = rec
        if latest is None:
            typer.echo(f"no {phase} record with turn_index={turn} in {path}", err=True)
            raise typer.Exit(code=1)
        return latest

    records = [r for r in iter_records(path) if r.phase == phase]
    if not records:
        typer.echo(f"no {phase} records in {path}", err=True)
        raise typer.Exit(code=1)
    if index is not None:
        try:
            return records[index]
        except IndexError as exc:
            raise typer.BadParameter(
                f"--index {index} out of range (0..{len(records) - 1})"
            ) from exc
    return records[-1]


def _print_result(result: PhaseResult, recorded: ReplayRecord, *, diff: bool) -> None:
    typer.echo(f"# status:      {result.status}")
    typer.echo(f"# latency_ms:  {result.latency_ms}")
    if result.error:
        typer.echo(f"# error:       {result.error}")
    if result.output is not None:
        typer.echo("\n--- replay output ---")
        typer.echo(json.dumps(result.output, ensure_ascii=False, indent=2, default=str))
    if diff:
        typer.echo("\n--- recorded output ---")
        typer.echo(json.dumps(recorded.output or {}, ensure_ascii=False, indent=2, default=str))


# --- shared option type aliases --------------------------------------------
# Annotated[Type, typer.Option(...)] keeps option metadata out of function
# default values (the form ruff flags as B008) while staying compatible
# with typer's annotation reader.

_RecordOpt = Annotated[Path, typer.Option(..., "--record", help="Path to audit_replay/<id>.jsonl")]
_TurnOpt = Annotated[
    int | None,
    typer.Option("--turn", help="Pick by turn_index (last match wins)"),
]
_IndexOpt = Annotated[int | None, typer.Option("--index", help="Pick by 0-based filtered index")]
_CwdOpt = Annotated[Path, typer.Option("--cwd", help="Working directory for the replay session")]
_ProviderOpt = Annotated[
    str | None,
    typer.Option(
        "--provider",
        help="LLM provider spec: ``module`` or ``module:{json_cfg}``",
    ),
]
_PromptOverrideOpt = Annotated[
    str | None,
    typer.Option("--prompt-override", help="Path to system-prompt override"),
]
_DiffOpt = Annotated[bool, typer.Option("--diff", help="Also print recorded output for comparison")]


# --- commands ----------------------------------------------------------------


_RunnerFn = Callable[..., Coroutine[Any, Any, PhaseResult]]
_PHASE_RUNNERS: dict[Phase, _RunnerFn] = {
    "extractor": replay_extractor_record,
    "auditor": replay_auditor_record,
}


def _run_single(
    phase: Phase,
    *,
    record: Path,
    turn: int | None,
    index: int | None,
    cwd: Path,
    provider: str | None,
    prompt_override: str | None,
    diff: bool,
) -> None:
    rec = _pick_record(record, phase=phase, turn=turn, index=index)
    # For extractor replays, union turn_texts from every prior firing
    # in the sidecar into rec.extras["prior_turn_texts"]. The runner
    # uses this to populate state.turn_texts for external_refs witness
    # checks — the live adapter has the full message trajectory, but
    # replay only has what each firing recorded.
    if phase == "extractor":
        prior_texts: dict[str, Any] = {}
        for prior in iter_records(record):
            if prior.ts_ns >= rec.ts_ns:
                continue
            for k, v in (prior.extras.get("turn_texts") or {}).items():
                prior_texts.setdefault(str(k), v)
        if prior_texts:
            merged_extras = dict(rec.extras)
            merged_extras["prior_turn_texts"] = prior_texts
            rec = ReplayRecord(
                phase=rec.phase,
                turn_index=rec.turn_index,
                session_id=rec.session_id,
                trace_id=rec.trace_id,
                ts_ns=rec.ts_ns,
                compose_kwargs=rec.compose_kwargs,
                payload=rec.payload,
                provider=rec.provider,
                output=rec.output,
                status=rec.status,
                error=rec.error,
                latency_ms=rec.latency_ms,
                extras=merged_extras,
            )
    result = asyncio.run(
        _PHASE_RUNNERS[phase](
            rec,
            cwd=str(cwd),
            provider_override=_parse_provider(provider),
            prompt_override=_load_prompt_override(prompt_override),
        )
    )
    _print_result(result, rec, diff=diff)


@app.command()
def extractor(
    record: _RecordOpt,
    turn: _TurnOpt = None,
    index: _IndexOpt = None,
    cwd: _CwdOpt = Path("."),
    provider: _ProviderOpt = None,
    prompt_override: _PromptOverrideOpt = None,
    diff: _DiffOpt = False,
) -> None:
    """Rerun one extractor record."""
    _run_single(
        "extractor",
        record=record,
        turn=turn,
        index=index,
        cwd=cwd,
        provider=provider,
        prompt_override=prompt_override,
        diff=diff,
    )


@app.command()
def auditor(
    record: _RecordOpt,
    turn: _TurnOpt = None,
    index: _IndexOpt = None,
    cwd: _CwdOpt = Path("."),
    provider: _ProviderOpt = None,
    prompt_override: _PromptOverrideOpt = None,
    diff: _DiffOpt = False,
) -> None:
    """Rerun one auditor record."""
    _run_single(
        "auditor",
        record=record,
        turn=turn,
        index=index,
        cwd=cwd,
        provider=provider,
        prompt_override=prompt_override,
        diff=diff,
    )


@app.command(name="list")
def list_records(
    record: _RecordOpt,
    phase: Annotated[
        str | None,
        typer.Option("--phase", help="Filter by phase: extractor or auditor"),
    ] = None,
) -> None:
    """Enumerate records in a sidecar file."""
    if phase not in (None, "extractor", "auditor"):
        raise typer.BadParameter("--phase must be extractor or auditor")
    rows: list[tuple[int, str, int, str, int]] = []
    for i, rec in enumerate(iter_records(record)):
        if phase and rec.phase != phase:
            continue
        rows.append((i, rec.phase, rec.turn_index, rec.status, rec.latency_ms))
    if not rows:
        typer.echo(f"no matching records in {record}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{'idx':>4}  {'phase':<10}  {'turn':>4}  {'status':<12}  {'latency_ms':>10}")
    for i, ph, turn_idx, status, latency in rows:
        typer.echo(f"{i:>4}  {ph:<10}  {turn_idx:>4}  {status:<12}  {latency:>10}")


@app.command()
def chain(
    record: _RecordOpt,
    phase: Annotated[
        str,
        typer.Option("--phase", help="extractor | auditor | both (default both)"),
    ] = "both",
    cwd: _CwdOpt = Path("."),
    provider: _ProviderOpt = None,
    extractor_prompt_override: Annotated[
        str | None,
        typer.Option(
            "--extractor-prompt-override",
            help="Path to extractor system-prompt override",
        ),
    ] = None,
    auditor_prompt_override: Annotated[
        str | None,
        typer.Option(
            "--auditor-prompt-override",
            help="Path to auditor system-prompt override",
        ),
    ] = None,
    diff_changes_only: Annotated[
        bool,
        typer.Option(
            "--diff-changes-only",
            help="Print only records where replay output differs from recorded",
        ),
    ] = False,
) -> None:
    """Bulk-replay every record (in file order); flag diffs vs recorded."""
    if phase not in ("extractor", "auditor", "both"):
        raise typer.BadParameter("--phase must be extractor, auditor, or both")
    phase_filter = cast(PhaseFilter, phase)
    provider_tuple = _parse_provider(provider)
    extractor_prompt = _load_prompt_override(extractor_prompt_override)
    auditor_prompt = _load_prompt_override(auditor_prompt_override)

    def _on_progress(idx: int, cr: ChainResult) -> None:
        changed = cr.result.output != cr.record.output
        if diff_changes_only and not changed:
            return
        marker = "!=" if changed else "=="
        typer.echo(
            f"[{idx:>4}] {cr.record.phase:<10} turn={cr.record.turn_index:>4}  "
            f"status={cr.result.status:<12}  recorded{marker}replayed  "
            f"latency_ms={cr.result.latency_ms}"
        )

    results = chain_replay_sync(
        record,
        cwd=str(cwd),
        phase=phase_filter,
        provider_override=provider_tuple,
        prompt_override_extractor=extractor_prompt,
        prompt_override_auditor=auditor_prompt,
        on_progress=_on_progress,
    )

    diffs = sum(1 for cr in results if cr.result.output != cr.record.output)
    errors = sum(1 for cr in results if cr.result.status != "ok")
    typer.echo(
        f"\n# chain: {len(results)} records replayed, "
        f"{diffs} outputs differ, {errors} replay failures"
    )


@app.command(name="agent-from-reminder")
def agent_from_reminder(
    audit_replay: Annotated[
        Path,
        typer.Option(
            ...,
            "--audit-replay",
            help="Path to .agentm/audit_replay/<session_id>.jsonl",
        ),
    ],
    turn: Annotated[
        int,
        typer.Option(..., "--turn", help="turn_index of the auditor record carrying the reminder"),
    ],
    session_dir: Annotated[
        Path | None,
        typer.Option(
            "--session-dir",
            help=(
                "Directory holding the source main-agent session JSONLs. "
                "Defaults to ``<audit-replay-dir>/../sessions/``."
            ),
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help=(
                "Directory to write the resume-command script into. "
                "Defaults to the audit-replay file's parent dir."
            ),
        ),
    ] = None,
    print_only: Annotated[
        bool,
        typer.Option(
            "--print-only",
            help="Only print the resume command; do not write a script file.",
        ),
    ] = False,
) -> None:
    """Branch a session at the end of turn ``t`` + emit a resume command.

    Given an auditor record that surfaced a reminder at turn ``t``, this
    command:

    1. Opens the source main-agent session JSONL.
    2. Picks the leaf entry that ends turn ``t`` (the ``t``-th
       ``message`` entry on the active branch — same indexing convention
       as ``llmharness.atom``).
    3. Calls ``SessionManager.create_branched_session`` to materialise
       a new persisted session whose tree mirrors the original prefix
       and whose header carries ``parent_session`` pointing at the
       source file.
    4. Prints (and, by default, writes) an ``agentm`` invocation that
       resumes the branched session with
       ``llmharness.replay.reminder_seed`` mounted — so the recorded
       reminder is delivered as the first injection of the next turn.
    """
    resolved_session_dir = (
        session_dir
        if session_dir is not None
        else (audit_replay.parent.parent / "sessions").resolve()
    )
    try:
        plan = make_plan(
            audit_replay_path=audit_replay,
            turn=turn,
            session_dir=resolved_session_dir,
        )
    except PrefixReplayError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"# source session: {plan.source_session_file}")
    typer.echo(f"# branched session: {plan.branched_session_file}")
    typer.echo(f"# branched session id: {plan.branched_session_id}")
    typer.echo("# reminder text:")
    for line in plan.reminder_text.splitlines() or [""]:
        typer.echo(f"#   {line}")
    typer.echo("")
    typer.echo(plan.command)

    if print_only:
        return
    resolved_out_dir = out_dir if out_dir is not None else audit_replay.parent
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    script_path = resolved_out_dir / f"replay-{plan.branched_session_id}-t{turn}.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + plan.command + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    typer.echo(f"\n# wrote: {script_path}")


def main() -> None:
    """Entry point for the ``llmharness-replay`` console script."""
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
