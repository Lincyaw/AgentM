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
from pathlib import Path
from typing import Annotated, Any

import typer

from .chain import ChainResult, chain_replay_sync
from .engine import PhaseResult
from .record import ReplayRecord, iter_records, read_records
from .runner import replay_auditor_record, replay_extractor_record

app = typer.Typer(
    name="llmharness-replay",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


# --- shared option helpers --------------------------------------------------


def _parse_provider(spec: str | None) -> tuple[str, dict[str, Any]] | None:
    """``module`` or ``module:json_config`` → coerced provider tuple."""
    if not spec:
        return None
    if ":" in spec:
        module, payload = spec.split(":", 1)
        try:
            cfg = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(
                f"--provider config is not valid JSON: {exc}"
            ) from exc
        if not isinstance(cfg, dict):
            raise typer.BadParameter("--provider config JSON must be an object")
        return module.strip(), cfg
    return spec.strip(), {}


def _load_prompt_override(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"--prompt-override file not found: {p}")
    return p.read_text(encoding="utf-8")


def _pick_record(
    path: Path, *, phase: str, turn: int | None, index: int | None
) -> ReplayRecord:
    records = read_records(path, phase=phase)  # type: ignore[arg-type]
    if not records:
        typer.echo(f"no {phase} records in {path}", err=True)
        raise typer.Exit(code=1)
    if turn is not None:
        candidates = [r for r in records if r.turn_index == turn]
        if not candidates:
            typer.echo(
                f"no {phase} record with turn_index={turn} in {path}", err=True
            )
            raise typer.Exit(code=1)
        return candidates[-1]
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
        typer.echo(
            json.dumps(recorded.output or {}, ensure_ascii=False, indent=2, default=str)
        )


# --- shared option type aliases --------------------------------------------
# Annotated[Type, typer.Option(...)] keeps option metadata out of function
# default values (the form ruff flags as B008) while staying compatible
# with typer's annotation reader.

_RecordOpt = Annotated[
    Path, typer.Option(..., "--record", help="Path to audit_replay/<id>.jsonl")
]
_TurnOpt = Annotated[
    int | None,
    typer.Option("--turn", help="Pick by turn_index (last match wins)"),
]
_IndexOpt = Annotated[
    int | None, typer.Option("--index", help="Pick by 0-based filtered index")
]
_CwdOpt = Annotated[
    Path, typer.Option("--cwd", help="Working directory for the replay session")
]
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
_DiffOpt = Annotated[
    bool, typer.Option("--diff", help="Also print recorded output for comparison")
]


# --- commands ----------------------------------------------------------------


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
    rec = _pick_record(record, phase="extractor", turn=turn, index=index)
    result = asyncio.run(
        replay_extractor_record(
            rec,
            cwd=str(cwd),
            provider_override=_parse_provider(provider),
            prompt_override=_load_prompt_override(prompt_override),
        )
    )
    _print_result(result, rec, diff=diff)


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
    rec = _pick_record(record, phase="auditor", turn=turn, index=index)
    result = asyncio.run(
        replay_auditor_record(
            rec,
            cwd=str(cwd),
            provider_override=_parse_provider(provider),
            prompt_override=_load_prompt_override(prompt_override),
        )
    )
    _print_result(result, rec, diff=diff)


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
    typer.echo(
        f"{'idx':>4}  {'phase':<10}  {'turn':>4}  {'status':<12}  {'latency_ms':>10}"
    )
    for i, ph, turn_idx, status, latency in rows:
        typer.echo(
            f"{i:>4}  {ph:<10}  {turn_idx:>4}  {status:<12}  {latency:>10}"
        )


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
        phase=phase,  # type: ignore[arg-type]
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


def main() -> None:
    """Entry point for the ``llmharness-replay`` console script."""
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
