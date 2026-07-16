"""Catalog indexer for atom-attribution metrics.

This module is the only writer of `metrics.jsonl`. It never writes
`decisions.jsonl`; that file is reserved for future decision-making
layers and is not derivable from raw observability.
"""

from __future__ import annotations

import json
from loguru import logger
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from agentm.core.lib.paths import expand_path
from agentm.core.runtime.catalog import _layout
from agentm.core.lib.trace_reader import TraceReader


@dataclass(slots=True)
class IndexerResult:
    n_atoms_attributed: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _EpochState:
    fingerprint: dict[str, str] | None = None
    scenario: str | None = None
    last_stop_reason: str | None = None
    tokens_total: int | None = None
    saw_tokens: bool = False
    saw_reload: bool = False

    def reset_for_reload(self, fingerprint: dict[str, str] | None, scenario: str | None) -> None:
        self.fingerprint = fingerprint
        self.scenario = scenario
        self.last_stop_reason = None
        self.tokens_total = None
        self.saw_tokens = False
        self.saw_reload = True


def _now_iso8601_utc() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_atom_versions(payload: Any) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    atoms = payload.get("atoms")
    if not isinstance(atoms, dict):
        return None
    parsed: dict[str, str] = {}
    for name, value in atoms.items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        _, _, version_hash = value.partition("@")
        if version_hash:
            parsed[name] = version_hash
    return parsed or None


def _parse_scenario(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    scenario = payload.get("scenario")
    return scenario if isinstance(scenario, str) else None


def _extract_usage_tokens(turn_body: Any) -> int | None:
    """Return input+output token count from an ``agentm.turn.summary`` log
    record body. Returns ``None`` when usage is absent (e.g. providers
    that don't report it).
    """
    if not isinstance(turn_body, dict):
        return None
    input_tokens = turn_body.get("input_tokens")
    output_tokens = turn_body.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return input_tokens + output_tokens


_CAUSE_KIND_TO_LABEL: dict[str, str] = {
    "ModelEndTurn": "end_turn",
    "ToolTerminated": "tool_terminated",
    "MaxTurnsExhausted": "max_turns",
    "SignalAborted": "aborted",
    "BudgetExhausted": "budget",
    # ProviderTruncated and ProviderProtocolViolation are mapped from
    # their own discriminators below (kind=max_tokens/error, or
    # the literal "protocol_violation" label).
}


def _extract_stop_reason_from_body(body: Any) -> str | None:
    """Read the termination identity from the body of an
    ``agentm.agent.end`` log record.

    The body shape is ``{"cause": <to_jsonable(TerminationCause)>, ...}``.
    The cause dict carries a ``cause_kind`` discriminator (the class name
    stamped by the standard dataclass serializer in
    ``agentm.core.lib.serialization``) plus any subclass-specific fields.

    Also tolerates a bare ``stop_reason`` string at the top level — used
    by legacy unit-test fixtures that hand-build the body without going
    through the writer.
    """
    if not isinstance(body, dict):
        return None
    legacy = body.get("stop_reason")
    if isinstance(legacy, str):
        return legacy
    cause = body.get("cause")
    if not isinstance(cause, dict):
        return None
    cause_kind = cause.get("cause_kind")
    if isinstance(cause_kind, str):
        if cause_kind == "ProviderTruncated":
            inner = cause.get("kind")
            return inner if inner in {"max_tokens", "error"} else "max_tokens"
        if cause_kind == "ProviderProtocolViolation":
            return "protocol_violation"
        label = _CAUSE_KIND_TO_LABEL.get(cause_kind)
        if label is not None:
            detail = cause.get("detail")
            if cause_kind == "BudgetExhausted" and isinstance(detail, str) and detail:
                return f"budget:{detail}"
            return label
    return None


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _symlink_run(trace_path: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    os.symlink(trace_path.resolve(), dest)


def _ensure_version_dir(
    atom_name: str,
    expected_hash: str,
    *,
    cwd_root: Path,
) -> str:
    from agentm.core._internal.catalog import get_source_at

    if not _is_content_hash(expected_hash):
        raise ValueError(
            f"atom {atom_name!r} has invalid content hash {expected_hash!r}"
        )
    # Reading the snapshot verifies source bytes, manifest identity, and the
    # directory hash before any derived attribution state is written.
    get_source_at(atom_name, expected_hash, cwd_root)
    return expected_hash


def _is_content_hash(value: str) -> bool:
    return len(value) == 12 and all(ch in "0123456789abcdef" for ch in value)


def _build_metrics_row(
    *,
    trace_id: str,
    scenario: str | None,
    completion_rate: float,
    tokens_per_task: int | None,
    mid_session_reload: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "indexed_at": _now_iso8601_utc(),
        "scenario": scenario,
        "task_type": None,
        "n_runs": 1,
        "metrics": {
            "task.completion_rate": completion_rate,
            "tokens_per_task": tokens_per_task,
        },
        "trace_id": trace_id,
        "regressed": False,
    }
    if mid_session_reload:
        row["mid_session_reload"] = True
    return row


def index_trace(trace_path: Path, *, root: Path | None = None) -> IndexerResult:
    """Index a single observability trace into the catalog.

    ``root`` is interpreted as the **cwd** (parent of ``.agentm/``); pass the
    same value the runtime uses for ``AgentSessionConfig.cwd``. ``None``
    derives the cwd from the trace path layout.
    """

    trace_path = trace_path.resolve()
    if root is None:
        cwd_root = trace_path.parent.parent.parent
    else:
        cwd_root = expand_path(root).resolve()
    _layout.catalog_root(root=cwd_root).mkdir(parents=True, exist_ok=True)
    result = IndexerResult()
    trace_id = trace_path.stem
    state = _EpochState()

    # OTLP/JSON ndjson: each line is one ResourceSpans or ResourceLogs
    # element. The indexer reads only log records (identity events live
    # there); spans carry timing/lineage but no aggregation signal we
    # don't already get from the paired log records.
    #
    # NB: the indexer's rebuild-from-raw idempotence (fail-stop test in
    # ``tests/unit/core/catalog/test_indexer.py``) depends on this walk
    # being deterministic over the on-disk byte order. TraceReader's
    # ``iter_log_records`` is a plain file-order walk over scopeLogs
    # entries, so order is preserved.
    try:
        log_records = list(TraceReader(trace_path).iter_log_records())
    except json.JSONDecodeError as exc:  # pragma: no cover — TraceReader skips
        raise ValueError(f"{trace_path}: invalid JSON: {exc}") from exc
    for record in log_records:
        event_name = record.event_name
        body = record.body
        if event_name == "agentm.session.fingerprint":
            if isinstance(body, dict):
                parsed = _parse_atom_versions(body)
                if parsed is not None:
                    state.fingerprint = parsed
                    state.scenario = _parse_scenario(body)
        elif event_name == "agentm.atom.reload":
            next_fingerprint = None
            next_scenario = state.scenario
            if isinstance(body, dict):
                next_fingerprint = _parse_atom_versions(
                    body.get("fingerprint_after")
                )
                next_scenario = (
                    _parse_scenario(body.get("fingerprint_after"))
                    or next_scenario
                )
            state.reset_for_reload(
                next_fingerprint or state.fingerprint, next_scenario
            )
        elif event_name == "agentm.agent.end":
            stop_reason = _extract_stop_reason_from_body(body)
            if stop_reason is not None:
                state.last_stop_reason = stop_reason
        elif event_name == "agentm.turn.summary":
            token_count = _extract_usage_tokens(body)
            if token_count is not None:
                state.tokens_total = (state.tokens_total or 0) + token_count
                state.saw_tokens = True

    if state.fingerprint is None:
        result.warnings.append(f"trace {trace_id!r} missing session.fingerprint record")
        return result

    # "Success" = the model voluntarily finished or a terminal tool ran to
    # completion. Provider truncation, max_turns, signal abort, and budget
    # exhaustion are all fail-stops.
    completion_rate = (
        1.0
        if state.last_stop_reason in {"end_turn", "stop", "tool_terminated"}
        else 0.0
    )
    tokens_per_task = state.tokens_total if state.saw_tokens else None

    for atom_name, expected_hash in sorted(state.fingerprint.items()):
        resolved_hash = _ensure_version_dir(
            atom_name,
            expected_hash,
            cwd_root=cwd_root,
        )
        row = _build_metrics_row(
            trace_id=trace_id,
            scenario=state.scenario,
            completion_rate=completion_rate,
            tokens_per_task=tokens_per_task,
            mid_session_reload=state.saw_reload,
        )
        metrics_path = _layout.atom_metrics_path(atom_name, resolved_hash, root=cwd_root)
        _append_jsonl_row(metrics_path, row)
        run_link = _layout.atom_runs_dir(atom_name, resolved_hash, root=cwd_root) / trace_id
        _symlink_run(trace_path, run_link)
        result.n_atoms_attributed += 1

    return result


def _wipe_derived_catalog_state(cwd_root: Path) -> None:
    atoms_root = _layout.atoms_dir(root=cwd_root)
    if not atoms_root.exists():
        return
    for atom_dir in atoms_root.iterdir():
        if not atom_dir.is_dir():
            continue
        for version_dir in atom_dir.iterdir():
            if not version_dir.is_dir():
                continue
            metrics_path = version_dir / _layout.METRICS_FILENAME
            if metrics_path.exists():
                metrics_path.unlink()
            runs_dir = version_dir / _layout.RUNS_DIRNAME
            if runs_dir.exists():
                for child in runs_dir.iterdir():
                    child.unlink()


def rebuild_catalog(*, root: Path, observability: Path) -> tuple[int, int, int, int]:
    _wipe_derived_catalog_state(root)
    n_traces = 0
    n_atoms = 0
    n_warnings = 0
    failures = 0
    if not observability.exists():
        return (0, 0, 0, 0)
    for trace_path in sorted(observability.glob("*.jsonl")):
        n_traces += 1
        try:
            result = index_trace(trace_path, root=root)
        except Exception as exc:
            failures += 1
            logger.warning(f"agentm catalog rebuild failed for {trace_path}: {exc!r}")
            continue
        n_atoms += result.n_atoms_attributed
        n_warnings += len(result.warnings)
    return (n_traces, n_atoms, n_warnings, failures)


app = typer.Typer(add_completion=False)


@app.command()
def rebuild(
    root: Annotated[Path, typer.Option(help="Project root.")] = Path.cwd(),
    observability: Annotated[
        Path | None,
        typer.Option(help="Observability directory (defaults to ProjectLayout)."),
    ] = None,
) -> None:
    """Re-derive catalog metrics from raw observability traces."""
    resolved_root = expand_path(root).resolve()
    if observability is None:
        # Resolve through the default project layout so the on-disk policy
        # lives in one place. Constructed lazily — no filesystem touch at
        # import time.
        from agentm.core.runtime.catalog import DefaultProjectLayout

        observability = DefaultProjectLayout(cwd=resolved_root).observability_root()
    resolved_obs = expand_path(observability).resolve()
    n_traces, n_atoms, n_warnings, failures = rebuild_catalog(
        root=resolved_root, observability=resolved_obs
    )
    typer.echo(f"n_traces={n_traces} n_atoms_attributed={n_atoms} n_warnings={n_warnings}")
    raise typer.Exit(code=1 if failures else 0)


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
