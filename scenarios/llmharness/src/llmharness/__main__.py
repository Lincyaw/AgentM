"""CLI entry: ``python -m llmharness {ingest,tick,inject}``.

Subcommands are designed to be invoked from Claude Code hooks. They speak
JSON on stdin/stdout and exit with code 0 on success. ``inject`` writes the
reminder text (if any) to stdout — silent otherwise — so a UserPromptSubmit
hook can pipe it directly into the next prompt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from .adapters.claude_code import (
    delta_against,
    parse_hook_payload,
    read_transcript_turns,
)
from .schema import Turn
from .store import HarnessStore
from .worker import tick as run_tick

_DEFAULT_ROOT = ".harness"
_ROOT_OPT = typer.Option(help=f"Harness storage root (default: {_DEFAULT_ROOT})")

app = typer.Typer(
    name="llmharness",
    help="LLM-as-harness CLI invoked by Claude Code hooks.",
    no_args_is_help=True,
    add_completion=False,
)


def _read_payload(arg: str) -> dict[str, Any]:
    raw = sys.stdin.read() if arg == "-" else Path(arg).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    return data


@app.command()
def ingest(
    root: Annotated[str, _ROOT_OPT] = _DEFAULT_ROOT,
    session: Annotated[
        str | None,
        typer.Option(help="Session id. Required for --input mode; auto-detected for --from-hook."),
    ] = None,
    input: Annotated[
        str | None,
        typer.Option(help="Path to a JSON payload of {turns: [...]}, or '-' to read stdin."),
    ] = None,
    from_hook: Annotated[
        bool,
        typer.Option(
            "--from-hook",
            help=(
                "Treat stdin as a Claude Code hook payload; derive session_id and "
                "transcript_path from it, then ingest the inbox delta. Silent "
                "no-op if the payload is unrecognizable."
            ),
        ),
    ] = False,
) -> None:
    """Append a transcript delta to the inbox."""
    store = HarnessStore(root)
    if from_hook:
        hook = parse_hook_payload(sys.stdin.read())
        if hook is None:
            return
        sid = session or hook.session_id
        if not hook.has_transcript:
            typer.echo(json.dumps({"appended": 0, "session": sid, "reason": "no transcript"}))
            return
        assert hook.transcript_path is not None  # guarded by has_transcript
        all_turns = read_transcript_turns(hook.transcript_path)
        existing = store.read_inbox(sid)
        delta = delta_against(existing, all_turns)
        store.append_inbox(sid, delta)
        typer.echo(json.dumps({"appended": len(delta), "session": sid}))
        return

    if not session:
        typer.echo("--session is required when not using --from-hook", err=True)
        raise typer.Exit(code=2)
    source = input if input is not None else "-"
    body = _read_payload(source)
    turns = [Turn.from_dict(t) for t in body.get("turns", [])]
    store.append_inbox(session, turns)
    typer.echo(json.dumps({"appended": len(turns)}))


@app.command()
def tick(
    session: Annotated[str, typer.Option(help="Session id.")],
    root: Annotated[str, _ROOT_OPT] = _DEFAULT_ROOT,
    confidence: Annotated[float, typer.Option(help="Drift confidence threshold.")] = 0.6,
    min_reminder_gap: Annotated[
        int,
        typer.Option(help="Minimum number of turns between two reminders for the same session."),
    ] = 5,
) -> None:
    """Run one summarize+detect pass for a session."""
    store = HarnessStore(root)
    result = run_tick(
        store,
        session,
        confidence_threshold=confidence,
        min_reminder_gap=min_reminder_gap,
    )
    payload = {
        "new_events": result.new_event_count,
        "last_turn_index": result.last_turn_index,
        "drift": result.verdict.drift,
        "drift_type": result.verdict.type.value if result.verdict.type else None,
        "confidence": result.verdict.confidence,
        "reminder_written": result.reminder_written,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))


@app.command()
def inject(
    root: Annotated[str, _ROOT_OPT] = _DEFAULT_ROOT,
    session: Annotated[
        str | None,
        typer.Option(help="Session id. Required for direct mode; auto-detected for --from-hook."),
    ] = None,
    from_hook: Annotated[
        bool,
        typer.Option("--from-hook", help="Read session_id from a Claude Code hook payload on stdin."),
    ] = False,
) -> None:
    """Print and consume any pending reminder for a session."""
    store = HarnessStore(root)
    sid = session
    if from_hook:
        payload = parse_hook_payload(sys.stdin.read())
        if payload is None:
            return
        sid = sid or payload.session_id
    if not sid:
        # No session resolvable; silent no-op (matches hook fail-open behavior).
        return
    reminder = store.pop_reminder(sid)
    if reminder is None:
        return
    sys.stdout.write(reminder.text)
    if not reminder.text.endswith("\n"):
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
