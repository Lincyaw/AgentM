"""Commands that inspect or control AgentM sessions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import typer

from agentm import AgentSessionConfig, CompactionRequest
from agentm.cli._display import EXIT_ERROR, EXIT_NOT_FOUND, is_tty, stderr_console
from agentm.config.resolver import DefaultSessionSpecResolver
from agentm.control import (
    InterruptDeliveryError,
    send_interrupt,
)
from agentm.extensions.builtin.llm_compaction import LlmCompactionConfig
from agentm.presenter.compaction import AgentSessionCompactor


session_app = typer.Typer(
    name="session",
    help="Inspect or control AgentM sessions.",
    add_completion=False,
)


def _select_format(fmt: str | None) -> str:
    if fmt not in {None, "text", "ndjson"}:
        raise typer.BadParameter(
            "format must be 'text' or 'ndjson'",
            param_hint="--format",
        )
    return fmt or ("text" if is_tty() else "ndjson")


def _emit_error(
    *,
    session_id: str,
    error_type: str,
    detail: str,
    fmt: str,
) -> None:
    if fmt == "ndjson":
        sys.stderr.write(
            json.dumps(
                {
                    "session_id": session_id,
                    "status": "error",
                    "error_type": error_type,
                    "error_detail": detail,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return
    stderr_console.print(f"[red]error: {detail}[/red]")


@session_app.command("interrupt")
def interrupt_cmd(
    session_id: str = typer.Argument(..., metavar="SESSION_ID"),
    message: str = typer.Option(..., "--message", "-m"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Interrupt the active turn and enqueue operator feedback."""
    chosen_format = _select_format(fmt)
    try:
        asyncio.run(send_interrupt(session_id, message))
    except FileNotFoundError as exc:
        _emit_error(
            session_id=session_id,
            error_type="session_not_found",
            detail=str(exc),
            fmt=chosen_format,
        )
        raise typer.Exit(EXIT_NOT_FOUND)
    except (InterruptDeliveryError, OSError, ValueError) as exc:
        _emit_error(
            session_id=session_id,
            error_type=type(exc).__name__,
            detail=str(exc),
            fmt=chosen_format,
        )
        raise typer.Exit(EXIT_ERROR)

    message_chars = len(message)
    record = {
        "session_id": session_id,
        "status": "accepted",
        "message_chars": message_chars,
    }
    if chosen_format == "ndjson":
        sys.stdout.write(json.dumps(record) + "\n")
    else:
        sys.stdout.write(
            f"interrupt accepted by session {session_id} ({message_chars} chars)\n"
        )


@session_app.command("compact")
def compact_cmd(
    session_id: str = typer.Argument(..., metavar="SESSION_ID"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Generate an auditable summary from committed session history."""
    chosen_format = _select_format(fmt)
    cwd = Path.cwd()
    project_config = cwd / "agentm.toml"
    resolver = DefaultSessionSpecResolver(
        project_config=project_config if project_config.exists() else None,
    )
    strategy = LlmCompactionConfig(
        keep_last_turns=4,
    )
    compactor = AgentSessionCompactor(
        AgentSessionConfig(
            cwd=str(cwd),
            extensions=[],
            spec_resolver=resolver,
        )
    )
    try:
        result = asyncio.run(
            compactor.compact(
                CompactionRequest(
                    source_session_id=session_id,
                    options=strategy.model_dump(mode="json"),
                )
            )
        )
    except (FileNotFoundError, KeyError) as exc:
        _emit_error(
            session_id=session_id,
            error_type="session_not_found",
            detail=str(exc),
            fmt=chosen_format,
        )
        raise typer.Exit(EXIT_NOT_FOUND)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit_error(
            session_id=session_id,
            error_type=type(exc).__name__,
            detail=str(exc),
            fmt=chosen_format,
        )
        raise typer.Exit(EXIT_ERROR)

    record = {
        "session_id": session_id,
        "status": "complete",
        "operation": "compact",
        "covered_start_turn_index": result.covered.start,
        "covered_end_turn_index": result.covered.end,
        "covered_through_turn_id": result.covered_through_turn_id,
        "producer_ref": result.producer_ref,
        "summary": result.summary,
    }
    if chosen_format == "ndjson":
        sys.stdout.write(json.dumps(record) + "\n")
    else:
        sys.stdout.write(result.summary + "\n")


__all__ = ["session_app"]
