"""Commands that inspect or control AgentM sessions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import typer

from agentm import CompactionRequest, CompactionResult
from agentm.cli._display import EXIT_ERROR, EXIT_NOT_FOUND, is_tty, stderr_console
from agentm.control import (
    InterruptDeliveryError,
    send_interrupt,
)


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


_COMPACTION_HOST_EXTENSIONS = [
    ("agentm.extensions.builtin.local_backend", {}),
    ("agentm.extensions.builtin.llm_compaction", {"keep_last_turns": 4}),
]


async def _run_compact(session_id: str) -> CompactionResult:
    """Compact a stored session via a minimal compaction-host session.

    The CLI composes a session containing the compaction atom and consumes
    its SESSION_COMPACTOR service — the sanctioned external-callability
    path; it never imports atom internals.
    """
    from agentm.config import DefaultSessionSpecResolver
    from agentm.core.abi.roles import SESSION_COMPACTOR
    from agentm.core.abi.session_api import AgentSessionConfig
    from agentm.sdk import AgentSession
    from agentm.storage.trajectory.resolve import resolve_trajectory_store_or_create

    project_candidate = Path.cwd() / "agentm.toml"
    resolver = DefaultSessionSpecResolver(
        project_config=(str(project_candidate) if project_candidate.exists() else None),
        user_config=None,
    )
    resolved = resolve_trajectory_store_or_create(str(Path.cwd()))
    try:
        config = AgentSessionConfig(
            cwd=str(Path.cwd()),
            extensions=list(_COMPACTION_HOST_EXTENSIONS),
            spec_resolver=resolver,
            trajectory_store=resolved.store,
            purpose="compaction_host",
        )
        spec = resolver.resolve(config)
        if spec.provider is None:
            raise RuntimeError(
                "no default model configured; set default_provider in config.toml"
            )
        session = await AgentSession.create(config)
        try:
            compactor = session.services.require_role(SESSION_COMPACTOR)
            return await compactor.compact(
                CompactionRequest(
                    source_session_id=session_id,
                    options={"keep_last_turns": 4},
                )
            )
        finally:
            await session.shutdown()
    finally:
        resolved.close()


@session_app.command("compact")
def compact_cmd(
    session_id: str = typer.Argument(..., metavar="SESSION_ID"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Generate an auditable summary from committed session history."""
    chosen_format = _select_format(fmt)
    try:
        result = asyncio.run(_run_compact(session_id))
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
