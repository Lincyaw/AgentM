"""Shared source selection for trace command modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from agentm.env import autoload_dotenv, resolve_cli_cwd

_clickhouse_module: Any = None


def clickhouse() -> Any:
    """Lazily import the ClickHouse backend module."""
    global _clickhouse_module
    if _clickhouse_module is None:
        from agentm.core.observability import clickhouse as module

        _clickhouse_module = module
    return _clickhouse_module


def trace_cwd(cwd: Path | None = None) -> Path:
    return resolve_cli_cwd(cwd)


def trace_clickhouse_url(cwd: Path | None = None) -> str | None:
    """Return a URL only when remote trace storage is configured."""
    autoload_dotenv(trace_cwd(cwd))
    if not (
        os.environ.get("AGENTM_CLICKHOUSE_URL")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None
    return clickhouse().get_url()


def local_session_file(
    session: str | None,
    cwd: Path | None,
) -> Path | None:
    """Return the local JSONL path for an explicit session if it exists."""
    if session is None:
        return None
    from agentm.core.observability.otel_export import resolve_observability_dir

    path = resolve_observability_dir(trace_cwd(cwd)) / f"{session}.jsonl"
    return path if path.is_file() else None


def clickhouse_session(
    file: Path | None,
    session: str | None,
    latest: bool,
    cwd: Path | None = None,
) -> tuple[str, str] | None:
    """Resolve a ClickHouse-backed session or request local fallback."""
    if file is not None:
        return None
    url = trace_clickhouse_url(cwd)
    if url is None:
        return None
    local_file = local_session_file(session, cwd)
    if local_file is not None:
        try:
            if clickhouse().session_header(url, session) is None:
                logger.debug(
                    "trace: ClickHouse has no header for session {}; "
                    "using local JSONL {}",
                    session,
                    local_file,
                )
                return None
        except Exception as exc:
            logger.debug(
                "trace: ClickHouse session lookup failed for {}; "
                "using local JSONL {}: {}",
                session,
                local_file,
                exc,
            )
            return None
    resolved_cwd = trace_cwd(cwd).resolve()
    session_id = clickhouse().resolve_session(
        url,
        session,
        latest,
        str(resolved_cwd),
    )
    if session_id is None:
        return None
    return url, session_id


__all__ = [
    "clickhouse",
    "clickhouse_session",
    "local_session_file",
    "trace_clickhouse_url",
    "trace_cwd",
]
