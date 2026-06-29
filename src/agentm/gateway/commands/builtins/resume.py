"""``/resume`` — switch this chat to a different session.

Bare ``/resume`` lists recent sessions as a ``session_list`` outbound so
interactive clients can open a picker; ``/resume latest`` jumps to the
most recent; ``/resume <prefix>`` resolves a unique prefix match;
``/resume <full_id>`` is an exact switch.

Shuts down the current in-memory session and points the persistent
:class:`ChatSessionMap` entry at the target so the next inbound message
resumes from that session's transcript.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..protocol import (
    CommandContext,
    CommandInvocation,
    CommandKind,
    CommandResult,
    OutboundBody,
)


def _resumed_notice(sid: str) -> str:
    return (
        f"\U0001f504 Resumed session `{sid[:12]}…`. "
        "Next message continues from that session's transcript."
    )


@dataclass(slots=True)
class ResumeCommand:
    name: str = "resume"
    namespace: str | None = None
    summary: str = "Resume a previous session (bare: list, or /resume <id|prefix|latest>)"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        target = inv.args.strip()

        if not target:
            return await self._list_sessions(ctx)

        if target.lower() == "latest":
            target = "latest"

        return await self._resume_by_id(target, ctx)

    # -- bare /resume: list recent sessions --------------------------------

    async def _list_sessions(self, ctx: CommandContext) -> CommandResult:
        sessions = await asyncio.to_thread(
            _discover_sessions, ctx.cwd, limit=30, with_titles=True
        )
        if not sessions:
            return CommandResult(
                outbound=[ctx.notice("No previous sessions found.")]
            )
        text = _format_text_list(sessions, ctx)
        outbound = OutboundBody(
            channel=ctx.channel,
            chat_id=ctx.chat_id,
            content=text,
            thread_id=ctx.thread_id,
            metadata={
                "kind": "session_list",
                "sessions": sessions,
            },
        )
        return CommandResult(outbound=[outbound])

    # -- /resume <id_or_prefix|latest> -------------------------------------

    async def _resume_by_id(
        self, target: str, ctx: CommandContext
    ) -> CommandResult:
        is_latest = target == "latest"
        sessions = await asyncio.to_thread(
            _discover_sessions,
            ctx.cwd,
            limit=1 if is_latest else 200,
            with_titles=False,
        )

        if is_latest:
            if not sessions:
                return CommandResult(
                    outbound=[
                        ctx.reply(
                            "No previous sessions found.", kind="diagnostic_error"
                        )
                    ]
                )
            sid = sessions[0]["session_id"]
            current = ctx.get_route_stats().get("session_id")
            if current == sid:
                return CommandResult(
                    outbound=[ctx.notice("Already on the latest session.")]
                )
            return await self._resume_with_history(ctx, sid)

        target_lower = target.lower()
        exact = [s for s in sessions if s["session_id"] == target_lower]
        if exact:
            sid = exact[0]["session_id"]
            return await self._resume_with_history(ctx, sid)

        prefix = [
            s for s in sessions if s["session_id"].startswith(target_lower)
        ]
        if len(prefix) == 1:
            sid = prefix[0]["session_id"]
            return await self._resume_with_history(ctx, sid)
        if len(prefix) > 1:
            ids = ", ".join(f"`{s['session_id'][:12]}…`" for s in prefix[:5])
            return CommandResult(
                outbound=[
                    ctx.reply(
                        f"Ambiguous prefix `{target}` matches {len(prefix)} "
                        f"sessions: {ids}. Use a longer prefix.",
                        kind="diagnostic_error",
                    )
                ]
            )
        return CommandResult(
            outbound=[
                ctx.reply(
                    f"Session not found: `{target}`",
                    kind="diagnostic_error",
                )
            ]
        )

    async def _resume_with_history(
        self, ctx: CommandContext, sid: str
    ) -> CommandResult:
        await ctx.resume_session(sid)
        outbound: list[OutboundBody] = []
        history = await ctx.load_session_history(sid)
        if history is not None:
            outbound.append(ctx.session_history(history))
        outbound.append(ctx.notice(_resumed_notice(sid)))
        return CommandResult(outbound=outbound)



# --- session discovery (ClickHouse → JSONL fallback) ----------------------


def _discover_sessions(
    cwd: str, *, limit: int = 30, with_titles: bool = True
) -> list[dict[str, Any]]:
    """List recent root sessions, newest first.

    Tries ClickHouse when available; falls back to scanning the JSONL
    observability directory.  When *with_titles* is False the expensive
    per-session first-user-message lookup is skipped (used by the
    resume-by-id / latest paths that only need session_id).

    Called via ``asyncio.to_thread`` so blocking I/O does not stall the
    gateway event loop.
    """
    try:
        from agentm.core.observability.clickhouse import get_url

        ch_url = get_url()
    except Exception:
        ch_url = None
    if ch_url:
        try:
            result = _discover_clickhouse(
                ch_url, cwd, limit=limit, with_titles=with_titles
            )
            if result:
                return result
        except Exception:
            logger.opt(exception=True).debug(
                "ClickHouse session listing failed, falling back to JSONL"
            )
    return _discover_jsonl(cwd, limit=limit, with_titles=with_titles)


# -- ClickHouse path -------------------------------------------------------


def _discover_clickhouse(
    url: str, cwd: str, *, limit: int = 30, with_titles: bool = True
) -> list[dict[str, Any]]:
    from agentm.core.observability.clickhouse import recent_sessions

    rows = recent_sessions(url, cwd=cwd, limit=limit)
    titles: dict[str, str] = {}
    if with_titles and rows:
        from agentm.core.observability.clickhouse import first_user_message

        for r in rows:
            titles[r["session_id"]] = first_user_message(url, r["session_id"])

    result: list[dict[str, Any]] = []
    for r in rows:
        sid = r["session_id"]
        created = _parse_timestamp(r.get("created_at", ""))
        result.append({
            "session_id": sid,
            "title": titles.get(sid, ""),
            "scenario": r.get("scenario") or "",
            "created_at": created,
        })
    return result


def _parse_timestamp(raw: str) -> float:
    if not raw:
        return 0.0
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return 0.0


# -- JSONL path ------------------------------------------------------------


def _discover_jsonl(
    cwd: str, *, limit: int = 30, with_titles: bool = True
) -> list[dict[str, Any]]:
    """Scan the observability directory using :class:`TraceReader`."""
    from agentm.core.lib.trace_reader import TraceReader
    from agentm.core.observability.otel_export import resolve_observability_dir

    obs_dir = resolve_observability_dir(cwd)
    if not obs_dir.is_dir():
        return []

    file_stats: list[tuple[float, Any]] = []
    for p in obs_dir.glob("*.jsonl"):
        if not p.is_file():
            continue
        try:
            file_stats.append((p.stat().st_mtime, p))
        except OSError:
            continue
    file_stats.sort(key=lambda t: t[0], reverse=True)

    result: list[dict[str, Any]] = []
    for mtime, f in file_stats[:limit]:
        reader = TraceReader(f)
        identity, _ = reader.scan_identity_and_line_count()
        if identity is None:
            continue
        if identity.parent_session_id:
            continue
        sid = identity.session_id or f.stem
        title = _first_user_message(reader) if with_titles else ""
        result.append({
            "session_id": sid,
            "title": title,
            "scenario": identity.scenario or "",
            "created_at": mtime,
        })
    return result


def _first_user_message(reader: Any) -> str:
    """Extract the first user message's content as a session title.

    Uses ``iter_log_records`` with early return to avoid reading the
    entire trace file.
    """
    try:
        for record in reader.iter_log_records(name="agentm.message.appended"):
            body = record.body if isinstance(record.body, dict) else {}
            payload = body.get("payload") or {}
            if payload.get("role") != "user":
                continue
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                return content[:120]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if isinstance(text, str) and text.strip():
                            return text[:120]
            return ""
    except Exception:
        logger.debug("failed to extract first user message from {}", reader.file_path)
    return ""


# --- text fallback --------------------------------------------------------


def _format_text_list(
    sessions: list[dict[str, Any]], ctx: CommandContext
) -> str:
    stats = ctx.get_route_stats()
    current = stats.get("session_id")
    lines = ["**Recent sessions** (use `/resume <id>` to switch)\n"]
    now = time.time()
    for s in sessions[:20]:
        sid = s["session_id"]
        title = s.get("title") or "(untitled)"
        if len(title) > 60:
            title = title[:57] + "…"
        age = _format_age(now - s.get("created_at", 0))
        marker = " ← current" if sid == current else ""
        lines.append(f"- `{sid[:12]}` {title} ({age}){marker}")
    if len(sessions) > 20:
        lines.append(f"\n_{len(sessions) - 20} more sessions not shown_")
    return "\n".join(lines)


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    days = int(seconds / 86400)
    if days < 7:
        return f"{days}d ago"
    return f"{days // 7}w ago"


HANDLER = ResumeCommand()
