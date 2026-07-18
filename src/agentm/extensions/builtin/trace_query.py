"""Builtin ``trace_query`` atom — query the parent session's trajectory.

Exposes read-only tools that let a child session inspect its parent's
conversation trajectory. ClickHouse is used when configured; otherwise the
tools fall back to the local ``$AGENTM_HOME/observability/<parent>.jsonl``
trace. Scoped automatically to ``api.parent_session_id`` — no session id
parameter needed, and no access to other sessions.

Typical consumer: the ``goal`` atom's checker agent, which needs to
verify whether the parent agent's work satisfies a completion condition.

§11: single file; ``MANIFEST`` + ``install(session, config)``; no atom-to-atom
imports; ``core.abi`` only; no ``core.runtime.*`` / ``core._internal``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Protocol

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import (
    FunctionTool,
    TextContent,
    ToolResult,
    TraceReader,
)
from agentm.core.lib import resolve_observability_dir
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="trace_query",
    description=(
        "Read-only trajectory query tools scoped to the parent session. "
        "Lets a child agent inspect what its parent did: list turns, "
        "read specific messages, query tool calls."
    ),
    registers=(
        "tool:list_turns",
        "tool:read_turn",
        "tool:get_tool_calls",
    ),
    requires=(),
    api_version=1,
    tier=1,
)


def _text(s: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=s)])


class _TraceBackend(Protocol):
    def turns(self) -> list[dict[str, Any]]: ...

    def messages(self, *, roles: set[str] | None = None) -> list[dict[str, Any]]: ...

    def tools(self) -> list[dict[str, Any]]: ...


class _ClickHouseTraceBackend:
    def __init__(self, module: Any, url: str, session_id: str) -> None:
        self._module = module
        self._url = url
        self._session_id = session_id

    def turns(self) -> list[dict[str, Any]]:
        return list(self._module.turns(self._url, self._session_id))

    def messages(self, *, roles: set[str] | None = None) -> list[dict[str, Any]]:
        return list(self._module.messages(self._url, self._session_id, roles=roles))

    def tools(self) -> list[dict[str, Any]]:
        return list(self._module.tools(self._url, self._session_id))


class _LocalTraceBackend:
    def __init__(self, path: Path) -> None:
        self._path = path

    def turns(self) -> list[dict[str, Any]]:
        return TraceReader(self._path).load_turn_summaries()

    def messages(self, *, roles: set[str] | None = None) -> list[dict[str, Any]]:
        records = TraceReader(self._path).load_messages()
        if roles is None:
            return records
        return [
            record
            for record in records
            if isinstance(record.get("payload"), dict)
            and record["payload"].get("role") in roles
        ]

    def tools(self) -> list[dict[str, Any]]:
        return [
            _tool_record(span, args_log, result_log)
            for span, args_log, result_log in TraceReader(self._path).tool_calls()
        ]


def _tool_record(span: Any, args_log: Any, result_log: Any) -> dict[str, Any]:
    tool_name = (
        span.attributes.get("gen_ai.tool.name")
        or span.name.removeprefix("execute_tool ").strip()
    )
    args_payload: Any = args_log.body if args_log is not None else None
    if args_payload is None:
        args_payload = _json_attr(span.attributes.get("gen_ai.tool.call.arguments"))
    result_payload: Any = result_log.body if result_log is not None else None
    if result_payload is None:
        result_payload = _json_attr(span.attributes.get("gen_ai.tool.call.result"))
    return {"tool": tool_name, "args": args_payload, "result": result_payload}


def _json_attr(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _get_clickhouse_backend(
    parent_sid: str, *, local_trace_path: Path | None
) -> _TraceBackend | None:
    """Return a ClickHouse backend, or None when local fallback should be used."""
    try:
        from agentm.core.observability import clickhouse

        url = clickhouse.get_url()
        if url is None:
            return None
        if local_trace_path is not None:
            try:
                if clickhouse.session_header(url, parent_sid) is None:
                    logger.debug(
                        "trace_query: ClickHouse has no header for parent session {}; "
                        "using local JSONL {}",
                        parent_sid,
                        local_trace_path,
                    )
                    return None
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "trace_query: ClickHouse lookup failed for parent session {}; "
                    "using local JSONL {}: {}",
                    parent_sid,
                    local_trace_path,
                    exc,
                )
                return None
        return _ClickHouseTraceBackend(clickhouse, url, parent_sid)
    except Exception as exc:  # noqa: BLE001
        logger.debug("trace_query: clickhouse backend unavailable: {}", exc)
        return None


def _local_trace_path(parent_sid: str, cwd: str) -> Path | None:
    path = resolve_observability_dir(cwd) / f"{parent_sid}.jsonl"
    if not path.is_file():
        return None
    return path


def _get_local_backend(path: Path | None) -> _TraceBackend | None:
    if path is None:
        return None
    return _LocalTraceBackend(path)


def _get_backend(parent_sid: str, cwd: str) -> _TraceBackend | None:
    local_path = _local_trace_path(parent_sid, cwd)
    return _get_clickhouse_backend(
        parent_sid, local_trace_path=local_path
    ) or _get_local_backend(local_path)


def install(session: Any, config: dict[str, Any]) -> None:
    del config
    runtime = _TraceQueryRuntime.from_api(session)
    if runtime is not None:
        runtime.install()


class _TraceQueryRuntime:
    def __init__(self, session: Any, *, backend: _TraceBackend) -> None:
        self._session = session
        self._backend = backend

    @classmethod
    def from_api(cls, session: Any) -> _TraceQueryRuntime | None:
        parent_sid = session.ctx.parent_session_id
        if parent_sid is None:
            logger.debug(
                "trace_query: no parent session — tools will return empty results"
            )
            return None

        backend = _get_backend(parent_sid, session.ctx.cwd)
        if backend is None:
            logger.warning(
                "trace_query: no ClickHouse or local parent trace — tools disabled"
            )
            return None

        return cls(session, backend=backend)

    def install(self) -> None:
        self._register_list_turns()
        self._register_read_turn()
        self._register_get_tool_calls()

    # -- list_turns --------------------------------------------------------

    def _register_list_turns(self) -> None:
        self._session.register_tool(
            FunctionTool(
                name="list_turns",
                description=(
                    "List turn-level summaries of the parent session's trajectory. "
                    "Shows which tools were called each turn with token counts. "
                    "Use this first to get an overview, then page through the "
                    "messages with read_turn."
                ),
                parameters=_ListTurnsArgs,
                fn=self.list_turns,
            )
        )

    async def list_turns(self, args: dict[str, Any]) -> ToolResult:
        parsed = _ListTurnsArgs.model_validate(args)
        records = self._backend.turns()
        return _text(self._render_turns(records, parsed.start, parsed.limit))

    def _render_turns(
        self, records: list[dict[str, Any]], start: int, limit: int
    ) -> str:
        total = len(records)
        sliced = records[start : start + limit]
        lines = [f"Parent session: {total} turns total"]
        for rec in sliced:
            idx = rec.get("turn_index", rec.get("turn_id", "?"))
            tools = rec.get("tool_calls", [])
            tool_str = ", ".join(tools) if tools else "—"
            tokens_in = rec.get("input_tokens", "?")
            tokens_out = rec.get("output_tokens", "?")
            lines.append(
                f"  [{idx}] tools=[{tool_str}] in={tokens_in} out={tokens_out}"
            )
        return "\n".join(lines)

    # -- read_turn ---------------------------------------------------------

    def _register_read_turn(self) -> None:
        self._session.register_tool(
            FunctionTool(
                name="read_turn",
                description=(
                    "Page through the parent session's messages. There is NO "
                    "turn selector — offset/limit paginate the flat message "
                    "sequence across the whole trajectory, optionally "
                    "filtered by role (user/assistant/tool_result/system). "
                    "Use this to inspect what the agent actually did and "
                    "what results it received. Blocks are clipped previews "
                    "with explicit truncation markers by default; pass "
                    "full=true (with a narrow window) for complete content."
                ),
                parameters=_ReadTurnArgs,
                fn=self.read_turn,
            )
        )

    async def read_turn(self, args: dict[str, Any]) -> ToolResult:
        parsed = _ReadTurnArgs.model_validate(args)
        roles = {parsed.role} if parsed.role else None
        records = self._backend.messages(roles=roles)
        return _text(
            self._render_messages(
                records, parsed.offset, parsed.limit, full=parsed.full,
            )
        )

    @staticmethod
    def _clip(text: str, limit: int, full: bool) -> str:
        if full or len(text) <= limit:
            return text
        return (
            text[:limit]
            + f"\n[truncated {len(text) - limit} chars — re-read this message "
            "with full=true and a narrow offset/limit window]"
        )

    def _render_messages(
        self,
        records: list[dict[str, Any]],
        offset: int,
        limit: int,
        *,
        full: bool = False,
    ) -> str:
        total = len(records)
        sliced = records[offset : offset + limit]
        parts = [
            f"Messages: {total} total (showing {offset}-{offset + len(sliced) - 1})"
        ]
        for rec in sliced:
            payload = rec.get("payload", {})
            role = payload.get("role", "?")
            content = payload.get("content", [])
            blocks: list[str] = []
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type", "")
                    if btype == "tool_call":
                        name = b.get("name", "")
                        a = json.dumps(b.get("arguments", {}), ensure_ascii=False)
                        blocks.append(f"[tool_call: {name}({self._clip(a, 500, full)})]")
                    elif btype == "tool_result":
                        sub = b.get("content", [])
                        txt = ""
                        if isinstance(sub, list):
                            for s in sub:
                                if isinstance(s, dict):
                                    txt += s.get("text", "")
                        err = " ERROR" if b.get("is_error") else ""
                        blocks.append(f"[tool_result{err}] {self._clip(txt, 1000, full)}")
                    elif b.get("text"):
                        blocks.append(self._clip(b["text"], 1500, full))
            body = "\n".join(blocks) if blocks else "(empty)"
            parts.append(f"[{role}]\n{body}\n")
        return "\n".join(parts)

    # -- get_tool_calls ----------------------------------------------------

    def _register_get_tool_calls(self) -> None:
        self._session.register_tool(
            FunctionTool(
                name="get_tool_calls",
                description=(
                    "Query tool calls from the parent session. "
                    "Optionally filter by exact tool name. Shows arguments "
                    "and result previews (first text block, capped at ~500 "
                    "chars — not full output). Use this to verify what "
                    "commands the agent actually ran and what they produced."
                ),
                parameters=_GetToolCallsArgs,
                fn=self.get_tool_calls,
            )
        )

    async def get_tool_calls(self, args: dict[str, Any]) -> ToolResult:
        parsed = _GetToolCallsArgs.model_validate(args)
        records = self._backend.tools()
        return _text(self._render_tool_calls(records, parsed.tool_name, parsed.limit))

    @staticmethod
    def _render_tool_calls(
        records: list[dict[str, Any]], tool_name: str | None, limit: int
    ) -> str:
        if tool_name:
            records = [r for r in records if r.get("tool") == tool_name]
        total = len(records)
        sliced = records[:limit]
        parts = [
            f"Tool calls: {total} total"
            + (f" (filter: {tool_name})" if tool_name else "")
        ]
        for rec in sliced:
            name = rec.get("tool", "?")
            args_data = rec.get("args")
            result_data = rec.get("result")
            args_str = json.dumps(args_data, ensure_ascii=False) if args_data else "{}"
            result_preview = ""
            if isinstance(result_data, dict):
                content = result_data.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("text"):
                            result_preview = c["text"][:500]
                            break
            parts.append(f"[{name}]\n  args: {args_str}\n  result: {result_preview}\n")
        return "\n".join(parts)


class _ListTurnsArgs(BaseModel):
    start: int = Field(default=0, description="Start turn index (inclusive)")
    limit: int = Field(default=50, description="Max turns to return")


class _ReadTurnArgs(BaseModel):
    role: str | None = Field(
        default=None,
        description="Filter by message role: user, assistant, tool_result, system",
    )
    limit: int = Field(default=20, description="Max messages to return")
    offset: int = Field(default=0, description="Skip this many messages")
    full: bool = Field(
        default=False,
        description=(
            "false (default): each content block is a preview clipped to a "
            "few hundred chars, with an explicit '[truncated N chars ...]' "
            "marker where content was cut. true: return blocks unclipped — "
            "combine with a narrow offset/limit window (e.g. limit=1) to "
            "read one message in full without flooding your context."
        ),
    )


class _GetToolCallsArgs(BaseModel):
    tool_name: str | None = Field(
        default=None,
        description="Filter by tool name (e.g. 'submit_final_report', 'query_sql')",
    )
    limit: int = Field(default=30, description="Max tool calls to return")


__all__: Final = ("MANIFEST", "install")
