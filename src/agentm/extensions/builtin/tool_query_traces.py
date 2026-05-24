"""Tool atom: query observability JSONL traces by ``task_class``.

Returns a lightweight summary list (one entry per matching trace) so the
tuner agent can pick which traces to drill into via the regular ``read``
tool. See ``.claude/designs/per-task-evolution-loop.md`` §4.2.

This atom is the index; ``read`` is the fetch — same shape as the skills
catalog (compact entries up front, body on demand).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_query_traces",
    description=(
        "Query .agentm/observability/*.jsonl traces by task_class. "
        "Returns lightweight summaries; use the read tool for full bodies."
    ),
    registers=("tool:query_traces",),
)


_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "task_class": {
            "type": "string",
            "description": "Required task_class to filter by.",
        },
        "fingerprint": {
            "type": "object",
            "description": (
                "Optional exact-match filter against session.fingerprint "
                "atoms map. Each key is an atom name; value must equal the "
                "trace's fingerprint atom hash."
            ),
            "additionalProperties": {"type": "string"},
        },
        "n": {
            "type": "integer",
            "default": 50,
            "description": "Max number of most-recent traces to return.",
        },
        "include_eval_runs": {
            "type": "boolean",
            "default": True,
            "description": (
                "If False, exclude traces produced by tool_eval_run "
                "(those carry a non-null eval_run_id)."
            ),
        },
        "only_eval_runs": {
            "type": "boolean",
            "default": False,
            "description": (
                "If True, return ONLY eval-run traces. Mutually exclusive "
                "with include_eval_runs=False (only_eval_runs wins)."
            ),
        },
    },
    "required": ["task_class"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    cwd = Path(api.cwd)
    obs_dir = cwd / ".agentm/observability"

    async def _execute(args: dict[str, Any]) -> ToolResult:
        task_class = str(args["task_class"])
        fingerprint_filter = args.get("fingerprint") or None
        if fingerprint_filter is not None and not isinstance(
            fingerprint_filter, dict
        ):
            return _error(
                "fingerprint must be a mapping of atom_name -> hash"
            )
        n = int(args.get("n", 50))
        include_eval_runs = bool(args.get("include_eval_runs", True))
        only_eval_runs = bool(args.get("only_eval_runs", False))

        if not obs_dir.is_dir():
            return _ok(json.dumps({"traces": []}))

        # Sort by mtime descending so "most recent" semantics hold even when
        # the trace_id sort order would diverge.
        files = sorted(
            (p for p in obs_dir.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        out: list[dict[str, Any]] = []
        for path in files:
            summary = _summarize_trace(path)
            if summary is None:
                continue
            if summary.get("task_class") != task_class:
                continue
            has_eval_id = summary.get("eval_run_id") is not None
            if only_eval_runs and not has_eval_id:
                continue
            if not include_eval_runs and has_eval_id:
                continue
            if fingerprint_filter is not None:
                atoms = summary.get("fingerprint_atoms") or {}
                if not all(
                    atoms.get(k) == v for k, v in fingerprint_filter.items()
                ):
                    continue
            # Strip the heavy fingerprint_atoms field from the summary
            # we surface to the agent — it inflates the tool result and
            # the agent can re-fetch by reading the trace directly.
            surfaced = {k: v for k, v in summary.items() if k != "fingerprint_atoms"}
            out.append(surfaced)
            if len(out) >= n:
                break

        return _ok(json.dumps({"traces": out}, indent=2))

    api.register_tool(
        FunctionTool(
            name="query_traces",
            description=(
                "Filter .agentm/observability/*.jsonl traces by task_class "
                "and optional fingerprint. Returns summaries only — fetch "
                "full bodies via the `read` tool."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )

    del config


def _summarize_trace(path: Path) -> dict[str, Any] | None:
    """Walk an OTLP/JSON trace once and extract load-bearing identity
    fields. Returns ``None`` if the file isn't recognizably a session
    trace (no ``agentm.session.fingerprint`` or ``agentm.session.ready``
    log record was seen).

    Identity lives in log records; ``total_turns`` is derived by counting
    ``chat`` spans, and ``total_cost_usd`` is accumulated from any
    provider that stamps cost on the chat span (the framework's default
    writer does not emit cost).
    """
    trace_id: str | None = None
    timestamp_ns: int | None = None
    task_class: str | None = None
    eval_run_id: str | None = None
    task_id: str | None = None
    fingerprint_atoms: dict[str, str] = {}
    total_cost: float = 0.0
    total_turns: int = 0
    stop_reason: str | None = None
    seen_identity = False
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    line_dict = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(line_dict, dict):
                    continue
                for scope_logs in line_dict.get("scopeLogs", []) or []:
                    for record in scope_logs.get("logRecords", []) or []:
                        event_name = record.get("eventName")
                        body = _unwrap_otlp(record.get("body"))
                        if event_name == "agentm.session.fingerprint":
                            seen_identity = True
                            if isinstance(body, dict):
                                task_meta = body.get("task_meta") or {}
                                if isinstance(task_meta, dict):
                                    task_class = (
                                        task_meta.get("task_class") or task_class
                                    )
                                    eval_run_id = (
                                        task_meta.get("eval_run_id") or eval_run_id
                                    )
                                    task_id = task_meta.get("task_id") or task_id
                                atoms = body.get("atoms") or {}
                                if isinstance(atoms, dict):
                                    fingerprint_atoms = {
                                        str(k): str(v) for k, v in atoms.items()
                                    }
                            if timestamp_ns is None:
                                timestamp_ns = _parse_ns(record.get("timeUnixNano"))
                        elif event_name == "agentm.session.ready":
                            seen_identity = True
                            if timestamp_ns is None:
                                timestamp_ns = _parse_ns(record.get("timeUnixNano"))
                        elif event_name == "agentm.agent.end" and isinstance(
                            body, dict
                        ):
                            sr = body.get("stop_reason")
                            if isinstance(sr, str) and sr:
                                stop_reason = sr
                            cause = body.get("cause")
                            if isinstance(cause, dict) and not stop_reason:
                                cause_kind = cause.get("cause_kind")
                                if isinstance(cause_kind, str):
                                    stop_reason = cause_kind
                for scope_spans in line_dict.get("scopeSpans", []) or []:
                    for span in scope_spans.get("spans", []) or []:
                        name = span.get("name")
                        if not isinstance(name, str):
                            continue
                        if name.startswith("chat "):
                            total_turns += 1
                            for attr in span.get("attributes", []) or []:
                                if attr.get("key") in {
                                    "agentm.cost_usd",
                                    "gen_ai.usage.cost_usd",
                                }:
                                    v = attr.get("value") or {}
                                    if "doubleValue" in v:
                                        try:
                                            total_cost += float(v["doubleValue"])
                                        except (TypeError, ValueError):
                                            pass
                if trace_id is None:
                    trace_id = path.stem
    except OSError:
        return None
    if not seen_identity:
        return None
    return {
        "trace_id": trace_id,
        "path": str(path),
        "timestamp": _ns_to_iso(timestamp_ns),
        "task_class": task_class,
        "eval_run_id": eval_run_id,
        "task_id": task_id,
        "stop_reason": stop_reason,
        "total_cost_usd": round(total_cost, 6),
        "total_turns": total_turns,
        "fingerprint_atoms": fingerprint_atoms,
    }


def _parse_ns(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _ns_to_iso(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    import datetime as dt

    return dt.datetime.fromtimestamp(value / 1e9, tz=dt.timezone.utc).isoformat()


def _unwrap_otlp(value: Any) -> Any:
    """OTLP proto-JSON tagged-union unwrapper. Same shape as the helpers
    in ``tool_eval_run`` / ``tool_guard_watch``; duplicated because §11
    forbids atom-to-atom imports.
    """
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        try:
            return float(value["doubleValue"])
        except (TypeError, ValueError):
            return value["doubleValue"]
    if "kvlistValue" in value:
        out: dict[str, Any] = {}
        for item in value["kvlistValue"].get("values", []) or []:
            key = item.get("key")
            if not isinstance(key, str):
                continue
            out[key] = _unwrap_otlp(item.get("value"))
        return out
    if "arrayValue" in value:
        return [_unwrap_otlp(v) for v in value["arrayValue"].get("values", []) or []]
    return value


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
