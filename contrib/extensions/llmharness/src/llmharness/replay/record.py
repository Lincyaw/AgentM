"""Replay-record I/O.

One JSONL line per phase invocation, written to
``<cwd>/.agentm/audit_replay/<session_id>.jsonl`` during live runs.
Each line is self-contained: it carries every input needed to rebuild the
extension list + payload, plus the recorded output for diffing.

Identity vocabulary mirrors AgentM core (``agentm.core.abi.extension``):
``session_id`` is the per-session id (= the OTel span_id of the
session-root span, and the ``.agentm/observability/<session_id>.jsonl``
filename) and is the sidecar/meta filename stem; ``trace_id`` is the
whole-tree group id (= core's ``api.root_session_id`` / the OTel
trace_id), shared by the root and all transitive children.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agentm.core.abi import ExtensionAPI

Phase = Literal["extractor", "auditor"]
Status = Literal["ok", "no_call", "spawn_error", "prompt_error"]

__all__ = [
    "Phase",
    "ReplayRecord",
    "Status",
    "audit_session_id",
    "iter_records",
    "now_ns",
    "read_records",
    "replay_log_path",
    "write_record",
]

@dataclass
class ReplayRecord:
    """One captured phase invocation.

    ``compose_kwargs`` carries the arguments handed to
    :func:`compose_extractor_extensions` / :func:`compose_auditor_extensions`.
    ``payload`` is the dict sent to ``child.prompt(json.dumps(payload))``.
    ``output`` is the parsed ``submit_*`` arguments (or ``None`` on failure).
    """

    phase: Phase
    turn_index: int
    session_id: str
    trace_id: str
    ts_ns: int
    compose_kwargs: dict[str, Any]
    payload: dict[str, Any]
    provider: list[Any] | None  # [module_str, config_dict] | None
    output: dict[str, Any] | None
    status: Status
    error: str | None = None
    latency_ms: int = 0
    extras: dict[str, Any] = field(default_factory=dict)
    raw_assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    """Serialized AssistantMessage content blocks from the child loop, in
    chronological order. Each entry is the JSON-friendly view of one block
    produced by ``adapters.agentm._serialize_block`` — thinking blocks keep
    ``{"type": "thinking", "text": "..."}``, tool_call blocks keep
    ``{"type": "tool_call", "name": ..., "arguments": {...}}``.

    Empty list when the child made no LLM call (e.g. spawn_error) or when
    the sidecar pre-dates this field. Only the assistant side is captured;
    user / tool_result messages are reconstructable from ``payload`` and
    ``output``.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict that round-trips via :meth:`from_dict`.

        All fields are emitted unconditionally (including ``extras`` and
        ``raw_assistant_messages``, even when empty) so callers stripping a
        subset for downstream consumers see a stable shape regardless of
        live-capture content. ``to_jsonl`` is the on-disk format and still
        omits empty optional fields to keep sidecars compact.
        """
        return {
            "phase": self.phase,
            "turn_index": self.turn_index,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "ts_ns": self.ts_ns,
            "compose_kwargs": self.compose_kwargs,
            "payload": self.payload,
            "provider": self.provider,
            "output": self.output,
            "status": self.status,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "extras": self.extras,
            "raw_assistant_messages": self.raw_assistant_messages,
        }

    def to_jsonl(self) -> str:
        d: dict[str, Any] = {
            "phase": self.phase,
            "turn_index": self.turn_index,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "ts_ns": self.ts_ns,
            "compose_kwargs": self.compose_kwargs,
            "payload": self.payload,
            "provider": self.provider,
            "output": self.output,
            "status": self.status,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }
        if self.extras:
            d["extras"] = self.extras
        if self.raw_assistant_messages:
            d["raw_assistant_messages"] = self.raw_assistant_messages
        return json.dumps(d, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReplayRecord:
        raw = d.get("raw_assistant_messages")
        raw_blocks: list[dict[str, Any]] = (
            [b for b in raw if isinstance(b, dict)] if isinstance(raw, list) else []
        )
        return cls(
            phase=d["phase"],
            turn_index=int(d["turn_index"]),
            session_id=str(d["session_id"]),
            trace_id=str(d["trace_id"]),
            ts_ns=int(d.get("ts_ns") or 0),
            compose_kwargs=dict(d.get("compose_kwargs") or {}),
            payload=dict(d.get("payload") or {}),
            provider=d.get("provider"),
            output=d.get("output"),
            status=d.get("status", "ok"),
            error=d.get("error"),
            latency_ms=int(d.get("latency_ms") or 0),
            extras=dict(d.get("extras") or {}),
            raw_assistant_messages=raw_blocks,
        )

def replay_log_path(cwd: str | os.PathLike[str], session_id: str) -> Path:
    """Canonical sidecar path for a given session.

    Keyed by ``session_id`` -- the per-session id that also names the
    ``.agentm/observability/<session_id>.jsonl`` file -- so the sidecar
    and its meta share a stem with the observability log.
    """
    return Path(cwd) / ".agentm" / "audit_replay" / f"{session_id}.jsonl"

def audit_session_id(api: ExtensionAPI) -> str:
    """The per-session id used to key sidecars: the persisted session_id,
    falling back to the OTel trace_id for in-memory sessions.

    Prefers ``api.session.get_session_id()`` (the persisted
    ``SessionManager`` header id, which is also the on-disk JSONL file
    name) so ``agent-from-reminder`` can relocate the source session.
    Falls back to ``api.root_session_id`` (the OTel trace_id assigned at
    session-construction time) when the session is in-memory /
    unpersisted -- that branch keeps the existing behaviour for embedded
    SDK callers who never write a session file at all.

    This is the canonical helper shared by every atom that keys a
    sidecar (``adapters.agentm``, ``distill.binding``); it lives in this
    lib module so atoms never import one another (§11).
    """
    sid: str
    try:
        sid = api.session.get_session_id()
    except AttributeError:
        sid = ""
    return sid or api.root_session_id

def write_record(path: Path, record: ReplayRecord) -> None:
    """Append one record; create the parent dir if missing.

    Best-effort: any IOError is logged and swallowed so a replay-log
    failure never breaks the live agent.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_jsonl())
            fh.write("\n")
    except OSError:
        import logging

        logging.getLogger(__name__).warning(
            "llmharness replay-log write failed: %s", path, exc_info=True
        )

def iter_records(path: Path) -> Iterator[ReplayRecord]:
    """Yield every record in a sidecar file. Skips malformed lines.

    Skips any topology-header line — those carry an experiment-header key
    (e.g. ``__fork_tree_header__``) and no ``phase`` key, so a fork-tree
    sidecar reads cleanly through this iterator. The ``phase`` check is
    header-key-agnostic on purpose: a missing ``phase`` is the single
    discriminator between a record line and a header line. See
    :func:`llmharness.replay.fork_tree.read_fork_tree_header` for the
    matching reader of that header.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "phase" not in obj:
                continue
            try:
                yield ReplayRecord.from_dict(obj)
            except (KeyError, ValueError):
                continue

def read_records(
    path: Path,
    *,
    phase: Phase | None = None,
    turn_index: int | None = None,
) -> list[ReplayRecord]:
    """Eager filter helper."""
    out = []
    for rec in iter_records(path):
        if phase is not None and rec.phase != phase:
            continue
        if turn_index is not None and rec.turn_index != turn_index:
            continue
        out.append(rec)
    return out

def now_ns() -> int:
    return time.time_ns()
