"""Live seams for :class:`HarnessRunner`.

Wraps a parent :class:`ExtensionAPI` so the runner's :class:`ChildRunner`
and :class:`OpSink` protocols can be satisfied against the live AgentM
session. :class:`LiveChildRunner` spawns extractor / auditor children
via ``api.spawn_child_session``; :class:`LiveOpSink` is a thin wrapper
over ``api.session.append_entry`` plus a :class:`DiagnosticEvent` emit
on failures.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Final, Literal

from agentm.core.abi.events import DiagnosticEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolCallBlock
from agentm.core.abi.session_config import AgentSessionConfig

from ...schema import Event
from .. import entry_types as _et
from ..auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
)
from ..extractor import FINALIZE_EXTRACTION_TOOL_NAME
from ..graph.ops import GraphOp
from ..runner import (
    AuditorChildResult,
    ExtractorSpawnError,
    _flatten_assistant_blocks,
)
from ..toolkit.extractor_directive import build_extractor_directive
from .session import find_terminal_tool_arguments, safe_shutdown

_logger = logging.getLogger(__name__)

_FAILURE_DIAGNOSTIC_LEVEL: Final[Literal["warning"]] = "warning"


# --- child runner -----------------------------------------------------------


class LiveChildRunner:
    """:class:`ChildRunner` impl that spawns children via ``api.spawn_child_session``."""

    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api

    async def run_extractor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Spawn the extractor child, drive it, return ``(terminator_called, raw_blocks)``.

        Spawn / prompt failures raise :class:`ExtractorSpawnError` so the
        runner can route them to ``EXTRACTOR_ERROR`` + sidecar.
        """
        del turn_window  # surfaced by the runner via append_failure context
        child_config = AgentSessionConfig(
            cwd=self._api.cwd,
            provider=provider,
            extensions=extensions,
            purpose="cognitive_audit_extractor",
        )
        try:
            child = await self._api.spawn_child_session(child_config)
        except Exception as exc:
            raise ExtractorSpawnError(str(exc)) from exc

        try:
            payload_json = json.dumps(payload, ensure_ascii=False, default=str)
            directive = build_extractor_directive(payload)
            messages = await child.prompt(directive + payload_json)
        except Exception as exc:
            await safe_shutdown(child)
            raise ExtractorSpawnError(str(exc)) from exc

        await safe_shutdown(child)
        return (
            _has_tool_call(messages, FINALIZE_EXTRACTION_TOOL_NAME),
            _flatten_assistant_blocks(messages),
        )

    async def run_auditor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Event],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> AuditorChildResult:
        """Run one auditor firing.

        Returns an :class:`AuditorChildResult` with ``verdict=None`` for
        spawn / prompt / no-call / malformed paths. Failures are recorded
        via ``api.session.append_entry`` plus a :class:`DiagnosticEvent`.
        ``latency_ms`` / ``error`` are propagated so the runner can
        surface them on the synthetic auditor record.
        """
        payload = {
            "graph": [e.to_dict() for e in graph_events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing),
        }
        child_config = AgentSessionConfig(
            cwd=self._api.cwd,
            provider=provider,
            extensions=extensions,
            purpose="cognitive_audit_auditor",
        )
        t0 = time.monotonic()

        def _elapsed_ms() -> int:
            return int((time.monotonic() - t0) * 1000)

        try:
            child = await self._api.spawn_child_session(child_config)
        except Exception as exc:
            err = str(exc)
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": err})
            return AuditorChildResult(
                verdict=None, raw_blocks=[], error=err, latency_ms=_elapsed_ms()
            )

        try:
            messages = await child.prompt(json.dumps(payload, ensure_ascii=False, default=str))
        except Exception as exc:
            err = str(exc)
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": err})
            await safe_shutdown(child)
            return AuditorChildResult(
                verdict=None, raw_blocks=[], error=err, latency_ms=_elapsed_ms()
            )

        await safe_shutdown(child)

        raw_blocks = _flatten_assistant_blocks(messages)
        latency_ms = _elapsed_ms()

        arguments = find_terminal_tool_arguments(messages, SUBMIT_VERDICT_TOOL_NAME)
        if arguments is None:
            reason = f"child returned without calling {SUBMIT_VERDICT_TOOL_NAME}"
            _record_failure(self._api, _et.AUDIT_NO_CALL, {"reason": reason})
            return AuditorChildResult(
                verdict=None,
                raw_blocks=raw_blocks,
                error=reason,
                latency_ms=latency_ms,
            )

        try:
            raw = RawVerdictOutput.from_dict(arguments)
        except AuditorOutputError as exc:
            err = f"malformed: {exc}"
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": err})
            return AuditorChildResult(
                verdict=None,
                raw_blocks=raw_blocks,
                error=err,
                latency_ms=latency_ms,
            )

        try:
            return AuditorChildResult(
                verdict=raw.to_verdict(),
                raw_blocks=raw_blocks,
                error=None,
                latency_ms=latency_ms,
            )
        except AuditorOutputError as exc:
            err = f"malformed: {exc}"
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": err})
            return AuditorChildResult(
                verdict=None,
                raw_blocks=raw_blocks,
                error=err,
                latency_ms=latency_ms,
            )


def _has_tool_call(messages: list[AgentMessage], tool_name: str) -> bool:
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return True
    return False


# --- op sink ----------------------------------------------------------------


class LiveOpSink:
    """:class:`OpSink` impl that appends to the live session log.

    All entry-type literals come from :mod:`llmharness.audit.entry_types`;
    failure entries route through :func:`_record_failure` so a
    :class:`DiagnosticEvent` is co-emitted (matches the legacy
    ``_record_failure`` chokepoint).
    """

    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api

    def append_op(
        self,
        op: GraphOp,
        *,
        firing_id: int,
        op_index: int,
        turn_window: list[int],
    ) -> None:
        payload = op.to_dict()
        payload["firing_id"] = firing_id
        payload["op_index"] = op_index
        payload["caused_by_turn_window"] = list(turn_window)
        self._api.session.append_entry(_et.AUDIT_GRAPH_OP, payload)

    def append_cursor(self, *, last_turn_index: int) -> None:
        self._api.session.append_entry(
            _et.EXTRACTOR_CURSOR,
            {
                "last_turn_index": last_turn_index,
                "extraction_run_id": uuid.uuid4().hex,
            },
        )

    def append_verdict(self, verdict: dict[str, Any]) -> None:
        self._api.session.append_entry(_et.VERDICT, verdict)

    def append_failure(self, entry_type: str, payload: dict[str, Any]) -> None:
        _record_failure(self._api, entry_type, payload)

    def append_partial(self, payload: dict[str, Any]) -> None:
        self._api.session.append_entry(_et.EXTRACTOR_PARTIAL, payload)


def _record_failure(api: ExtensionAPI, entry_type: str, payload: dict[str, Any]) -> None:
    """Mirror of the legacy ``adapters.agentm._record_failure`` chokepoint.

    Persists a typed failure entry on the branch AND emits a
    :class:`DiagnosticEvent` so the failure shows up in the OTel jsonl.
    Never raises.
    """
    api.session.append_entry(entry_type, payload)
    reason_raw = payload.get("reason") if isinstance(payload, dict) else None
    reason = str(reason_raw) if reason_raw is not None else ""
    message = f"{entry_type}: {reason}" if reason else entry_type
    try:
        api.events.emit_sync(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level=_FAILURE_DIAGNOSTIC_LEVEL,
                source="llmharness.audit",
                message=message,
            ),
        )
    except Exception:
        _logger.exception("llmharness audit diagnostic emit failed; suppressing.")


__all__ = ["LiveChildRunner", "LiveOpSink"]
