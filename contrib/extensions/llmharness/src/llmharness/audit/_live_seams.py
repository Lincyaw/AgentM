"""Live seams for :class:`HarnessRunner`.

Wraps a parent :class:`ExtensionAPI` so the runner's :class:`ChildRunner`
and :class:`OpSink` protocols can be satisfied against the live AgentM
session. Bodies of :class:`LiveChildRunner` are lifted verbatim from the
legacy ``_spawn_extractor_child`` / ``_run_auditor`` in
``adapters/agentm.py``; :class:`LiveOpSink` is a thin wrapper over
``api.session.append_entry`` + a :class:`DiagnosticEvent` emit on
failures (matching ``_record_failure``).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Final, Literal

from agentm.core.abi.events import DiagnosticEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolCallBlock
from agentm.core.abi.session_config import AgentSessionConfig

from ..schema import Edge, Event, Phase, Verdict
from . import entry_types as _et
from ._runner import (
    ExtractorSpawnError,
    _flatten_assistant_blocks,
)
from ._session_helpers import find_terminal_tool_arguments, safe_shutdown
from .auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
)
from .extractor import FINALIZE_EXTRACTION_TOOL_NAME
from .graph_ops import GraphOp

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

        Body lifted verbatim from the legacy ``_spawn_extractor_child``.
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
            recent_n = len(payload.get("recent_graph") or [])
            next_id = payload.get("next_event_id")
            directive = (
                "Below is the firing input. Workflow:\n"
                "(1) Build the graph incrementally with upsert_node / "
                "upsert_edge (and delete_node / delete_edge as needed). Every "
                "edit is validated immediately; errors come back as a "
                "three-section actionable message naming concrete next "
                "options. Internal events must be true branch points "
                "(in-degree>=2 or out-degree>=2); passthrough (in=1, out=1) "
                "events are rejected at finalize.\n"
                "(2) Call finalize_extraction (no payload) when you are done. "
                "On a clean degree check the firing terminates; on rejection "
                "the firing stays alive so you can promote passthrough nodes "
                "with further upserts and re-call.\n"
                f"(3) Start event ids at {next_id} and increment strictly — "
                "do NOT restart at 1 and do NOT reuse any id from recent_graph.\n"
                f"(4) Cross-firing references: recent_graph has {recent_n} "
                "entries. To link this firing's events to prior firings, emit "
                "upsert_edge with src/dst spanning the boundary — the folded "
                "view already contains prior-firing nodes by id. Most evid "
                "events in this firing answer a hyp/act from earlier firings; "
                "linking them is what turns a single firing into a connected "
                "investigation.\n\n"
                + payload_json
            )
            messages = await child.prompt(directive)
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
    ) -> tuple[Verdict | None, list[dict[str, Any]]]:
        """Run one auditor firing; mirror of the legacy ``_run_auditor``.

        ``None`` return covers spawn / prompt / no-call / malformed paths.
        Failures are recorded via ``api.session.append_entry`` + a
        :class:`DiagnosticEvent`, matching the legacy ``_record_failure``
        chokepoint.
        """
        payload = {
            "graph": [e.to_dict() for e in graph_events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(
                continuation_notes_from_prior_firing
            ),
        }
        child_config = AgentSessionConfig(
            cwd=self._api.cwd,
            provider=provider,
            extensions=extensions,
            purpose="cognitive_audit_auditor",
        )
        try:
            child = await self._api.spawn_child_session(child_config)
        except Exception as exc:
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": str(exc)})
            return None, []

        try:
            messages = await child.prompt(
                json.dumps(payload, ensure_ascii=False, default=str)
            )
        except Exception as exc:
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": str(exc)})
            await safe_shutdown(child)
            return None, []

        await safe_shutdown(child)

        raw_blocks = _flatten_assistant_blocks(messages)

        arguments = find_terminal_tool_arguments(messages, SUBMIT_VERDICT_TOOL_NAME)
        if arguments is None:
            _record_failure(
                self._api,
                _et.AUDIT_NO_CALL,
                {
                    "reason": (
                        f"child returned without calling {SUBMIT_VERDICT_TOOL_NAME}"
                    )
                },
            )
            return None, raw_blocks

        try:
            raw = RawVerdictOutput.from_dict(arguments)
        except AuditorOutputError as exc:
            _record_failure(
                self._api, _et.AUDIT_ERROR, {"reason": f"malformed: {exc}"}
            )
            return None, raw_blocks

        try:
            return raw.to_verdict(), raw_blocks
        except AuditorOutputError as exc:
            _record_failure(
                self._api, _et.AUDIT_ERROR, {"reason": f"malformed: {exc}"}
            )
            return None, raw_blocks


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
        # Legacy ``_drain_extractor`` wrote EXTRACTOR_PARTIAL with a plain
        # ``append_entry`` — no DiagnosticEvent. Keep that exact shape.
        self._api.session.append_entry(_et.EXTRACTOR_PARTIAL, payload)

    def append_legacy_event(self, ev: Event) -> None:
        self._api.session.append_entry(_et.AUDIT_EVENT, ev.to_dict())

    def append_legacy_edge(self, ed: Edge) -> None:
        self._api.session.append_entry(_et.AUDIT_EDGE, ed.to_dict())

    def append_legacy_phase(self, ph: Phase) -> None:
        self._api.session.append_entry(_et.AUDIT_PHASE, ph.to_dict())


def _record_failure(
    api: ExtensionAPI, entry_type: str, payload: dict[str, Any]
) -> None:
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
        _logger.exception(
            "llmharness audit diagnostic emit failed; suppressing."
        )


__all__ = ["LiveChildRunner", "LiveOpSink"]
