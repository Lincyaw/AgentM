"""Live seams for :class:`HarnessRunner`.

Wraps a parent :class:`ExtensionAPI` so the runner's :class:`ChildRunner`
and :class:`OpSink` protocols can be satisfied against the live AgentM
session. :class:`LiveChildRunner` spawns extractor / auditor children
via ``api.spawn_child_session`` using declarative scenario manifests
(``AgentSessionConfig(scenario=...)``); :class:`LiveOpSink` is a thin
wrapper over ``api.session.append_entry`` plus a :class:`DiagnosticEvent`
emit on failures.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Final, Literal

from agentm.core.abi.events import DiagnosticEvent
from agentm.core.abi.extension import ExtensionAPI

from ...agents import auditor_scenario, extractor_scenario
from ...agents.auditor.output import AuditorOutputError, RawVerdictOutput
from ...agents.auditor.submit_verdict import SUBMIT_VERDICT_TOOL_NAME
from ...agents.extractor.extractor_tools import FINALIZE_EXTRACTION_TOOL_NAME
from ...agents.extractor.state import ExtractionState
from ...child_collect import flatten_assistant_blocks
from ...child_task import run_child_task
from ...schema import Event
from .. import entry_types as _et
from ..graph.ops import GraphOp
from ..runner import (
    AuditorChildResult,
    ExtractorSpawnError,
)
from ..toolkit.extractor_directive import build_extractor_directive

_logger = logging.getLogger(__name__)

_FAILURE_DIAGNOSTIC_LEVEL: Final[Literal["warning"]] = "warning"


# --- child runner -----------------------------------------------------------


class LiveChildRunner:
    """:class:`ChildRunner` impl that spawns children via ``api.spawn_child_session``.

    Uses declarative agent manifests (scenario-based spawning) with
    ``atom_config_overrides`` and ``extra_extensions`` to configure
    per-firing state, prompt text, and optional budget atoms.
    """

    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api
        self._extractor_scenario = extractor_scenario()
        self._auditor_scenario = auditor_scenario()

    async def run_extractor(
        self,
        *,
        state: ExtractionState,
        prompt_text: str,
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
        tool_call_budget: int | None = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Spawn the extractor child via scenario manifest.

        Configures the child via ``atom_config_overrides`` for the
        extractor_tools state and system_prompt text. Optional budget
        atoms are appended via ``extra_extensions``.

        Returns ``(terminator_called, raw_blocks)``. Spawn / prompt
        failures raise :class:`ExtractorSpawnError` so the runner can
        route them to ``EXTRACTOR_ERROR`` + sidecar.
        """
        del turn_window  # surfaced by the runner via append_failure context
        overrides: dict[str, dict[str, Any]] = {
            "extractor_tools": {"state": state},
            "system_prompt": {"prompt": prompt_text},
        }
        extra: list[tuple[str, dict[str, Any]]] = []
        if tool_call_budget is not None:
            budget = int(tool_call_budget)
            extra.extend([
                ("agentm.extensions.builtin.loop_budget", {"max_tool_calls": budget}),
                ("agentm.extensions.builtin.turn_reminder", {"warn_within": budget}),
            ])

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        directive = build_extractor_directive(payload)
        result = await run_child_task(
            self._api,
            scenario=self._extractor_scenario,
            atom_config_overrides=overrides,
            extra_extensions=extra,
            provider=provider,
            prompt=directive + payload_json,
            purpose="cognitive_audit_extractor",
            terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
        )
        if result.error is not None:
            raise ExtractorSpawnError(result.error)
        return (result.terminal_called, flatten_assistant_blocks(result.messages))

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Event],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> AuditorChildResult:
        """Run one auditor firing via scenario manifest.

        Configures the child via ``atom_config_overrides`` for the
        auditor_tools config and system_prompt text.

        Returns an :class:`AuditorChildResult` with ``verdict=None`` for
        spawn / prompt / no-call / malformed paths. Failures are recorded
        via ``api.session.append_entry`` plus a :class:`DiagnosticEvent`.
        ``latency_ms`` / ``error`` are propagated so the runner can
        surface them on the synthetic auditor record.
        """
        overrides: dict[str, dict[str, Any]] = {
            "auditor_tools": tools_config,
            "system_prompt": {"prompt": prompt_text},
        }
        payload = {
            "graph": [e.to_dict() for e in graph_events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing),
        }
        result = await run_child_task(
            self._api,
            scenario=self._auditor_scenario,
            atom_config_overrides=overrides,
            provider=provider,
            prompt=json.dumps(payload, ensure_ascii=False, default=str),
            purpose="cognitive_audit_auditor",
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
        )
        latency_ms = result.latency_ms

        if result.error is not None:
            err = result.error
            _record_failure(self._api, _et.AUDIT_ERROR, {"reason": err})
            return AuditorChildResult(verdict=None, raw_blocks=[], error=err, latency_ms=latency_ms)

        raw_blocks = flatten_assistant_blocks(result.messages)

        arguments = result.terminal_args
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


# --- op sink ----------------------------------------------------------------


class LiveOpSink:
    """:class:`OpSink` impl that appends to the live session log.

    All entry-type literals come from :mod:`llmharness.audit.entry_types`;
    failure entries route through :func:`_record_failure` so a
    :class:`DiagnosticEvent` is co-emitted.
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
    """Persist a typed failure entry on the branch and emit a diagnostic.

    Appends the entry AND emits a :class:`DiagnosticEvent` so the failure
    shows up in the OTel jsonl. Never raises.
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
