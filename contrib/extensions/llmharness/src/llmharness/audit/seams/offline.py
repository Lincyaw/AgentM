"""Offline seams for :class:`HarnessRunner`.

Symmetric to :mod:`llmharness.audit.seams.live` but for paths that drive
the audit pipeline without a parent :class:`ExtensionAPI`:

* :class:`StandaloneChildRunner` — spawns a top-level audit child via
  :func:`llmharness.tools.engine.run_phase_standalone` (no
  ``api.spawn_child_session``). Used by
  :mod:`llmharness.replay.offline_driver` for full-trajectory offline
  replay. (Single-firing replay calls ``run_phase_standalone`` directly
  in :mod:`llmharness.replay.runner` and does not use this seam.)

* :class:`InMemorySink` — drops every persisted entry into a Python list
  instead of the AgentM session log. Used by offline drivers and tests
  that want the runner's full behaviour but no on-disk session state.

Neither seam emits ``DiagnosticEvent`` s; that channel only exists on the
live ExtensionAPI.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...agents.auditor.output import AuditorOutputError, RawVerdictOutput
from ...agents.auditor.submit_verdict import SUBMIT_VERDICT_TOOL_NAME
from ...agents.extractor.extractor_tools import FINALIZE_EXTRACTION_TOOL_NAME
from ...agents.extractor.state import ExtractionState
from ...schema import Event
from ...tools.engine import run_phase_standalone
from ..graph.ops import GraphOp
from ..runner import (
    AuditorChildResult,
    ExtractorSpawnError,
    _flatten_assistant_blocks,
)
from ..toolkit.extractor_directive import build_extractor_directive

_logger = logging.getLogger(__name__)


# --- child runner -----------------------------------------------------------


class StandaloneChildRunner:
    """:class:`ChildRunner` impl driven by :func:`run_phase_standalone`.

    Accepts domain-level parameters (state, prompt text, tool config)
    from the runner and composes extension lists internally for replay
    compatibility. Same return signatures and failure routing as
    :class:`llmharness.audit.seams.live.LiveChildRunner`, but spawns a
    top-level session per phase. There is no parent ``ExtensionAPI``;
    ``cwd`` is the working directory the child sessions execute in.
    """

    def __init__(
        self,
        cwd: str,
        *,
        parent_session_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._cwd = cwd
        self._parent_session_id = parent_session_id
        self._trace_id = trace_id

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
        """Run one extractor firing as a top-level session.

        Composes the extension list internally from domain-level params,
        then delegates to :func:`run_phase_standalone`.

        Returns ``(terminator_called, raw_assistant_blocks)``. Spawn /
        prompt failures raise :class:`ExtractorSpawnError`; everything
        else (including ``no_call``) is signalled via ``terminator_called=False``.
        """
        del turn_window  # surfaced by the runner via append_failure context
        _EXT_TOOLS = "llmharness.agents.extractor.extractor_tools"
        _OBS = "agentm.extensions.builtin.observability"
        _OPS = "agentm.extensions.builtin.operations"
        _SYS = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_OBS, {}), (_OPS, {}),
            (_EXT_TOOLS, {"state": state, "llmharness.extractor_state": state}),
            (_SYS, {"prompt": prompt_text}),
        ]
        if tool_call_budget is not None:
            budget = int(tool_call_budget)
            extensions.extend([
                ("agentm.extensions.builtin.loop_budget", {"max_tool_calls": budget}),
                ("agentm.extensions.builtin.turn_reminder", {"warn_within": budget}),
            ])

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        directive = build_extractor_directive(payload)
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=directive + payload_json,
            terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
            purpose="cognitive_audit_extractor_offline",
            parent_session_id=self._parent_session_id,
            trace_id=self._trace_id,
        )
        if result.status in ("spawn_error", "prompt_error"):
            raise ExtractorSpawnError(result.error or result.status)
        raw_blocks = _flatten_assistant_blocks(result.messages)
        return (result.status == "ok", raw_blocks)

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
        """Run one auditor firing as a top-level session.

        Composes the extension list internally from domain-level params
        (``prompt_text`` and ``tools_config``), then delegates to
        :func:`run_phase_standalone`. Returns an :class:`AuditorChildResult`
        with ``verdict=None`` for every failure mode (spawn / prompt /
        no-call / malformed); ``error`` / ``latency_ms`` are taken
        verbatim from :func:`run_phase_standalone`'s
        :class:`PhaseResult`. Failure persistence is not this seam's
        responsibility — offline drivers route through ``InMemorySink``
        which has no diagnostic channel to emit on.
        """
        _AUD_TOOLS = "llmharness.agents.auditor.auditor_tools"
        _OBS = "agentm.extensions.builtin.observability"
        _OPS = "agentm.extensions.builtin.operations"
        _SYS = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_OBS, {}), (_OPS, {}),
            (_AUD_TOOLS, dict(tools_config)),
            (_SYS, {"prompt": prompt_text}),
        ]

        payload: dict[str, Any] = {
            "graph": [e.to_dict() for e in graph_events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing),
        }
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=payload,
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_auditor_offline",
            parent_session_id=self._parent_session_id,
            trace_id=self._trace_id,
        )
        raw_blocks = _flatten_assistant_blocks(result.messages)
        latency_ms = result.latency_ms

        if result.status != "ok" or result.output is None:
            return AuditorChildResult(
                verdict=None,
                raw_blocks=raw_blocks,
                error=result.error,
                latency_ms=latency_ms,
            )

        try:
            raw = RawVerdictOutput.from_dict(result.output)
            return AuditorChildResult(
                verdict=raw.to_verdict(),
                raw_blocks=raw_blocks,
                error=None,
                latency_ms=latency_ms,
            )
        except AuditorOutputError as exc:
            err = f"malformed: {exc}"
            _logger.debug("offline auditor returned malformed verdict: %s", exc)
            return AuditorChildResult(
                verdict=None,
                raw_blocks=raw_blocks,
                error=err,
                latency_ms=latency_ms,
            )


# --- op sink ----------------------------------------------------------------


class InMemorySink:
    """:class:`OpSink` impl that captures every entry in local lists.

    Designed for offline drivers and tests: every ``append_*`` call just
    grows the corresponding list. There is no session log, no
    diagnostic channel, no on-disk persistence. The cumulative state
    held by the runner is authoritative; this sink is for inspection
    only.

    All lists are public so callers (tests in particular) can assert on
    counts and contents directly.
    """

    def __init__(self) -> None:
        self.ops: list[tuple[GraphOp, int, int, list[int]]] = []
        """``(op, firing_id, op_index, turn_window)`` per :meth:`append_op` call."""

        self.cursors: list[int] = []
        """One ``last_turn_index`` per :meth:`append_cursor` call."""

        self.verdicts: list[dict[str, Any]] = []
        self.failures: list[tuple[str, dict[str, Any]]] = []
        self.partials: list[dict[str, Any]] = []

    def append_op(
        self,
        op: GraphOp,
        *,
        firing_id: int,
        op_index: int,
        turn_window: list[int],
    ) -> None:
        self.ops.append((op, firing_id, op_index, list(turn_window)))

    def append_cursor(self, *, last_turn_index: int) -> None:
        self.cursors.append(last_turn_index)

    def append_verdict(self, verdict: dict[str, Any]) -> None:
        self.verdicts.append(dict(verdict))

    def append_failure(self, entry_type: str, payload: dict[str, Any]) -> None:
        self.failures.append((entry_type, dict(payload)))

    def append_partial(self, payload: dict[str, Any]) -> None:
        self.partials.append(dict(payload))


__all__ = ["InMemorySink", "StandaloneChildRunner"]
