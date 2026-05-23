"""Offline seams for :class:`HarnessRunner`.

Symmetric to :mod:`llmharness.audit._live_seams` but for paths that drive
the audit pipeline without a parent :class:`ExtensionAPI`:

* :class:`StandaloneChildRunner` — spawns a top-level audit child via
  :func:`llmharness.tools.engine.run_phase_standalone` (no
  ``api.spawn_child_session``). Used by
  :mod:`llmharness.replay.offline_driver` (P2) and the planned thin
  wrapper around single-firing replay (P3).

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

from ..schema import Edge, Event, Phase
from ..tools.engine import run_phase_standalone
from ._extractor_directive import build_extractor_directive
from ._runner import (
    AuditorChildResult,
    ExtractorSpawnError,
    _flatten_assistant_blocks,
)
from .auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
)
from .extractor import FINALIZE_EXTRACTION_TOOL_NAME
from .graph_ops import GraphOp

_logger = logging.getLogger(__name__)


# --- child runner -----------------------------------------------------------


class StandaloneChildRunner:
    """:class:`ChildRunner` impl driven by :func:`run_phase_standalone`.

    Mirrors :class:`llmharness.audit._live_seams.LiveChildRunner` shape
    for shape — same return signatures, same failure routing — but
    spawns a top-level session per phase. There is no parent
    ``ExtensionAPI``; ``cwd`` is the working directory the child
    sessions execute in.
    """

    def __init__(self, cwd: str) -> None:
        self._cwd = cwd

    async def run_extractor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Run one extractor firing as a top-level session.

        The four-step directive preamble is built here via
        :func:`build_extractor_directive` and prepended to the JSON
        payload before being passed to :func:`run_phase_standalone` as
        a verbatim string. :func:`run_phase_standalone` no longer
        builds any preamble — both the live and offline extractor
        children now see the same byte-identical user message, which
        is what the design's invariant #1 (live === offline) requires.

        Returns ``(terminator_called, raw_assistant_blocks)``. Spawn /
        prompt failures raise :class:`ExtractorSpawnError`; everything
        else (including ``no_call``) is signalled via ``terminator_called=False``.
        """
        del turn_window  # surfaced by the runner via append_failure context
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        directive = build_extractor_directive(payload)
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=directive + payload_json,
            terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
            purpose="cognitive_audit_extractor_offline",
        )
        if result.status in ("spawn_error", "prompt_error"):
            raise ExtractorSpawnError(result.error or result.status)
        raw_blocks = _flatten_assistant_blocks(result.messages)
        return (result.status == "ok", raw_blocks)

    async def run_auditor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Event],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> AuditorChildResult:
        """Run one auditor firing as a top-level session.

        Mirrors :meth:`LiveChildRunner.run_auditor` — composes the
        ``{graph, recent_verdicts, continuation_notes_from_prior_firing}``
        payload and parses the terminal tool's args via
        :class:`RawVerdictOutput`. Returns an :class:`AuditorChildResult`
        with ``verdict=None`` for every failure mode (spawn / prompt /
        no-call / malformed); ``error`` / ``latency_ms`` are taken
        verbatim from :func:`run_phase_standalone`'s
        :class:`PhaseResult`. Failure persistence is not this seam's
        responsibility — offline drivers route through ``InMemorySink``
        which has no diagnostic channel to emit on.
        """
        payload: dict[str, Any] = {
            "graph": [e.to_dict() for e in graph_events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(
                continuation_notes_from_prior_firing
            ),
        }
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=payload,
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_auditor_offline",
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
        self.legacy_events: list[Event] = []
        self.legacy_edges: list[Edge] = []
        self.legacy_phases: list[Phase] = []

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

    def append_legacy_event(self, ev: Event) -> None:
        self.legacy_events.append(ev)

    def append_legacy_edge(self, ed: Edge) -> None:
        self.legacy_edges.append(ed)

    def append_legacy_phase(self, ph: Phase) -> None:
        self.legacy_phases.append(ph)


class NoopSink:
    """:class:`OpSink` impl that drops every persisted entry.

    Used by single-firing replay (:func:`replay_extractor_record` /
    :func:`replay_auditor_record`) where the runner is constructed
    purely to host one fire-from-record call. No session log, no
    in-memory inspection — the caller cares only about the returned
    :class:`PhaseResult`.
    """

    def append_op(
        self,
        op: GraphOp,
        *,
        firing_id: int,
        op_index: int,
        turn_window: list[int],
    ) -> None:
        return

    def append_cursor(self, *, last_turn_index: int) -> None:
        return

    def append_verdict(self, verdict: dict[str, Any]) -> None:
        return

    def append_failure(self, entry_type: str, payload: dict[str, Any]) -> None:
        return

    def append_partial(self, payload: dict[str, Any]) -> None:
        return

    def append_legacy_event(self, ev: Event) -> None:
        return

    def append_legacy_edge(self, ed: Edge) -> None:
        return

    def append_legacy_phase(self, ph: Phase) -> None:
        return


__all__ = ["InMemorySink", "NoopSink", "StandaloneChildRunner"]
