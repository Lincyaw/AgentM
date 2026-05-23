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

import logging
from typing import Any

from ..schema import Edge, Event, Phase, Verdict
from ..tools.engine import run_phase_standalone
from ._runner import (
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

        The four-step directive preamble the live ``LiveChildRunner``
        prepends to the payload lives, for the standalone path, inside
        :func:`run_phase_standalone` (it special-cases
        ``terminal_tool == "finalize_extraction"`` and prepends a
        recent-graph-aware preamble). Reuse that so the prompt-side
        contract stays in one place.

        Returns ``(terminator_called, raw_assistant_blocks)``. Spawn /
        prompt failures raise :class:`ExtractorSpawnError`; everything
        else (including ``no_call``) is signalled via ``terminator_called=False``.
        """
        del turn_window  # surfaced by the runner via append_failure context
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=payload,
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
    ) -> tuple[Verdict | None, list[dict[str, Any]]]:
        """Run one auditor firing as a top-level session.

        Mirrors :meth:`LiveChildRunner.run_auditor` — composes the
        ``{graph, recent_verdicts, continuation_notes_from_prior_firing}``
        payload and parses the terminal tool's args via
        :class:`RawVerdictOutput`. Returns ``(verdict | None, raw_blocks)``.

        Failure modes (spawn / prompt / no-call / malformed) all return
        ``(None, raw_blocks)`` — failure persistence is not this seam's
        responsibility (offline drivers route through ``InMemorySink``
        which has no diagnostic channel to emit on).
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

        if result.status != "ok" or result.output is None:
            return None, raw_blocks

        try:
            raw = RawVerdictOutput.from_dict(result.output)
            return raw.to_verdict(), raw_blocks
        except AuditorOutputError as exc:
            _logger.debug("offline auditor returned malformed verdict: %s", exc)
            return None, raw_blocks


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


__all__ = ["InMemorySink", "StandaloneChildRunner"]
