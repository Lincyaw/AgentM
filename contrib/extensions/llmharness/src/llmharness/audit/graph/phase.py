"""Mechanical phase merger over raw extractor events.

Raw events are extracted one-per-turn (issue #134 §1: each AgentM turn
is a ReAct step). For long trajectories that produces dozens of
``act`` events the auditor must reason over. The phase merger
collapses consecutive ``act`` runs into a single "basic block"-style
:class:`~llmharness.schema.Phase` so the auditor sees high-level
investigation phases instead of every tool call.

Boundaries:

- ``task`` / ``hyp`` / ``dec`` / ``concl`` always become singleton
  phases. They are decision / framing points and never merge.
- A maximal run of consecutive ``act`` events between two break-kind
  events becomes one merged phase with kind ``"act_run"`` (or the
  singleton kind if the run has length 1).
- Phase ids are fresh-numbered per firing starting at 1, matching the
  raw event id namespace convention.

The merger is **purely deterministic** — no LLM input. Given the same
event list it always produces the same phase list.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...schema import Event, EventKind, Phase

# Event kinds that always force a singleton phase. They are decision /
# framing points and we never want their summary fused into a
# neighbouring run.
_BREAK_KINDS: frozenset[EventKind] = frozenset(
    {EventKind.TASK, EventKind.HYP, EventKind.DEC, EventKind.CONCL}
)
# Event kinds that are eligible to be merged into a multi-event run.
_RUN_KINDS: frozenset[EventKind] = frozenset({EventKind.ACT})

# Soft cap on the joined ``summary`` field of a merged-run phase so the
# auditor prompt does not grow without bound on long investigation
# tracks. Raw events stay verbatim on disk; the auditor can drill down
# via ``get_event_detail`` for the full text.
_MAX_RUN_SUMMARY_CHARS = 1200
_RUN_SUMMARY_SEPARATOR = " | "
_RUN_KIND_LABEL = "act_run"


def merge_to_phases(events: Sequence[Event]) -> list[Phase]:
    """Walk ``events`` in registration order and collapse runs into phases.

    Empty input yields the empty list. Order of phases follows the
    order of their first member event.
    """

    phases: list[Phase] = []
    next_id = 1
    run_buffer: list[Event] = []

    def _flush_run() -> None:
        nonlocal next_id
        if not run_buffer:
            return
        if len(run_buffer) == 1:
            ev = run_buffer[0]
            phases.append(
                Phase(
                    id=next_id,
                    kind=ev.kind.value,
                    member_event_ids=(ev.id,),
                    source_turns=tuple(sorted(set(ev.source_turns))),
                    summary=ev.summary,
                )
            )
        else:
            ids = tuple(e.id for e in run_buffer)
            turns = tuple(sorted({t for e in run_buffer for t in e.source_turns}))
            joined = _RUN_SUMMARY_SEPARATOR.join(e.summary for e in run_buffer)
            if len(joined) > _MAX_RUN_SUMMARY_CHARS:
                joined = joined[: _MAX_RUN_SUMMARY_CHARS - 3] + "..."
            phases.append(
                Phase(
                    id=next_id,
                    kind=_RUN_KIND_LABEL,
                    member_event_ids=ids,
                    source_turns=turns,
                    summary=joined,
                )
            )
        next_id += 1
        run_buffer.clear()

    for ev in events:
        if ev.kind in _BREAK_KINDS:
            _flush_run()
            phases.append(
                Phase(
                    id=next_id,
                    kind=ev.kind.value,
                    member_event_ids=(ev.id,),
                    source_turns=tuple(sorted(set(ev.source_turns))),
                    summary=ev.summary,
                )
            )
            next_id += 1
        elif ev.kind in _RUN_KINDS:
            run_buffer.append(ev)
        else:  # pragma: no cover — defensive; EventKind is exhaustive
            _flush_run()

    _flush_run()
    return phases


__all__ = ["merge_to_phases"]
