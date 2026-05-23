"""Bulk replay across every record in a sidecar.

Single-phase replay (``llmharness-replay {extractor,auditor} --turn N``)
is the right tool when you have one suspicious turn. Chain replay is
the right tool when you want to ask "across all 325 auditor firings in
this run, what changes if I swap the prompt / model?" — it iterates in
turn order and yields one :class:`ChainResult` per matched record.

Threading semantics (changed in P2 — cf.
``.claude/designs/harness-runner.md`` §1.2): chain replay **does**
re-thread fresh extractor outputs back into the auditor's input graph.
Each extractor firing's freshly-produced ops are folded into a
:class:`CumulativeAuditState`, and each subsequent auditor firing sees
the threaded ``graph`` / ``recent_verdicts`` / ``continuation_notes``
view via cumulative state — not the recorded view. This makes chain
replay a faithful re-execution of the pipeline modulo the prompt /
model knobs the caller wanted to bisect on, instead of a bag of
independent per-record fire-and-forgets.

The bisection variable is still one-at-a-time (you change exactly one
of: extractor prompt, auditor prompt, provider), but cumulative state
is now derived from the chain-replayed outputs, not from the recorded
ones. To reproduce the legacy "no-re-thread" behaviour, replay records
individually via :func:`replay_extractor_record` /
:func:`replay_auditor_record`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..audit._runner import CumulativeAuditState
from ..audit.graph_ops import EdgeUpsert, NodeUpsert
from ..schema import Edge, Event, Verdict
from ..tools.engine import PhaseResult
from .record import Phase, ReplayRecord, iter_records
from .runner import replay_auditor_record, replay_extractor_record

PhaseFilter = Phase | Literal["both"]


@dataclass
class ChainResult:
    """One record's replay outcome paired with the recorded baseline."""

    record: ReplayRecord
    result: PhaseResult


def _thread_extractor_record(
    record: ReplayRecord, cumulative: CumulativeAuditState
) -> ReplayRecord:
    """Return a copy of ``record`` whose payload reflects ``cumulative``.

    Overrides ``recent_graph`` (from the cumulative event view) and
    ``next_event_id`` so the replayed extractor firing produces
    globally-consistent ids continuing the threaded counter. Other
    payload fields (``new_turns`` in particular) are preserved
    verbatim — the trajectory window is what the caller wanted to
    replay against.
    """
    events, _edges, _phases = cumulative.graph_view()
    new_payload = dict(record.payload)
    new_payload["recent_graph"] = [e.to_dict() for e in events]
    new_payload["next_event_id"] = cumulative.next_event_id()
    return replace(record, payload=new_payload)


def _thread_auditor_record(
    record: ReplayRecord, cumulative: CumulativeAuditState
) -> ReplayRecord:
    """Return a copy of ``record`` whose payload reflects ``cumulative``.

    Overrides ``graph`` / ``recent_verdicts`` /
    ``continuation_notes_from_prior_firing`` so the auditor firing
    sees the threaded state. ``compose_kwargs`` is untouched —
    :func:`replay_auditor_record` rebuilds the extension list from
    ``compose_kwargs`` (events/edges/phases/findings etc.), so for a
    faithful threading we would also need to override those.  This
    helper takes the more conservative line: thread the user-facing
    ``payload`` (what the LLM sees) but leave the compose-side
    framing as captured, matching what the caller bisected on.
    """
    new_payload = dict(record.payload)
    events, _edges, _phases = cumulative.graph_view()
    new_payload["graph"] = [e.to_dict() for e in events]
    new_payload["recent_verdicts"] = list(cumulative.recent_verdicts)
    new_payload["continuation_notes_from_prior_firing"] = list(
        cumulative.last_continuation_notes
    )
    return replace(record, payload=new_payload)


def _absorb_extractor_output(
    result: PhaseResult, cumulative: CumulativeAuditState
) -> None:
    """Fold a chain-replayed extractor output back into cumulative state.

    Mirrors :meth:`CumulativeAuditState.hydrate_from_session_log` 's
    legacy-AUDIT_EVENT / AUDIT_EDGE → :class:`NodeUpsert` /
    :class:`EdgeUpsert` translation: chain replay only has the
    high-level ``Event`` / ``Edge`` view to work with (the op-log is
    not in the sidecar), so we synthesise ops on the same shape the
    hydrate path uses. Best-effort: malformed entries are silently
    skipped — chain replay is a diagnostic tool, not a correctness
    gate.
    """
    if result.status != "ok" or not isinstance(result.output, dict):
        return
    ops: list[Any] = []
    for raw in result.output.get("events") or []:
        if not isinstance(raw, dict):
            continue
        try:
            ev = Event.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            continue
        ops.append(
            NodeUpsert(
                id=ev.id,
                kind=ev.kind.value,
                summary=ev.summary,
                source_turns=tuple(ev.source_turns),
                external_refs=ev.external_refs,
            )
        )
    for raw in result.output.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        try:
            ed = Edge.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            continue
        ops.append(
            EdgeUpsert(
                src=ed.src,
                dst=ed.dst,
                kind=ed.kind.value,
                reason=ed.reason,
                cited_entities=ed.cited_entities,
                cited_quote=ed.cited_quote,
                src_turns=ed.src_turns,
                dst_turns=ed.dst_turns,
            )
        )
    if not ops:
        return
    firing_id = cumulative.firing_id_counter
    cumulative.absorb_extractor_firing(
        firing_ops=ops,
        firing_cursor=cumulative.cursor_last_turn_index,
        firing_id=firing_id,
    )


def _absorb_auditor_output(
    result: PhaseResult, cumulative: CumulativeAuditState
) -> None:
    """Fold a chain-replayed auditor verdict back into cumulative state."""
    if result.status != "ok" or not isinstance(result.output, dict):
        return
    raw_verdict = result.output.get("verdict") or result.output
    if not isinstance(raw_verdict, dict):
        return
    try:
        verdict = Verdict.from_dict(raw_verdict)
    except (KeyError, TypeError, ValueError):
        return
    cumulative.absorb_auditor_verdict(
        verdict.to_dict(), is_silent=not verdict.surface_reminder
    )


async def chain_replay(
    records_path: Path,
    *,
    cwd: str,
    phase: PhaseFilter = "both",
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override_extractor: str | None = None,
    prompt_override_auditor: str | None = None,
    on_progress: Callable[[int, ChainResult], None] | None = None,
) -> AsyncIterator[ChainResult]:
    """Iterate records in file order; replay matching phases sequentially.

    Sequential by design — provider rate limits make parallel replay a
    footgun, and the typical use case (debug bisection) doesn't need
    it. Cumulative state is threaded across firings so the next
    auditor firing sees the chain-replayed extractor's fresh graph,
    not the recorded one (see module docstring).
    """
    cumulative = CumulativeAuditState.fresh()
    idx = 0
    for record in iter_records(records_path):
        if phase != "both" and record.phase != phase:
            continue
        if record.phase == "extractor":
            threaded = _thread_extractor_record(record, cumulative)
            result = await replay_extractor_record(
                threaded,
                cwd=cwd,
                provider_override=provider_override,
                prompt_override=prompt_override_extractor,
            )
            _absorb_extractor_output(result, cumulative)
        else:
            threaded = _thread_auditor_record(record, cumulative)
            result = await replay_auditor_record(
                threaded,
                cwd=cwd,
                provider_override=provider_override,
                prompt_override=prompt_override_auditor,
            )
            _absorb_auditor_output(result, cumulative)
        cr = ChainResult(record=record, result=result)
        if on_progress is not None:
            on_progress(idx, cr)
        idx += 1
        yield cr


def chain_replay_sync(
    records_path: Path,
    *,
    cwd: str,
    phase: PhaseFilter = "both",
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override_extractor: str | None = None,
    prompt_override_auditor: str | None = None,
    on_progress: Callable[[int, ChainResult], None] | None = None,
) -> list[ChainResult]:
    """Eager sync wrapper. Convenient for CLI / quick scripts."""

    async def _collect() -> list[ChainResult]:
        out: list[ChainResult] = []
        async for cr in chain_replay(
            records_path,
            cwd=cwd,
            phase=phase,
            provider_override=provider_override,
            prompt_override_extractor=prompt_override_extractor,
            prompt_override_auditor=prompt_override_auditor,
            on_progress=on_progress,
        ):
            out.append(cr)
        return out

    return asyncio.run(_collect())
