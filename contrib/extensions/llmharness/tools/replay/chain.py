"""Bulk replay across every record in a sidecar.

Single-phase replay (``llmharness-replay {extractor,auditor} --turn N``)
is the right tool when you have one suspicious turn. Chain replay is
the right tool when you want to ask "across all 325 auditor firings in
this run, what changes if I swap the prompt / model?" — it iterates in
turn order and yields one :class:`ChainResult` per matched record.

Threading semantics: chain replay re-threads fresh extractor outputs
back into the auditor's input graph. Each extractor firing's
freshly-produced ops are folded into a :class:`CumulativeAuditState`, and
each subsequent auditor firing sees the threaded ``graph`` /
``recent_verdicts`` / ``continuation_notes`` view via cumulative state —
not the recorded view. This makes chain replay a faithful re-execution
of the pipeline modulo the prompt / model knobs the caller wanted to
bisect on, instead of a bag of independent per-record fire-and-forgets.

The bisection variable is one-at-a-time (you change exactly one of:
extractor prompt, auditor prompt, provider); cumulative state is derived
from the chain-replayed outputs, not from the recorded ones. Replaying
single records individually via :func:`replay_extractor_record` /
:func:`replay_auditor_record` skips the re-threading.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from llmharness.agents.extractor.tools import parse_op
from llmharness.atom import CumulativeAuditState
from llmharness.schema import Verdict

from .engine import PhaseResult
from llmharness.replay.record import Phase, ReplayRecord, iter_records
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

    Overrides ``graph`` (plus legacy ``recent_graph`` / ``recent_edges``)
    and ``next_event_id`` so the replayed extractor firing sees the
    cumulative nodes+edges and produces globally-consistent ids continuing
    the threaded counter. Other payload fields (``new_turns`` in
    particular) are preserved verbatim — the trajectory window is what the
    caller wanted to replay against.
    """
    events, edges, _phases = cumulative.graph_view()
    new_payload = dict(record.payload)
    nodes_payload = [e.to_dict() for e in events]
    edges_payload = [ed.to_dict() for ed in edges]
    new_payload["graph"] = {"nodes": nodes_payload, "edges": edges_payload}
    new_payload["recent_graph"] = nodes_payload
    new_payload["recent_edges"] = edges_payload
    new_payload["next_event_id"] = cumulative.next_event_id()
    return replace(record, payload=new_payload)


def _thread_auditor_record(record: ReplayRecord, cumulative: CumulativeAuditState) -> ReplayRecord:
    """Return a copy of ``record`` whose payload AND compose_kwargs reflect ``cumulative``.

    :func:`replay_auditor_record` rebuilds the auditor's installed
    extensions from ``compose_kwargs`` (events / edges / phases /
    continuation_notes / findings). If we threaded only the payload,
    the LLM would see the threaded graph in the user message but the
    installed extension context (cards, continuation notes, findings)
    would still be whatever was captured at record time — half state
    is worse than no state.

    So override both layers: ``payload.graph`` /
    ``payload.recent_verdicts`` /
    ``payload.continuation_notes_from_prior_firing`` AND
    ``compose_kwargs.events`` / ``compose_kwargs.edges`` /
    ``compose_kwargs.phases`` / ``compose_kwargs.continuation_notes``.

    Limitation: chain replay does NOT re-run audit-check registries,
    so ``compose_kwargs.findings`` and ``compose_kwargs.check_errors``
    are cleared to empty — they are produced live by ``_drain_auditor``
    against a scenario-installed registry that is not available in
    chain replay. The replayed auditor therefore sees the same graph
    the live auditor would have seen at this firing point, modulo
    audit-check findings.
    """
    new_payload = dict(record.payload)
    events, edges, phases = cumulative.graph_view()
    new_payload["graph"] = [e.to_dict() for e in events]
    new_payload["recent_verdicts"] = list(cumulative.recent_verdicts)
    new_payload["continuation_notes_from_prior_firing"] = list(cumulative.last_continuation_notes)

    new_compose = dict(record.compose_kwargs)
    new_compose["events"] = [e.to_dict() for e in events]
    new_compose["edges"] = [ed.to_dict() for ed in edges]
    new_compose["phases"] = [ph.to_dict() for ph in phases]
    new_compose["continuation_notes"] = list(cumulative.last_continuation_notes)
    # Audit-check registry is scenario-installed and chain replay does
    # not re-execute it; leave findings / check_errors empty rather
    # than carry stale captured values.
    new_compose["findings"] = []
    new_compose["check_errors"] = {}

    return replace(record, payload=new_payload, compose_kwargs=new_compose)


def _absorb_extractor_output(
    result: PhaseResult, cumulative: CumulativeAuditState, *, firing_cursor: int
) -> None:
    """Fold a chain-replayed extractor output back into cumulative state.

    Reads ``output.ops`` (the v4 op log) and replays them through
    :meth:`CumulativeAuditState.absorb_extractor_firing`. Malformed ops
    are silently skipped — chain replay is a diagnostic tool, not a
    correctness gate.
    """
    if result.status != "ok" or not isinstance(result.output, dict):
        return
    raw_ops = result.output.get("ops") or result.output.get("graph_ops") or []
    ops: list[Any] = []
    for raw in raw_ops:
        if not isinstance(raw, dict):
            continue
        try:
            ops.append(parse_op(raw))
        except (KeyError, TypeError, ValueError):
            continue
    if not ops:
        return
    firing_id = cumulative.firing_id_counter
    cumulative.absorb_extractor_firing(
        firing_ops=ops,
        firing_cursor=firing_cursor,
        firing_id=firing_id,
    )


def _absorb_auditor_output(result: PhaseResult, cumulative: CumulativeAuditState) -> None:
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
    cumulative.absorb_auditor_verdict(verdict.to_dict())


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
            _absorb_extractor_output(result, cumulative, firing_cursor=record.turn_index)
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
