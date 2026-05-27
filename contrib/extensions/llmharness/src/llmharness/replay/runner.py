"""Single-firing replay: re-run one recorded extractor/auditor firing.

These functions back chain replay, the
``llmharness-replay {extractor,auditor} --turn N`` CLI, and the RL
prompts exporter. A replay record already carries a finished
``payload`` + ``compose_kwargs``, so a firing needs none of the live
``HarnessRunner`` machinery (cadence, cumulative state, sinks): rebuild
the per-firing inputs from the record and call
:func:`run_phase_standalone` directly.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from ..audit.auditor import SUBMIT_VERDICT_TOOL_NAME, compose_auditor_extensions
from ..audit.extractor import (
    FINALIZE_EXTRACTION_TOOL_NAME,
    ExtractionState,
    RawExtractorOutput,
)
from ..audit.runner import AuditorSettings, ExtractorSettings
from ..audit.seams.session import bind_extractor_state
from ..audit.toolkit.extractor_directive import build_extractor_directive
from ..schema import Edge, Event, Finding, Phase
from ..tools.engine import PhaseResult, run_phase_standalone
from .record import ReplayRecord


def _coerce_provider(record: ReplayRecord) -> tuple[str, dict[str, Any]] | None:
    raw = record.provider
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    module, cfg = raw[0], raw[1]
    if not isinstance(module, str):
        return None
    return module, dict(cfg) if isinstance(cfg, dict) else {}


def _coerce_schema_list(cls: Any, items: Any) -> list[Any]:
    """Best-effort dict-to-dataclass coercion for replay-record schema fields."""
    out: list[Any] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(cls.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def replay_extractor_record(
    record: ReplayRecord,
    *,
    cwd: str,
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override: str | None = None,
) -> PhaseResult:
    """Run extractor on a recorded payload.

    ``provider_override`` / ``prompt_override`` let callers swap model
    or system prompt while keeping the input payload + tool surface
    identical — the A/B knobs that motivate this module.

    Rebuilds an :class:`ExtractionState` from ``record.payload`` +
    ``extras.turn_texts`` (so external_refs can be witnessed against
    trajectory text), binds it onto the extractor extension list, runs
    the firing, and snapshots :class:`RawExtractorOutput` from the bound
    state. The returned ``events`` / ``edges`` / ``dropped_edges`` /
    ``ops`` shape is pinned by tests.
    """
    if record.phase != "extractor":
        raise ValueError(f"expected extractor record, got phase={record.phase!r}")
    provider = provider_override or _coerce_provider(record)
    settings = ExtractorSettings.from_compose_kwargs(
        record.compose_kwargs, prompt_override=prompt_override
    )

    state = ExtractionState()
    nxt = (record.payload or {}).get("next_event_id")
    if isinstance(nxt, int) and nxt >= 1:
        state.next_event_id = nxt

    # JSON-loaded turn_texts has string keys; ExtractionState wants ints.
    extras = record.extras or {}
    for k, v in (extras.get("turn_texts") or {}).items():
        with contextlib.suppress(TypeError, ValueError):
            state.turn_texts[int(k)] = str(v)
    # Union in any prior-firing turn texts the caller supplied (the CLI
    # computes these from earlier records so external_refs can be
    # witnessed; the live adapter does the same enrichment at firing time).
    for k, v in (extras.get("prior_turn_texts") or {}).items():
        with contextlib.suppress(TypeError, ValueError):
            state.turn_texts.setdefault(int(k), str(v))

    # Enrich recent_graph entries with source_turn_texts and populate
    # state.recent_graph so external_refs can be witnessed.
    payload = dict(record.payload or {})
    graph_obj = payload.get("graph")
    graph_raw: dict[str, Any] = graph_obj if isinstance(graph_obj, dict) else {}
    recent_graph_raw = graph_raw.get("nodes") or payload.get("recent_graph") or []
    recent_edges_raw = graph_raw.get("edges") or payload.get("recent_edges") or []
    enriched_recent: list[dict[str, Any]] = []
    recent_events: list[Event] = []
    for entry in recent_graph_raw:
        if not isinstance(entry, dict):
            continue
        enriched = dict(entry)
        enriched["source_turn_texts"] = [
            state.turn_texts.get(int(t), "")
            for t in (entry.get("source_turns") or [])
            if isinstance(t, int)
        ]
        enriched_recent.append(enriched)
        try:
            recent_events.append(Event.from_dict(entry))
        except (KeyError, ValueError, TypeError):
            continue
    payload["graph"] = {"nodes": enriched_recent, "edges": list(recent_edges_raw)}
    payload["recent_graph"] = enriched_recent
    payload["recent_edges"] = list(recent_edges_raw)
    tool_call_budget = settings.compose_kwargs.get("tool_call_budget")
    if isinstance(tool_call_budget, int) and tool_call_budget > 0:
        payload["tool_call_budget"] = tool_call_budget
    state.recent_graph = tuple(recent_events)
    state.recent_graph_dict = {e.id: e for e in recent_events}

    recent_edges: list[Edge] = []
    for entry in recent_edges_raw:
        if not isinstance(entry, dict):
            continue
        try:
            recent_edges.append(Edge.from_dict(entry))
        except (KeyError, ValueError, TypeError):
            continue
    state.recent_edges_dict = {(ed.src, ed.dst, ed.kind.value): ed for ed in recent_edges}
    state._refold()

    extensions = bind_extractor_state(settings.extensions, state=state)

    result = await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=build_extractor_directive(payload)
        + json.dumps(payload, ensure_ascii=False, default=str),
        terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
        purpose="cognitive_audit_extractor_replay",
    )
    if result.status == "ok":
        snapshot = RawExtractorOutput.from_state(state)
        result = PhaseResult(
            output={
                "events": [e.to_dict() for e in snapshot.events],
                "edges": [ed.to_dict() for ed in snapshot.edges],
                "dropped_edges": list(snapshot.dropped_edges),
                "ops": [op.to_dict() for op in state.pending_ops],
            },
            status=result.status,
            error=result.error,
            latency_ms=result.latency_ms,
            messages=result.messages,
        )
    return result


async def replay_auditor_record(
    record: ReplayRecord,
    *,
    cwd: str,
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override: str | None = None,
) -> PhaseResult:
    """Run auditor on a recorded graph + payload.

    Composes auditor extensions from ``record.compose_kwargs`` (events /
    edges / phases / findings / continuation_notes / tools) and passes
    ``record.payload`` to the child as the user message verbatim.
    """
    if record.phase != "auditor":
        raise ValueError(f"expected auditor record, got phase={record.phase!r}")
    provider = provider_override or _coerce_provider(record)
    settings = AuditorSettings.from_compose_kwargs(
        record.compose_kwargs, prompt_override=prompt_override
    )

    ck = record.compose_kwargs or {}
    extensions = compose_auditor_extensions(
        base_prompt=settings.base_prompt or None,
        observability_config=settings.observability_config,
        trajectory_snapshot=ck.get("trajectory_snapshot"),
        events=tuple(_coerce_schema_list(Event, ck.get("events") or [])),
        edges=tuple(_coerce_schema_list(Edge, ck.get("edges") or [])),
        phases=tuple(_coerce_schema_list(Phase, ck.get("phases") or [])),
        findings=_coerce_schema_list(Finding, ck.get("findings") or []),
        check_errors=dict(ck.get("check_errors") or {}),
        continuation_notes=list(ck.get("continuation_notes") or []),
        summary_threshold=settings.summary_threshold,
        tools=settings.tools or None,
    )
    return await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=record.payload or {},
        terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
        purpose="cognitive_audit_auditor_replay",
    )
