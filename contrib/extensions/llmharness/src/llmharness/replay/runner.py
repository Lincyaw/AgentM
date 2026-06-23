"""Single-firing replay: re-run one recorded extractor/auditor firing.

These functions back chain replay, dev-checkout replay tooling, and the RL
prompts exporter. A replay record already carries a finished
``payload`` + ``compose_kwargs``, so a firing needs none of the live
machinery (cadence, cumulative state, sinks): rebuild the per-firing
inputs from the record and call :func:`run_phase_standalone` directly.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, Literal

from llmharness.agents.auditor.context import build_auditor_system_prompt
from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.agents.extractor.tools import (
    FINALIZE_EXTRACTION_TOOL_NAME,
    ExtractionState,
)
from llmharness.context_index import build_context_index
from llmharness.replay.record import ReplayRecord
from llmharness.schema import Edge, Event, Finding, Phase

from .engine import PhaseResult, run_phase_standalone

# ---------------------------------------------------------------------------
# Settings dataclasses (replace old runtime.runner Settings)
# ---------------------------------------------------------------------------

ContextMode = Literal["graph", "index", "both"]


def _coerce_context_mode(raw: Any) -> ContextMode:
    if raw == "graph":
        return "graph"
    if raw == "both":
        return "both"
    return "index"


class ExtractorSettings:
    """Minimal config needed to replay an extractor firing."""

    def __init__(
        self,
        *,
        base_prompt: str | None = None,
        tool_call_budget: int | None = None,
        compose_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_prompt = base_prompt
        self.tool_call_budget = tool_call_budget
        self.compose_kwargs = compose_kwargs or {}

    @classmethod
    def from_compose_kwargs(
        cls,
        compose_kwargs: dict[str, Any],
        *,
        prompt_override: str | None = None,
    ) -> ExtractorSettings:
        from llmharness.agents.extractor.context import load_extractor_prompt

        base_prompt = prompt_override
        if base_prompt is None:
            prompt_name = compose_kwargs.get("prompt_name") or "default"
            base_prompt = load_extractor_prompt(prompt_name)
        return cls(
            base_prompt=base_prompt,
            tool_call_budget=compose_kwargs.get("tool_call_budget"),
            compose_kwargs=compose_kwargs,
        )

    @classmethod
    def default(cls) -> ExtractorSettings:
        from llmharness.agents.extractor.context import load_extractor_prompt

        return cls(base_prompt=load_extractor_prompt("default"))


class AuditorSettings:
    """Minimal config needed to replay an auditor firing."""

    def __init__(
        self,
        *,
        base_prompt: str | None = None,
        summary_threshold: int = 30,
        context_mode: ContextMode = "index",
        tools: tuple[str, ...] | None = None,
        observability_config: dict[str, Any] | None = None,
    ) -> None:
        self.base_prompt = base_prompt
        self.summary_threshold = summary_threshold
        self.context_mode = context_mode
        self.tools = tools
        self.observability_config = observability_config

    @classmethod
    def from_compose_kwargs(
        cls,
        compose_kwargs: dict[str, Any],
        *,
        prompt_override: str | None = None,
    ) -> AuditorSettings:
        from llmharness.agents.auditor.context import load_auditor_prompt

        base_prompt = prompt_override
        if base_prompt is None:
            prompt_name = compose_kwargs.get("prompt_name") or "minimal_index"
            base_prompt = load_auditor_prompt(prompt_name)
        tools_raw = compose_kwargs.get("tools")
        tools = tuple(tools_raw) if isinstance(tools_raw, (list, tuple)) else None
        context_mode = _coerce_context_mode(compose_kwargs.get("context_mode"))
        return cls(
            base_prompt=base_prompt,
            summary_threshold=int(compose_kwargs.get("summary_threshold", 30)),
            context_mode=context_mode,
            tools=tools,
            observability_config=compose_kwargs.get("observability_config"),
        )

    @classmethod
    def default(cls) -> AuditorSettings:
        from llmharness.agents.auditor.context import load_auditor_prompt

        return cls(base_prompt=load_auditor_prompt("minimal_index"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_provider(record: ReplayRecord) -> tuple[str, dict[str, Any]] | None:
    raw = record.provider
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    module, cfg = raw[0], raw[1]
    if not isinstance(module, str):
        return None
    return module, dict(cfg) if isinstance(cfg, dict) else {}


def _coerce_trajectory(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


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


# ---------------------------------------------------------------------------
# Extractor replay
# ---------------------------------------------------------------------------


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
    ``extras.turn_texts``, binds it onto the extractor extension list,
    runs the firing, and snapshots results from the bound state.
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
    for k, v in (extras.get("prior_turn_texts") or {}).items():
        with contextlib.suppress(TypeError, ValueError):
            state.turn_texts.setdefault(int(k), str(v))

    # Enrich recent index records and populate state.
    payload = dict(record.payload or {})
    graph_obj = payload.get("graph")
    graph_raw: dict[str, Any] = graph_obj if isinstance(graph_obj, dict) else {}
    recent_records_raw = (
        payload.get("recent_records")
        or payload.get("records")
        or graph_raw.get("nodes")
        or payload.get("recent_graph")
        or []
    )
    recent_links_raw = (
        payload.get("recent_links")
        or payload.get("links")
        or graph_raw.get("edges")
        or payload.get("recent_edges")
        or []
    )
    enriched_recent: list[dict[str, Any]] = []
    recent_events: list[Event] = []
    for entry in recent_records_raw:
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
    payload["records"] = enriched_recent
    payload["links"] = list(recent_links_raw)
    payload["recent_records"] = enriched_recent
    payload["recent_links"] = list(recent_links_raw)
    tool_call_budget = settings.compose_kwargs.get("tool_call_budget")
    if isinstance(tool_call_budget, int) and tool_call_budget > 0:
        payload["tool_call_budget"] = tool_call_budget
    state.recent_records = tuple(recent_events)
    state.recent_record_dict = {e.id: e for e in recent_events}

    recent_edges: list[Edge] = []
    for entry in recent_links_raw:
        if not isinstance(entry, dict):
            continue
        try:
            recent_edges.append(Edge.from_dict(entry))
        except (KeyError, ValueError, TypeError):
            continue
    state.recent_links = tuple(recent_edges)
    state.recent_link_dict = {(ed.src, ed.dst, ed.kind.value): ed for ed in recent_edges}
    state._refold()

    _EXT_TOOLS = "llmharness.agents.extractor.tools"
    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"
    _SYS = "agentm.extensions.builtin.system_prompt"
    extensions: list[tuple[str, dict[str, Any]]] = [
        (_OBS, {}), (_OPS, {}),
        (_EXT_TOOLS, {"state": state, "llmharness.extractor_state": state}),
        (_SYS, {"prompt": settings.base_prompt}),
    ]
    if settings.tool_call_budget is not None:
        budget = int(settings.tool_call_budget)
        extensions.extend([
            ("agentm.extensions.builtin.loop_budget", {"max_tool_calls": budget}),
            ("agentm.extensions.builtin.turn_reminder", {"warn_within": budget}),
        ])

    result = await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=json.dumps(payload, ensure_ascii=False, default=str),
        terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
        purpose="cognitive_audit_extractor_replay",
    )
    # Salvage: commit-on-stop if the extractor didn't finalize.
    state.salvage()
    if state.pending_ops:
        result = PhaseResult(
            output={
                "events": [e.to_dict() for e in state.events],
                "edges": [ed.to_dict() for ed in state.edges],
                "dropped_edges": list(state.dropped_edges),
                "ops": [op.to_dict() for op in state.pending_ops],
            },
            status="ok",
            error=result.error,
            latency_ms=result.latency_ms,
            messages=result.messages,
        )
    return result


# ---------------------------------------------------------------------------
# Auditor replay
# ---------------------------------------------------------------------------


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
    events_t = tuple(_coerce_schema_list(Event, ck.get("events") or []))
    edges_t = tuple(_coerce_schema_list(Edge, ck.get("edges") or []))
    phases_t = tuple(_coerce_schema_list(Phase, ck.get("phases") or []))
    raw_context_index = ck.get("context_index")
    context_index = dict(raw_context_index) if isinstance(raw_context_index, dict) else None
    if context_index is None:
        trajectory = _coerce_trajectory(ck.get("trajectory_snapshot"))
        if trajectory:
            context_index = build_context_index(
                trajectory=trajectory,
                events=events_t,
                edges=edges_t,
            ).to_dict()
    prompt_text = build_auditor_system_prompt(
        events=events_t,
        edges=edges_t,
        phases=phases_t,
        findings=_coerce_schema_list(Finding, ck.get("findings") or []),
        check_errors=dict(ck.get("check_errors") or {}),
        continuation_notes=list(ck.get("continuation_notes") or []),
        summary_threshold=settings.summary_threshold,
        base_prompt=settings.base_prompt or None,
        context_index=context_index,
        context_mode=settings.context_mode,
    )
    tools_config: dict[str, Any] = {"tools": list(settings.tools or (SUBMIT_VERDICT_TOOL_NAME,))}
    _AUDITOR_TOOLS = "llmharness.agents.auditor.tools"
    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"
    _SYS = "agentm.extensions.builtin.system_prompt"
    extensions: list[tuple[str, dict[str, Any]]] = []
    obs_cfg = settings.observability_config
    if obs_cfg is not None:
        extensions.append((_OBS, dict(obs_cfg)))
    extensions.append((_OPS, {}))
    extensions.append((_AUDITOR_TOOLS, dict(tools_config)))
    extensions.append((_SYS, {"prompt": prompt_text}))
    return await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=record.payload or {},
        terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
        purpose="cognitive_audit_auditor_replay",
    )
