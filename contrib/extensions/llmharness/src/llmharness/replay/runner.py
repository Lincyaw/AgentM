"""Rebuild extension lists from a recorded :class:`ReplayRecord` and run.

The compose-args captured in the record are the same kwargs the live adapter
passed to :func:`compose_extractor_extensions` /
:func:`compose_auditor_extensions`. Replay rebuilds the extension list
deterministically, mints a fresh ``ExtractionState`` (extractor only), and
calls :func:`run_phase_standalone` — so the resulting submit_* args are
directly comparable to ``record.output``.
"""

from __future__ import annotations

from typing import Any

from ..audit.auditor.extensions import compose_auditor_extensions
from ..audit.auditor.submit_tool import SUBMIT_VERDICT_TOOL_NAME
from ..audit.extractor.extensions import (
    EXTRACTOR_STATE_SERVICE_KEY,
    compose_extractor_extensions,
)
from ..audit.extractor.state import ExtractionState
from ..audit.extractor.tools import SUBMIT_EVENTS_TOOL_NAME
from ..schema import Edge, Event, Finding, Phase
from .engine import PhaseResult, run_phase_standalone
from .record import ReplayRecord

_EXTRACTOR_TOOLS_MODULE = "llmharness.audit.extractor.tools"
_SYSTEM_PROMPT_MODULE = "agentm.extensions.builtin.system_prompt"


def _coerce_provider(record: ReplayRecord) -> tuple[str, dict[str, Any]] | None:
    raw = record.provider
    if not raw:
        return None
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    module, cfg = raw[0], raw[1]
    if not isinstance(module, str):
        return None
    cfg_dict = cfg if isinstance(cfg, dict) else {}
    return module, dict(cfg_dict)


def _bind_extractor_state(
    base: list[tuple[str, dict[str, Any]]],
    *,
    state: ExtractionState,
    turn_window_json: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Mirror of the live adapter's ``_bind_extractor_state``.

    Injects a fresh ``ExtractionState`` into the extractor-tools atom config
    and substitutes ``{TURN_WINDOW_JSON}`` in the system prompt.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for module, cfg in base:
        new_cfg = dict(cfg)
        if module == _EXTRACTOR_TOOLS_MODULE:
            new_cfg["state"] = state
            new_cfg.setdefault(EXTRACTOR_STATE_SERVICE_KEY, state)
        elif module == _SYSTEM_PROMPT_MODULE:
            prompt = new_cfg.get("prompt")
            if isinstance(prompt, str) and "{TURN_WINDOW_JSON}" in prompt:
                new_cfg["prompt"] = prompt.replace(
                    "{TURN_WINDOW_JSON}", turn_window_json
                )
        out.append((module, new_cfg))
    return out


async def replay_extractor_record(
    record: ReplayRecord,
    *,
    cwd: str,
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override: str | None = None,
) -> PhaseResult:
    """Run extractor on a recorded payload.

    ``provider_override`` / ``prompt_override`` let callers swap model or
    system prompt while keeping the input payload + tool surface identical
    — the A/B knobs that motivate this whole module.
    """
    if record.phase != "extractor":
        raise ValueError(f"expected extractor record, got phase={record.phase!r}")

    compose_kwargs = dict(record.compose_kwargs)
    if prompt_override is not None:
        compose_kwargs["prompt_override"] = prompt_override

    turn_texts_raw = compose_kwargs.pop("turn_texts", None) or {}
    turn_window_json = compose_kwargs.pop("turn_window_json", "[]")

    base = compose_extractor_extensions(
        prompt_override=compose_kwargs.get("prompt_override"),
        cards_tools_config=compose_kwargs.get("cards_tools_config"),
        observability_config=compose_kwargs.get("observability_config"),
    )

    state = ExtractionState()
    # JSON loaded turn_texts has string keys; ExtractionState expects ints.
    for k, v in turn_texts_raw.items():
        try:
            state.turn_texts[int(k)] = str(v)
        except (TypeError, ValueError):
            continue

    extensions = _bind_extractor_state(
        base, state=state, turn_window_json=turn_window_json
    )

    provider = provider_override or _coerce_provider(record)
    return await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=record.payload,
        terminal_tool=SUBMIT_EVENTS_TOOL_NAME,
        purpose="cognitive_audit_extractor_replay",
    )


def _coerce_events(items: list[Any]) -> tuple[Event, ...]:
    out: list[Event] = []
    for it in items:
        if isinstance(it, dict):
            try:
                out.append(Event.from_dict(it))
            except Exception:
                continue
    return tuple(out)


def _coerce_edges(items: list[Any]) -> tuple[Edge, ...]:
    out: list[Edge] = []
    for it in items:
        if isinstance(it, dict):
            try:
                out.append(Edge.from_dict(it))
            except Exception:
                continue
    return tuple(out)


def _coerce_phases(items: list[Any]) -> tuple[Phase, ...]:
    out: list[Phase] = []
    for it in items:
        if isinstance(it, dict):
            try:
                out.append(Phase.from_dict(it))
            except Exception:
                continue
    return tuple(out)


def _coerce_findings(items: list[Any]) -> list[Finding]:
    out: list[Finding] = []
    for it in items:
        if isinstance(it, dict):
            try:
                out.append(Finding.from_dict(it))
            except Exception:
                continue
    return out


async def replay_auditor_record(
    record: ReplayRecord,
    *,
    cwd: str,
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override: str | None = None,
) -> PhaseResult:
    """Run auditor on a recorded graph + payload."""
    if record.phase != "auditor":
        raise ValueError(f"expected auditor record, got phase={record.phase!r}")

    ck = dict(record.compose_kwargs)
    if prompt_override is not None:
        ck["prompt_override"] = prompt_override

    extensions = compose_auditor_extensions(
        prompt_override=ck.get("prompt_override"),
        cards_tools_config=ck.get("cards_tools_config"),
        observability_config=ck.get("observability_config"),
        trajectory_snapshot=ck.get("trajectory_snapshot"),
        events=_coerce_events(ck.get("events") or []),
        edges=_coerce_edges(ck.get("edges") or []),
        phases=_coerce_phases(ck.get("phases") or []),
        findings=_coerce_findings(ck.get("findings") or []),
        check_errors=dict(ck.get("check_errors") or {}),
        continuation_notes=list(ck.get("continuation_notes") or []),
        summary_threshold=int(ck.get("summary_threshold", 30)),
    )

    provider = provider_override or _coerce_provider(record)
    return await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=record.payload,
        terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
        purpose="cognitive_audit_auditor_replay",
    )
