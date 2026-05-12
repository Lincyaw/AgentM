"""Rebuild extension lists from a recorded :class:`ReplayRecord` and run.

The compose-args captured in the record are the same kwargs the live adapter
passed to :func:`compose_extractor_extensions` /
:func:`compose_auditor_extensions`. Replay rebuilds the extension list
deterministically, mints a fresh ``ExtractionState`` (extractor only), and
calls :func:`run_phase_standalone` — so the resulting submit_* args are
directly comparable to ``record.output``.
"""

from __future__ import annotations

import contextlib
from typing import Any, TypeVar

from ..audit._session_helpers import bind_extractor_state
from ..audit.auditor.extensions import compose_auditor_extensions
from ..audit.auditor.submit_tool import SUBMIT_VERDICT_TOOL_NAME
from ..audit.extractor.extensions import compose_extractor_extensions
from ..audit.extractor.state import ExtractionState
from ..audit.extractor.tools import SUBMIT_EVENTS_TOOL_NAME
from ..schema import Edge, Event, Finding, Phase
from .engine import PhaseResult, run_phase_standalone
from .record import ReplayRecord

_T = TypeVar("_T")


def _coerce_provider(record: ReplayRecord) -> tuple[str, dict[str, Any]] | None:
    raw = record.provider
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    module, cfg = raw[0], raw[1]
    if not isinstance(module, str):
        return None
    return module, dict(cfg) if isinstance(cfg, dict) else {}


def _coerce_list(cls: type[_T], items: list[Any]) -> list[_T]:
    """Apply ``cls.from_dict`` to every dict-shaped item; skip malformed.

    The schema's ``from_dict`` raises ``KeyError`` / ``TypeError`` /
    ``ValueError`` on missing or wrong-typed fields — narrowing the
    except clause to those would be more correct, but the broad swallow
    matches the rest of llmharness's "best-effort on read" pattern.
    """
    out: list[_T] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        with contextlib.suppress(KeyError, TypeError, ValueError):
            out.append(cls.from_dict(it))  # type: ignore[attr-defined]
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

    ck = record.compose_kwargs
    extras = record.extras
    # Support both the new ``base_prompt`` key and the legacy
    # ``prompt_override`` key from pre-profile replay sidecars.
    effective_prompt = (
        prompt_override
        if prompt_override is not None
        else ck.get("base_prompt") or ck.get("prompt_override")
    )

    base = compose_extractor_extensions(
        base_prompt=effective_prompt,
        cards_tools_config=ck.get("cards_tools_config"),
        observability_config=ck.get("observability_config"),
    )

    state = ExtractionState()
    # JSON loaded turn_texts has string keys; ExtractionState expects ints.
    for k, v in (extras.get("turn_texts") or {}).items():
        with contextlib.suppress(TypeError, ValueError):
            state.turn_texts[int(k)] = str(v)

    extensions = bind_extractor_state(
        base, state=state, turn_window_json=extras.get("turn_window_json", "[]")
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

    ck = record.compose_kwargs
    effective_prompt = (
        prompt_override
        if prompt_override is not None
        else ck.get("base_prompt") or ck.get("prompt_override")
    )

    tools_raw = ck.get("tools")
    tools_tuple = (
        tuple(str(t) for t in tools_raw) if isinstance(tools_raw, list) else None
    )

    extensions = compose_auditor_extensions(
        base_prompt=effective_prompt,
        cards_tools_config=ck.get("cards_tools_config"),
        observability_config=ck.get("observability_config"),
        trajectory_snapshot=ck.get("trajectory_snapshot"),
        events=tuple(_coerce_list(Event, ck.get("events") or [])),
        edges=tuple(_coerce_list(Edge, ck.get("edges") or [])),
        phases=tuple(_coerce_list(Phase, ck.get("phases") or [])),
        findings=_coerce_list(Finding, ck.get("findings") or []),
        check_errors=dict(ck.get("check_errors") or {}),
        continuation_notes=list(ck.get("continuation_notes") or []),
        summary_threshold=int(ck.get("summary_threshold", 30)),
        tools=tools_tuple,
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
