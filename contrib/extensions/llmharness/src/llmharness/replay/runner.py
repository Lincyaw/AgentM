"""Single-firing replay wrappers around :class:`HarnessRunner`.

These two functions exist for back-compat with chain replay, the
``llmharness-replay {extractor,auditor} --turn N`` CLI, and the RL
prompts exporter. Each wrapper is a thin shim: build a per-record
:class:`HarnessRunner` with no-op sink and no sidecar, then delegate to
``fire_extractor_from_record`` / ``fire_auditor_from_record``. That is
the design's "single-firing replay is a degenerate case" invariant —
the runner owns the body, the wrappers exist purely so callers don't
have to construct a runner themselves.
"""

from __future__ import annotations

from typing import Any

from ..audit.runner import (
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    HarnessRunner,
)
from ..audit.seams.offline import NoopSink, StandaloneChildRunner
from ..tools.engine import PhaseResult
from .record import ReplayRecord


def _coerce_provider(record: ReplayRecord) -> tuple[str, dict[str, Any]] | None:
    raw = record.provider
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    module, cfg = raw[0], raw[1]
    if not isinstance(module, str):
        return None
    return module, dict(cfg) if isinstance(cfg, dict) else {}


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
    identical — the A/B knobs that motivate this whole module.

    The v18 ``witness_retry_budget`` knob is gone in v19: each upsert
    gets per-edit validation feedback so there is no batch to bounce
    back. Replay sidecars from v18 that carried the knob in
    ``compose_kwargs`` are ignored — the new tool surface handles
    witness fixes via the three-section error template inside the
    handler.
    """
    if record.phase != "extractor":
        raise ValueError(f"expected extractor record, got phase={record.phase!r}")
    provider = provider_override or _coerce_provider(record)
    runner = HarnessRunner(
        cumulative=CumulativeAuditState.fresh(),
        child=StandaloneChildRunner(cwd),
        sink=NoopSink(),
        sidecar=None,
        extractor_settings=ExtractorSettings.from_compose_kwargs(
            record.compose_kwargs, prompt_override=prompt_override
        ),
        auditor_settings=AuditorSettings.empty(),
        extractor_interval=1,
        audit_interval=1,
        enable_auditor=False,
        session_id=record.session_id,
        trace_id=record.trace_id,
        provider_extractor=provider,
        provider_auditor=None,
        cwd=cwd,
    )
    return await runner.fire_extractor_from_record(record)


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
    provider = provider_override or _coerce_provider(record)
    runner = HarnessRunner(
        cumulative=CumulativeAuditState.fresh(),
        child=StandaloneChildRunner(cwd),
        sink=NoopSink(),
        sidecar=None,
        extractor_settings=ExtractorSettings(extensions=[], compose_kwargs={}),
        auditor_settings=AuditorSettings.from_compose_kwargs(
            record.compose_kwargs, prompt_override=prompt_override
        ),
        extractor_interval=1,
        audit_interval=1,
        enable_auditor=False,
        session_id=record.session_id,
        trace_id=record.trace_id,
        provider_extractor=None,
        provider_auditor=provider,
        cwd=cwd,
    )
    return await runner.fire_auditor_from_record(record)
