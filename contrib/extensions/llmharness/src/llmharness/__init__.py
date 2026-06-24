"""LLM-as-harness: cognitive-audit AgentM extension."""

from __future__ import annotations

from typing import Any

from .eval.replay.offline_driver import (
    OfflineRunResult,
    SurfaceFiring,
    replay_pipeline_over_trajectory,
)
from .schema import (
    Reminder,
    Verdict,
)

# Backward-compat aliases from the deleted root offline.py
SurfacePoint = SurfaceFiring
"""Alias for :class:`SurfaceFiring` (old name from root ``offline.py``)."""

OfflineAuditResult = OfflineRunResult
"""Alias for :class:`OfflineRunResult` (old name from root ``offline.py``)."""


async def offline_audit(
    messages: list[Any],
    *,
    cwd: str,
    provider: tuple[str, dict[str, Any]],
    audit_interval: int = 5,
    auditor_prompt: str = "minimal_index",
    stop_on_first_surface: bool = False,
    min_surface_turn_index: int = 0,
    **kwargs: Any,
) -> OfflineRunResult:
    """Thin wrapper around :func:`replay_pipeline_over_trajectory`.

    Preserves the old ``offline_audit()`` call signature used by
    ``rca_eval.replay_fork``.  Extra ``kwargs`` are silently dropped
    for backward compatibility.
    """
    from agentm.core.abi import AgentMessage
    from loguru import logger

    from .agents.auditor.context import load_auditor_prompt
    from .eval.replay.offline import StandaloneChildRunner
    from .eval.replay.runner import AuditorSettings

    typed_messages: list[AgentMessage] = list(messages)
    assert len(typed_messages) > 0, "offline_audit: empty message list"

    base_prompt = load_auditor_prompt(auditor_prompt)
    auditor_settings = AuditorSettings(base_prompt=base_prompt)

    session_id = f"offline-audit-{id(messages)}"
    child = StandaloneChildRunner(cwd)

    logger.info(
        f"offline_audit: {len(typed_messages)} messages, "
        f"interval={audit_interval}, prompt={auditor_prompt}"
    )
    result = await replay_pipeline_over_trajectory(
        messages=typed_messages,
        cwd=cwd,
        session_id=session_id,
        provider=provider,
        auditor_settings=auditor_settings,
        audit_interval=audit_interval,
        enable_auditor=True,
        stop_on_first_surface=stop_on_first_surface,
        child=child,
    )
    # Filter surfaces below min_surface_turn_index
    if min_surface_turn_index > 0:
        result.surfaces = [
            s for s in result.surfaces if s.turn_index > min_surface_turn_index
        ]
    return result


__all__ = [
    "OfflineAuditResult",
    "OfflineRunResult",
    "Reminder",
    "SurfaceFiring",
    "SurfacePoint",
    "Verdict",
    "offline_audit",
]
