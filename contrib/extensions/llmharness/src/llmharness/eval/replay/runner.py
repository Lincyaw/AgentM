"""Single-firing replay: re-run one recorded auditor firing.

These functions back chain replay, dev-checkout replay tooling, and the RL
prompts exporter. A replay record already carries a finished
``payload`` + ``compose_kwargs``, so a firing needs none of the live
machinery (cadence, cumulative state, sinks): rebuild the per-firing
inputs from the record and call :func:`run_phase_standalone` directly.
"""

from __future__ import annotations

from typing import Any

from llmharness.agents.auditor.context import build_auditor_system_prompt
from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.context_index import build_context_index
from llmharness.eval.replay.record import ReplayRecord

from .engine import PhaseResult, run_phase_standalone

# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------

class AuditorSettings:
    """Minimal config needed to replay an auditor firing."""

    def __init__(
        self,
        *,
        base_prompt: str | None = None,
        tools: tuple[str, ...] | None = None,
        observability_config: dict[str, Any] | None = None,
    ) -> None:
        self.base_prompt = base_prompt
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
        return cls(
            base_prompt=base_prompt,
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
    """Run auditor on a recorded context-index payload.

    Composes auditor extensions from ``record.compose_kwargs``
    (continuation_notes / tools) and passes ``record.payload`` to the
    child as the user message verbatim.
    """
    if record.phase != "auditor":
        raise ValueError(f"expected auditor record, got phase={record.phase!r}")
    provider = provider_override or _coerce_provider(record)
    settings = AuditorSettings.from_compose_kwargs(
        record.compose_kwargs, prompt_override=prompt_override
    )

    ck = record.compose_kwargs or {}
    raw_context_index = ck.get("context_index")
    context_index = dict(raw_context_index) if isinstance(raw_context_index, dict) else None
    if context_index is None:
        trajectory = _coerce_trajectory(ck.get("trajectory_snapshot"))
        if trajectory:
            context_index = build_context_index(
                trajectory=trajectory,
                symbols=list(ck.get("symbols") or []),
                references=list(ck.get("references") or []),
            ).to_dict()
    prompt_text = build_auditor_system_prompt(
        check_errors=dict(ck.get("check_errors") or {}),
        continuation_notes=list(ck.get("continuation_notes") or []),
        base_prompt=settings.base_prompt or None,
        context_index=context_index,
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
