# code-health: ignore-file[AM025] -- event payloads are untyped at the boundary
"""``policy_engine`` — record, detect, intervene.

The whole package is three moves over one stream of raw data:

1. **Record**: every tool result is appended to a per-session SQLite file
   (``recording``). That corpus is also the calibration bench — every
   trigger added here must first prove its rates on it (``__main__ replay``).
2. **Detect**: ``signals`` maintains O(1) trajectory counters; ``triggers``
   binds checklist items to them. Structural items fire deterministically;
   critic items are questions saved for the stop decision.
3. **Intervene** (``deliver``): a firing either injects a message into the
   working agent's own loop, or spawns a reviewer subagent whose confirmed
   verdict is injected.

Interventions are off by default so baseline batches stay clean; enable per
run via ``AGENTM_CHECKLIST_WATCH_ENABLED=true`` (name kept for continuity
with existing run scripts). Recording is always on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    TextContent,
    ToolResult,
)
from agentm.core.abi.events import (
    DecideEvent,
    LoopAction,
    Stop,
    ToolResultEvent,
    TurnCommittedEvent,
)
from agentm.extensions import ExtensionManifest

from .deliver import build_injection, run_critic, self_check_message
from .paths import default_policy_db_path, resolve_policy_path
from .recording import ToolEventRecorder
from .plane import DataPlane
from .triggers import TriggerEngine, load_items, load_signals, render_message


class PolicyEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist: str = "package:checklist.yaml"
    signals: str = "package:signals.yaml"
    db_path: str | None = None
    max_injections: int = 3
    signal_params: dict[str, float] = Field(default_factory=dict)
    # Delivery for critic-tier items at the stop decision:
    #   "off"        — structural tier only
    #   "self_check" — inject the checklist questions for the agent's own review
    #   "subagent"   — spawn a reviewer child and inject its confirmed verdict
    critic: str = "off"


MANIFEST = ExtensionManifest(
    name="policy_engine",
    description="Records tool events, evaluates checklist triggers, intervenes.",
    registers=(),
    config_schema=PolicyEngineConfig,
    priority=AtomInstallPriority.POLICY,
)


def _interventions_enabled() -> bool:
    # Read directly: the runtime has no generic env-config layer.
    value = os.environ.get("AGENTM_CHECKLIST_WATCH_ENABLED", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _message_text(message: object) -> str:
    content = getattr(message, "content", None)  # code-health: ignore[AM021]
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)  # code-health: ignore[AM021]
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _result_text(result: ToolResult | None) -> str:
    if result is None:
        return ""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)  # code-health: ignore[AM021]
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


@dataclass(slots=True)
class _Runtime:
    api: AtomAPI
    config: PolicyEngineConfig
    recorder: ToolEventRecorder
    turn: int = 0
    claims: list[tuple[int, bool, str]] = field(default_factory=list)
    plane: DataPlane | None = None
    triggers: TriggerEngine | None = None
    injections: int = 0
    critic_done: bool = False

    def install(self) -> None:
        self.api.on(ToolResultEvent.CHANNEL, self._on_tool_result)
        self.api.on(TurnCommittedEvent.CHANNEL, self._on_turn_committed)

        if not _interventions_enabled():
            logger.info("policy_engine: recording only (interventions disabled)")
            return
        items_path = resolve_policy_path(
            self.config.checklist, cwd=Path(self.api.ctx.cwd)
        )
        signals_path = resolve_policy_path(
            self.config.signals, cwd=Path(self.api.ctx.cwd)
        )
        items = load_items(items_path) if items_path else {}
        signals = load_signals(signals_path) if signals_path else {}
        if not items or not signals:
            logger.warning("policy_engine: missing checklist/signals; inert")
            return
        self.triggers = TriggerEngine(
            items=items,
            signals=signals,
            param_overrides=dict(self.config.signal_params),
        )
        self.api.on(DecideEvent.CHANNEL, self._on_decide)
        logger.info(
            "policy_engine: interventions active ({} items, critic={})",
            len(items),
            self.config.critic,
        )

    # -- record + track ------------------------------------------------------

    def _on_turn_committed(self, event: TurnCommittedEvent) -> None:
        self.turn += 1

    def _on_tool_result(self, event: ToolResultEvent) -> None:
        # Record only — every derived fact comes from the data plane, which
        # rebuilds from these rows at each decision point. Failed bash runs
        # are recorded like any other row: a nonzero exit is marked is_error
        # by the runtime, and dropping those rows blinded the old watcher to
        # every red run (found 2026-07-24).
        self.recorder.record(
            turn=self.turn,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id or None,
            args=dict(event.args),
            is_error=event.result is not None and event.result.is_error,
            result_text=_result_text(event.result),
            exit_code=event.exit_code,
            duration_ms=event.duration_ms,
            cwd=self.api.ctx.cwd,
        )

    # -- detect + intervene ----------------------------------------------------

    async def _on_decide(self, event: DecideEvent) -> LoopAction | None:
        if self.triggers is None or self.injections >= self.config.max_injections:
            return None
        stopping = isinstance(event.observation.default_action, Stop)

        claim = _message_text(event.observation.assistant_message)
        if claim:
            self.claims.append((self.turn, stopping, claim))

        if self.plane is None:
            self.plane = DataPlane.open(self.recorder.db_path)
        self.plane.rebuild()
        self.plane.ingest_claims(self.claims)

        firing = self.triggers.evaluate_inject(self.plane, stopping=stopping)
        if firing is not None:
            self.injections += 1
            message = render_message(firing)
            logger.info(
                "policy_engine: injecting {} ({} chars, {}/{})",
                firing.item.item_id,
                len(message),
                self.injections,
                self.config.max_injections,
            )
            return build_injection(message)

        if stopping and not self.critic_done and self.config.critic != "off":
            self.critic_done = True
            items = self.triggers.open_critic_items(self.plane)
            evidence = self.triggers.critic_evidence(self.plane)
            if not items and not evidence:
                return None
            if self.config.critic == "self_check":
                self.injections += 1
                return build_injection(self_check_message(items, evidence))
            if self.config.critic == "subagent":
                verdict = await run_critic(
                    self.api, items=items, evidence=evidence, plane=self.plane
                )
                if verdict is not None:
                    self.injections += 1
                    return build_injection(verdict)
        return None


def install(api: AtomAPI, config: PolicyEngineConfig) -> None:
    base = (
        Path(config.db_path).expanduser()
        if config.db_path
        else default_policy_db_path()
    )
    session_id = api.ctx.session_id
    safe = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id
    )[:128]
    recorder = ToolEventRecorder(
        db_path=base.parent / "sessions" / f"{safe or 'session'}.db",
        session_id=session_id,
    )
    _Runtime(api=api, config=config, recorder=recorder).install()


__all__ = [
    "MANIFEST",
    "PolicyEngineConfig",
    "TextContent",
    "install",
]
