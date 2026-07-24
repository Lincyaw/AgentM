# code-health: ignore-file[AM025] -- event payloads are untyped at the boundary
"""``policy_engine`` — detect structural failure patterns, intervene.

Live flow:

1. **ToolResultEvent** — queue repository-index refresh for read/write/edit;
   accumulate tool call info for the current turn.
2. **TurnCommittedEvent** — advance turn counter, reset per-turn state.
3. **DecideEvent** (async) — process pending symbol refreshes, run tagger,
   evaluate signals + trigger predicates, intervene if a checklist item fires.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BashOperations,
)
from agentm.core.abi.events import (
    DecideEvent,
    LoopAction,
    Stop,
    ToolResultEvent,
    TurnCommittedEvent,
)
from agentm.core.abi.roles import BASH_OPERATIONS_SERVICE
from agentm.extensions import ExtensionManifest

from .deliver import build_injection, verify_item
from .ifg.repository_index import RepositoryIndex, RepositoryRefreshPlan
from .paths import resolve_policy_path
from .pg_query import PgQuerySource
from .symbol_sync import extract_symbols_for_paths, write_symbols
from .tagger import (
    _content_text,
    annotate_turn,
    write_annotation,
)
from .triggers import TriggerEngine, load_items, load_signals, render_message


class PolicyEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist: str = "package:checklist.yaml"
    signals: str = "package:signals.yaml"
    trajectory_dsn: str = ""
    max_injections: int = 3
    signal_params: dict[str, float] = Field(default_factory=dict)
    critic: str = "off"


MANIFEST = ExtensionManifest(
    name="policy_engine",
    description="Detects structural failure patterns in trajectories, intervenes.",
    registers=(),
    config_schema=PolicyEngineConfig,
    priority=AtomInstallPriority.POLICY,
)


def _interventions_enabled() -> bool:
    value = os.environ.get("AGENTM_CHECKLIST_WATCH_ENABLED", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _file_path_from_args(args: Mapping[str, object]) -> str | None:
    path = args.get("path") or args.get("file_path")
    return path if isinstance(path, str) and path.strip() else None


@dataclass(slots=True)
class _ToolCallRecord:
    name: str
    arguments: Mapping[str, object]
    result_text: str
    is_error: bool


@dataclass(slots=True)
class _Runtime:
    api: AtomAPI
    config: PolicyEngineConfig
    session_id: str
    turn: int = 0
    triggers: TriggerEngine | None = None
    injections: int = 0
    repo_index: RepositoryIndex | None = None
    _pending_refreshes: list[RepositoryRefreshPlan] = field(default_factory=list)
    _synced_paths: set[str] = field(default_factory=set)
    _pg: PgQuerySource | None = None
    _current_turn_calls: list[_ToolCallRecord] = field(default_factory=list)
    _task_classified: bool = False
    _active_tags: set[str] = field(default_factory=set)

    def install(self) -> None:
        self.api.on(ToolResultEvent.CHANNEL, self._on_tool_result)
        self.api.on(TurnCommittedEvent.CHANNEL, self._on_turn_committed)

        bash = self.api.services.get(BASH_OPERATIONS_SERVICE)
        if isinstance(bash, BashOperations):  # code-health: ignore[AM025]
            self.repo_index = RepositoryIndex(root=self.api.ctx.cwd, bash=bash)
            logger.info(
                "policy_engine: repository index enabled (root={})",
                self.api.ctx.cwd,
            )

        if not _interventions_enabled():
            logger.info("policy_engine: symbol sync only (interventions disabled)")
            return
        if not self.config.trajectory_dsn:
            logger.warning("policy_engine: no trajectory_dsn; interventions disabled")
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
        self._pg = PgQuerySource(self.config.trajectory_dsn, self.session_id)
        self.api.on(DecideEvent.CHANNEL, self._on_decide)
        logger.info(
            "policy_engine: interventions active ({} items, critic={})",
            len(items),
            self.config.critic,
        )

    # -- observe ---------------------------------------------------------------

    def _on_tool_result(self, event: ToolResultEvent) -> None:
        args = dict(event.args)
        self._current_turn_calls.append(
            _ToolCallRecord(
                name=event.tool_name,
                arguments=args,
                result_text=_content_text(event.result, 1500),
                is_error=event.result is not None and event.result.is_error,
            )
        )

        if event.tool_name not in {"read", "write", "edit"}:
            return
        path = _file_path_from_args(args)
        if path is None or self.repo_index is None:
            return
        cwd = self.api.ctx.cwd
        normalized = (
            posixpath.normpath(path)
            if posixpath.isabs(path)
            else posixpath.normpath(posixpath.join(cwd, path))
        )
        is_mutation = event.tool_name in {"write", "edit"}
        if normalized in self._synced_paths and not is_mutation:
            return
        self._pending_refreshes.append(
            RepositoryRefreshPlan(paths=(normalized,), reason=f"tool:{event.tool_name}")
        )

    def _on_turn_committed(self, event: TurnCommittedEvent) -> None:
        self.turn += 1
        self._current_turn_calls = []

    # -- detect + intervene ----------------------------------------------------

    async def _on_decide(self, event: DecideEvent) -> LoopAction | None:
        await self._flush_symbol_refreshes()
        await self._run_tagger(event)

        if self.triggers is None or self._pg is None:
            return None
        if self.injections >= self.config.max_injections:
            return None
        stopping = isinstance(event.observation.default_action, Stop)
        active = frozenset(self._active_tags)

        firing = self.triggers.evaluate_inject(
            self._pg, stopping=stopping, active_tags=active
        )
        if firing is not None:
            self.injections += 1
            logger.info(
                "policy_engine: injecting {} ({}/{})",
                firing.item.item_id,
                self.injections,
                self.config.max_injections,
            )
            return build_injection(render_message(firing))

        if stopping and self.config.critic != "off":
            return await self._run_critic(active)

        return None

    async def _run_critic(self, active_tags: frozenset[str]) -> LoopAction | None:
        if self.triggers is None or self._pg is None:
            return None
        candidates = self.triggers.open_critic_items(self._pg, active_tags=active_tags)
        if not candidates:
            return None
        for item in candidates[:5]:
            violated, reasoning = await verify_item(self.api, self._pg, item)
            if violated:
                self.injections += 1
                message = (
                    f"Process check ({item.dimension}):\n"
                    f"{item.check}\n\n"
                    f"Finding: {reasoning}"
                )
                logger.info(
                    "policy_engine: critic confirmed {} ({})",
                    item.item_id,
                    reasoning[:80],
                )
                return build_injection(message)
        return None

    async def _run_tagger(self, event: DecideEvent) -> None:
        if self._pg is None:
            return

        assistant_text = _content_text(event.observation.assistant_message, 3000)
        tool_calls = [
            {
                "name": tc.name,
                "arguments": dict(tc.arguments),
                "result_text": tc.result_text,
                "is_error": tc.is_error,
            }
            for tc in self._current_turn_calls
        ]
        if not assistant_text and not tool_calls:
            return

        task_text = ""
        if not self._task_classified:
            self._task_classified = True
            task_text = self._first_user_message()

        annotation = await annotate_turn(
            self.api,
            session_id=self.session_id,
            turn_index=self.turn,
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            task_text=task_text,
        )
        if annotation is not None:
            write_annotation(self._pg, annotation)
            self._active_tags.update(annotation.tags)
            logger.debug(
                "policy_engine: turn {} → phase={} tags={}",
                self.turn,
                annotation.phase,
                annotation.tags,
            )

    def _first_user_message(self) -> str:
        messages = self.api.get_messages()
        for msg in messages:
            role = getattr(msg, "role", None)  # code-health: ignore[AM021]
            if role == "user":
                return _content_text(msg, 3000)
        return ""

    async def _flush_symbol_refreshes(self) -> None:
        if not self._pending_refreshes or self.repo_index is None:
            return
        plans = list(self._pending_refreshes)
        self._pending_refreshes.clear()
        for plan in plans:
            try:
                await self.repo_index.refresh(plan)
            except Exception as exc:  # noqa: BLE001
                logger.warning("policy_engine: repo index refresh failed: {}", exc)
                continue
            if self._pg is None:
                continue
            rows = extract_symbols_for_paths(
                self.repo_index,
                session_id=self.session_id,
                paths=list(plan.paths),
            )
            if rows:
                written = write_symbols(
                    self._pg,
                    rows,
                    session_id=self.session_id,
                    paths=list(plan.paths),
                )
                logger.debug(
                    "policy_engine: synced {} symbols for {}",
                    written,
                    plan.paths,
                )
            self._synced_paths.update(plan.paths)


def install(api: AtomAPI, config: PolicyEngineConfig) -> None:
    _Runtime(api=api, config=config, session_id=api.ctx.session_id).install()


__all__ = [
    "MANIFEST",
    "PolicyEngineConfig",
    "install",
]
