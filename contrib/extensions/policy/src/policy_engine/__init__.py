# code-health: ignore-file[AM025] -- event payloads are untyped at the boundary
"""``policy_engine`` — detect structural failure patterns, intervene.

Live flow:

1. **ToolResultEvent** — queue repository-index refresh for read/write/edit;
   accumulate tool call info for the current turn.
2. **TurnCommittedEvent** — advance turn counter, reset per-turn state.
3. **DecideEvent** (async) — process pending symbol refreshes, run the tagger,
   evaluate trigger predicates, inject a mid-work check if one fires.
   When the agent wraps up instead, raise one stop-checkpoint check.
4. **submit tool** — the agent's way out of a check, and the only thing that
   ends the session cleanly.

The agent finishes the way it always did, in prose. That moment is the stop
checkpoint, and the check lands there. What the old design lacked was an exit:
a reply to a check looked exactly like a fresh attempt to finish, so it drew
the next check, and the next, until the budget ran out — the agent could not
end the session, only outlast the engine. ``submit`` is that exit, named in
every check, so one tool call always finishes. It never refuses; whether a
check was really met is a question for the trajectory afterwards, not for a
gate the agent has no way to pass.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BashOperations,
    FunctionTool,
    JsonValue,
    ProviderConfig,
    TextContent,
    ToolResult,
    ToolTerminate,
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

from .deliver import (
    AcceptanceVerdict,
    build_acceptance_prompt,
    build_injection,
    review_submission,
)
from .ifg.repository_index import RepositoryIndex, RepositoryRefreshPlan
from .paths import resolve_policy_path
from .pg_query import PgQuerySource
from .symbol_sync import extract_symbols_for_paths, write_symbols
from .tagger import (
    TaggerConversation,
    _content_text,
    render_turn,
    write_annotation,
)
from .triggers import TriggerEngine, load_items, render_check, render_stop_check


class PolicyEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist: str = "package:checklist.yaml"
    trajectory_dsn: str = ""
    # Provider registry name for the tagger and critic. Empty means the
    # session's own active provider — a registry name, not a config.toml
    # profile key, so a wrong value disables interventions rather than
    # silently falling back.
    provider: str = ""
    # Acceptance review at submit time: "off", or "on" to read the run against
    # the task before the session is allowed to end.
    critic: str = "off"
    # Mid-work checks, injected while the agent is working.
    max_injections: int = 5
    # Checks raised when the agent wraps up. Past the cap it is left to finish.
    max_stop_checks: int = 3
    # How many times acceptance may send the agent back. The session must
    # always be able to end, and a reviewer that never yields is a stuck loop.
    max_review_rounds: int = 2


MANIFEST = ExtensionManifest(
    name="policy_engine",
    description="Detects structural failure patterns in trajectories, intervenes.",
    registers=("tool:submit",),
    config_schema=PolicyEngineConfig,
    priority=AtomInstallPriority.POLICY,
)

_SUBMIT_DESCRIPTION = (
    "Confirm the task is finished and end the session. Use this to close out a "
    "process check once you have addressed it or established that it does not "
    "apply to this task."
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
    _stop_checks: int = 0
    _submitted: bool = False
    _tagger: TaggerConversation | None = None
    _provider: ProviderConfig | None = None
    _events: list[str] = field(default_factory=list)
    _concerns: list[str] = field(default_factory=list)
    _review_rounds: int = 0

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
        items = load_items(items_path) if items_path else {}
        if not items:
            logger.warning("policy_engine: missing checklist; inert")
            return
        provider = self.api.get_provider(self.config.provider or None)
        if provider is None:
            logger.warning(
                "policy_engine: provider {!r} not registered; interventions disabled",
                self.config.provider or "<active>",
            )
            return
        self._provider = provider
        self._tagger = TaggerConversation(
            session_id=self.session_id,
            stream_fn=provider.stream_fn,
            model=provider.model,
        )
        self.triggers = TriggerEngine(items=items)
        self._pg = PgQuerySource(self.config.trajectory_dsn, self.session_id)
        self.api.on(DecideEvent.CHANNEL, self._on_decide)
        self.api.register_tool(
            FunctionTool(
                name="submit",
                description=_SUBMIT_DESCRIPTION,
                parameters={  # code-health: ignore[AM011]
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": ("What was changed and what verified it."),
                        }
                    },
                    "required": ["summary"],
                },
                fn=self._submit,
            )
        )
        logger.info(
            "policy_engine: interventions active ({} items, provider={}, critic={})",
            len(items),
            provider.name,
            self.config.critic,
        )

    # -- submit ----------------------------------------------------------------

    async def _submit(self, args: dict[str, JsonValue]) -> ToolResult | ToolTerminate:
        """The agent's way out, and the moment acceptance runs.

        With the critic off this never refuses: a check the agent can only
        answer in words is not a gate, and the earlier refusing version was
        answered with wording rather than work. With it on, the refusal is
        backed by a reading of the run against the task, so "it does not apply"
        is something that can be checked rather than merely asserted.
        """
        verdict = await self._review(args)
        if verdict is not None:
            self._review_rounds += 1
            logger.info(
                "policy_engine: submission sent back ({}/{}): {}",
                self._review_rounds,
                self.config.max_review_rounds,
                verdict.finding[:120],
            )
            return ToolResult(
                content=[TextContent(type="text", text=verdict.as_message())],
                is_error=True,
            )

        self._submitted = True
        logger.info(
            "policy_engine: submitted after {} stop check(s), {} review round(s)",
            self._stop_checks,
            self._review_rounds,
        )
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="Submitted.")]),
            reason="policy:submitted",
        )

    async def _review(self, args: dict[str, JsonValue]) -> AcceptanceVerdict | None:
        """The acceptance verdict when it rejects, else None."""
        if self.config.critic != "on" or self._provider is None:
            return None
        if self._review_rounds >= self.config.max_review_rounds:
            return None
        summary = args.get("summary")
        prompt = build_acceptance_prompt(
            task=self._first_user_message(),
            summary=summary if isinstance(summary, str) else "",
            events=self._events,
            concerns=self._concerns,
        )
        verdict = await review_submission(
            prompt,
            stream_fn=self._provider.stream_fn,
            model=self._provider.model,
        )
        return None if verdict.accepted else verdict

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

        if self.triggers is None:
            return None
        stopping = isinstance(event.observation.default_action, Stop)

        # Wrapping up in prose is how this agent finishes; that moment is the
        # stop checkpoint. Hand it one check and name the way out. The old
        # design had no exit — every reply drew another check until the budget
        # ran out — so the check now carries `submit` with it.
        if stopping:
            if self._submitted or self._stop_checks >= self.config.max_stop_checks:
                return None
            item = self.triggers.next_triggered(
                stopping=True, active_tags=frozenset(self._active_tags)
            )
            if item is None:
                return None
            self._stop_checks += 1
            self._concerns.append(item.check)
            logger.info(
                "policy_engine: stop check {} ({}/{})",
                item.item_id,
                self._stop_checks,
                self.config.max_stop_checks,
            )
            return build_injection(render_stop_check(item))

        if self.injections >= self.config.max_injections:
            return None
        item = self.triggers.next_triggered(
            stopping=False, active_tags=frozenset(self._active_tags)
        )
        if item is None:
            return None

        self.injections += 1
        self._concerns.append(item.check)
        logger.info(
            "policy_engine: injecting {} ({}/{})",
            item.item_id,
            self.injections,
            self.config.max_injections,
        )
        return build_injection(render_check(item))

    async def _run_tagger(self, event: DecideEvent) -> None:
        if self._pg is None or self._tagger is None:
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

        self._events.append(render_turn(self.turn, assistant_text, tool_calls))
        annotation = await self._tagger.annotate(
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
