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

import asyncio
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
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.events import (
    DecideEvent,
    LoopAction,
    RunEndEvent,
    Stop,
    ToolResultEvent,
    TurnCommittedEvent,
)
from agentm.core.abi.roles import BASH_OPERATIONS_SERVICE
from agentm.extensions import ExtensionManifest

from .acceptance import AcceptanceReviewer, AcceptanceVerdict, build_prompt
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
from .triggers import (
    TriggerEngine,
    build_injection,
    load_items,
    render_check,
    render_stop_check,
)


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
    # Turns per tagger call. The tags feed a cumulative set that triggers read
    # as a whole, so per-turn resolution buys nothing and costs one model call
    # per turn. The buffer is flushed early whenever a decision needs the tags.
    tagger_interval: int = 5
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
    _active_tags: set[str] = field(default_factory=set)
    _stop_checks: int = 0
    _submitted: bool = False
    _tagger: TaggerConversation | None = None
    _provider_missing_logged: bool = False
    # Every turn, rendered once. The tagger reads from _tagged onward; the
    # reviewer reads the whole thing at submit.
    _turns: list[str] = field(default_factory=list)
    _tagged: int = 0
    _reviewer: AcceptanceReviewer | None = None
    _concerns: list[str] = field(default_factory=list)
    _review_rounds: int = 0

    def install(self) -> None:
        self.api.on(ToolResultEvent.CHANNEL, self._on_tool_result)
        self.api.on(TurnCommittedEvent.CHANNEL, self._on_turn_committed)
        self.api.on(RunEndEvent.CHANNEL, self._on_run_end)

        bash = self.api.services.get(BASH_OPERATIONS_SERVICE)
        if isinstance(bash, BashOperations):  # code-health: ignore[AM025]
            self.repo_index = RepositoryIndex(root=self.api.ctx.cwd, bash=bash)
            self._reviewer = AcceptanceReviewer(
                api=self.api, bash=bash, cwd=self.api.ctx.cwd
            )
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
            "policy_engine: interventions active ({} items, critic={})",
            len(items),
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
        await self._flush_tagger()
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
        if self.config.critic != "on":
            return None
        if self._review_rounds >= self.config.max_review_rounds:
            return None
        if self._reviewer is None:
            return None
        summary = args.get("summary")
        prompt = build_prompt(
            task=self._first_user_message(),
            summary=summary if isinstance(summary, str) else "",
            events=self._turns,
            concerns=self._concerns,
        )
        verdict = await self._reviewer.review(prompt)
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

    async def _on_run_end(self, event: RunEndEvent) -> None:
        """Tag whatever is still buffered when the run stops.

        Batching means turns can be in hand but unread when the session ends —
        including the last stretch before submit, which is where a session
        tends to be most worth reading. Nothing acts on these tags, but they
        are what the offline analysis and the next checklist revision see.
        """
        await self._flush_tagger()

    # -- detect + intervene ----------------------------------------------------

    async def _on_decide(self, event: DecideEvent) -> LoopAction | None:
        # Nothing on this path reads the symbol index — triggers read the tag
        # set — so indexing runs alongside the tagger rather than in front of
        # it. Serialised, its ast-grep and PG time was pure added latency on
        # every turn.
        indexing = asyncio.create_task(self._flush_symbol_refreshes())
        self._record_turn(event)

        if self.triggers is None:
            await indexing
            return None
        stopping = isinstance(event.observation.default_action, Stop)

        # Wrapping up is a decision point, so the buffer is flushed there
        # regardless of how few turns it holds.
        pending = len(self._turns) - self._tagged
        if stopping or pending >= self.config.tagger_interval:
            await self._flush_tagger()
        await indexing

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

    def _resolve_tagger(self) -> TaggerConversation | None:
        """The tagger, once a provider exists to run it on.

        Resolved on first use rather than at install: atoms install by priority
        band and POLICY (300) comes before PROVIDER (400), so at install time
        the registry is still empty and the atom would disable itself.
        """
        if self._tagger is not None:
            return self._tagger
        provider = self.api.get_provider(self.config.provider or None)
        if provider is None:
            if not self._provider_missing_logged:
                self._provider_missing_logged = True
                logger.warning(
                    "policy_engine: provider {!r} not registered; tagging off",
                    self.config.provider or "<active>",
                )
            return None
        logger.info("policy_engine: tagging on provider {}", provider.name)
        self._tagger = TaggerConversation(
            session_id=self.session_id,
            stream_fn=provider.stream_fn,
            model=provider.model,
        )
        return self._tagger

    def _record_turn(self, event: DecideEvent) -> None:
        """Buffer this turn for the tagger, and keep it for the reviewer."""
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

        # The task text rides along with the first turn, so it stays in the
        # tagger's cached prefix for the rest of the session.
        task_text = self._first_user_message() if not self._turns else ""
        self._turns.append(
            render_turn(self.turn, assistant_text, tool_calls, task_text=task_text)
        )

    async def _flush_tagger(self) -> None:
        """Tag the buffered turns as one batch.

        Called on the interval, and unconditionally before anything reads the
        tag set — a check decided on stale tags is a check decided on the wrong
        session.
        """
        tagger = self._resolve_tagger()
        if self._pg is None or tagger is None:
            return
        batch = self._turns[self._tagged :]
        if not batch:
            return
        self._tagged = len(self._turns)
        annotation = await tagger.annotate(batch, turn_index=self.turn)
        if annotation is None:
            return
        write_annotation(self._pg, annotation)
        new_tags = set(annotation.tags) - self._active_tags
        self._active_tags.update(annotation.tags)
        logger.debug(
            "policy_engine: turns ..{} ({} batched) → phase={} new tags={}",
            self.turn,
            len(batch),
            annotation.phase,
            sorted(new_tags),
        )

    def _first_user_message(self) -> str:
        messages = self.api.get_messages()
        for msg in messages:
            role = getattr(msg, "role", None)  # code-health: ignore[AM021]
            if role == "user":
                return _content_text(msg, 3000)
        return ""

    async def _flush_symbol_refreshes(self) -> None:
        """Re-index the paths touched since the last flush.

        Deduped first: a turn that edits one file three times used to spawn
        three ast-grep runs over the same file, since each tool call queued its
        own plan and the already-synced check ran only after the flush.
        """
        if not self._pending_refreshes or self.repo_index is None:
            return
        by_reason: dict[str, set[str]] = {}
        for plan in self._pending_refreshes:
            by_reason.setdefault(plan.reason, set()).update(plan.paths)
        self._pending_refreshes.clear()

        for reason, path_set in by_reason.items():
            paths = sorted(path_set)
            try:
                await self.repo_index.refresh(
                    RepositoryRefreshPlan(paths=tuple(paths), reason=reason)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("policy_engine: repo index refresh failed: {}", exc)
                continue
            self._synced_paths.update(paths)
            if self._pg is None:
                continue
            rows = extract_symbols_for_paths(
                self.repo_index, session_id=self.session_id, paths=paths
            )
            if rows:
                written = write_symbols(
                    self._pg, rows, session_id=self.session_id, paths=paths
                )
                logger.debug("policy_engine: synced {} symbols for {}", written, paths)


def install(api: AtomAPI, config: PolicyEngineConfig) -> None:
    _Runtime(api=api, config=config, session_id=api.ctx.session_id).install()


__all__ = [
    "MANIFEST",
    "PolicyEngineConfig",
    "install",
]
