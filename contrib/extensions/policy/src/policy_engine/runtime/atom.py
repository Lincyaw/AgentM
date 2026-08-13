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
from policy_engine.runtime.actions import Action, InjectAction, ReviewAction
from policy_engine.runtime.critic import (
    PLAN_APPROVER_SERVICE,
    Critic,
    CriticVerdict,
    PlanCritic,
    build_prompt,
    build_revision_prompt,
)
from policy_engine.runtime.critic import (
    PURPOSE as ACCEPTANCE_PURPOSE,
)
from policy_engine.runtime.ifg.repository_index import (
    RepositoryIndex,
    RepositoryRefreshPlan,
)
from policy_engine.runtime.record import content_text as _content_text
from policy_engine.runtime.record import render_turn
from policy_engine.runtime.revision import RevisionDetector
from policy_engine.runtime.symbol_sync import extract_symbols_for_paths, write_symbols
from policy_engine.runtime.tagger import (
    TaggerConversation,
    tagger_system_prompt,
    write_annotation,
)
from policy_engine.runtime.triggers import (
    REWORKED,
    STOPPING,
    Budget,
    ChecklistItem,
    Moment,
    TriggerEngine,
    build_injection,
    load_items,
    render_finish_reminder,
)
from policy_engine.shared import facts
from policy_engine.shared.paths import resolve_policy_path
from policy_engine.shared.pg_query import PgQuerySource


class PolicyEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist: str = "package:checklist.yaml"
    # Where the run's turns are written. Empty falls back to the same env var
    # the trajectory store itself reads, for the same reason the schema below
    # does: the two halves of one connection must not be resolved differently,
    # or a run whose store is pointed elsewhere leaves this atom querying a
    # database it never wrote to -- where every precondition reads "unmet" and
    # every gated item goes quiet, with one warning as the only evidence.
    trajectory_dsn: str = ""
    # Schema the run's turns are written to. Facts are queried per-schema, so a
    # precondition cannot be evaluated without it. Empty falls back to the same
    # env var the trajectory store itself reads, which is what keeps the two
    # pointing at one place.
    trajectory_schema: str = ""
    # Provider registry name for the tagger. Empty means the session's own
    # active provider — a registry name, not a config.toml profile key, so a
    # wrong value disables interventions rather than silently falling back.
    provider: str = ""
    # Provider for the acceptance reviewer, which is the most reasoning-heavy
    # thing here and does not want the tagger's cheap model. Empty inherits
    # `provider`, and that default once put the reviewer on the config's
    # default_model while the agent it reviewed ran on a far stronger one.
    critic_provider: str = ""
    # Acceptance review at submit time: "off", or "on" to read the run against
    # the task before the session is allowed to end.
    critic: str = "off"
    # Composition the reviewer runs under, for the verdict entries (submit and
    # rework) — the same shell submit_for_review names. Empty inherits this
    # scenario whole, which hands the reviewer the worker's write tools and
    # prepends the worker's prompt to the critic's.
    critic_scenario: str = ""
    # Turns per tagger call. The tags feed a cumulative set that triggers read
    # as a whole, so per-turn resolution buys nothing and costs one model call
    # per turn. The buffer is flushed early whenever a decision needs the tags.
    tagger_interval: int = 5
    # Everything the engine may spend interrupting one run, of any kind. This
    # replaced three separate caps -- mid-work injections, stop checks, reviews
    # at reworks -- which summed to eleven interruptions that nothing computed.
    # An item may still cap itself with `max_fires`.
    max_interventions: int = 6
    # How many times acceptance may send the agent back from `submit`. Not part
    # of the budget above: that one rations the agent's attention, this one
    # guarantees the session can end. A reviewer that never yields is a stuck
    # loop whatever the budget says.
    max_review_rounds: int = 2


MANIFEST = ExtensionManifest(
    name="policy_engine",
    description="Detects structural failure patterns in trajectories, intervenes.",
    registers=("tool:submit",),
    config_schema=PolicyEngineConfig,
)

_SUBMIT_DESCRIPTION = (
    "Confirm the task is finished and end the session. Use this to close out a "
    "process check once you have addressed it or established that it does not "
    "apply to this task."
)


#: Budget label for the exit contract, so it is bounded by the same pool as
#: everything else rather than by a counter of its own.
_FINISH_REMINDER = "finish-reminder"
_MAX_FINISH_REMINDERS = 2

#: Enough of the exit contract's opening to recognise it in a restored turn,
#: taken from the renderer itself so the two cannot drift apart.
_FINISH_REMINDER_MARK = render_finish_reminder().split(".", 1)[0]

#: What a restored interruption is charged to when the item that paid for it
#: cannot be named — a reviewer's finding quotes the reviewer, not the item.
_RESTORED_INTERVENTION = "restored-intervention"


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
    #: Installed actions by the name an item's ``deliver`` uses. An item naming
    #: something absent is skipped loudly; before this there was one action and
    #: the field pretended otherwise.
    actions: dict[str, Action] = field(default_factory=dict)
    repo_index: RepositoryIndex | None = None
    _pending_refreshes: list[RepositoryRefreshPlan] = field(default_factory=list)
    _synced_paths: set[str] = field(default_factory=set)
    _pg: PgQuerySource | None = None
    _current_turn_calls: list[_ToolCallRecord] = field(default_factory=list)
    _active_tags: set[str] = field(default_factory=set)
    _submitted: bool = False
    _tagger: TaggerConversation | None = None
    _provider_missing_logged: bool = False
    # Every turn, rendered once. The tagger reads from _tagged onward; the
    # reviewer reads the whole thing at submit.
    _turns: list[str] = field(default_factory=list)
    _tagged: int = 0
    _reviewer: Critic | None = None
    _concerns: list[str] = field(default_factory=list)
    _review_rounds: int = 0
    #: A predicate source, not a mechanism of its own: it answers whether this
    #: turn reworked earlier work, and the answer joins the tagger's tags in
    #: one namespace the gates read.
    _revisions: RevisionDetector = field(default_factory=RevisionDetector)
    # Preconditions answered so far this turn. A fact is derived from committed
    # turns, so within one decide the answer cannot change -- and one decide asks
    # twice, once on the stop path and once on the continuous one, for every item
    # whose tags match. Cleared on commit, because between turns it very much can.
    _fact_cache: dict[str, bool] = field(default_factory=dict)

    def install(self) -> None:
        # A reviewer must not install the policy that spawned it: it would
        # register its own submit, tag its own turns, and spawn a reviewer of
        # its own.
        if self.api.ctx.purpose == ACCEPTANCE_PURPOSE:
            logger.debug("policy_engine: inert inside an acceptance review")
            return
        self.api.on(ToolResultEvent.CHANNEL, self._on_tool_result)
        self.api.on(TurnCommittedEvent.CHANNEL, self._on_turn_committed)
        self.api.on(RunEndEvent.CHANNEL, self._on_run_end)

        dsn = self.config.trajectory_dsn or os.environ.get("AGENTM_TRAJECTORY_DSN", "")
        bash = self.api.services.get(BASH_OPERATIONS_SERVICE)
        if isinstance(bash, BashOperations):  # code-health: ignore[AM025]
            # Symbols sync to the trajectory database, so without a DSN there
            # is nowhere to write and indexing would be per-turn cost for
            # nothing.
            if dsn:
                self.repo_index = RepositoryIndex(root=self.api.ctx.cwd, bash=bash)
                self._pg = PgQuerySource(dsn, self.session_id)
                logger.info(
                    "policy_engine: repository index enabled (root={})",
                    self.api.ctx.cwd,
                )
            else:
                logger.info(
                    "policy_engine: repository index off "
                    "(no trajectory_dsn to sync symbols to)"
                )

        interventions = self._install_interventions(dsn)
        # DecideEvent drives both halves — draining the refresh queue, and,
        # with interventions on, tagging plus injection. Subscribed once, after
        # the checks above decided which halves exist: a handler subscribed
        # before an early return used to leave the refresh queue filling with
        # nothing ever draining it.
        if self.repo_index is not None or interventions:
            self.api.on(DecideEvent.CHANNEL, self._on_decide)

    def _install_interventions(self, dsn: str) -> bool:
        if not _interventions_enabled():
            logger.info("policy_engine: interventions disabled")
            return False
        if not dsn:
            logger.warning("policy_engine: no trajectory_dsn; interventions disabled")
            return False
        items_path = resolve_policy_path(
            self.config.checklist, cwd=Path(self.api.ctx.cwd)
        )
        items = load_items(items_path) if items_path else {}
        if not items and self.config.critic != "on":
            # An empty checklist is only fatal when nothing else would act. The
            # exit contract and any review-delivering item both come from the
            # critic being on, and neither reads the checklist.
            logger.warning(
                "policy_engine: checklist {!r} not found or empty; "
                "interventions disabled",
                self.config.checklist,
            )
            return False
        self._reviewer = Critic(
            api=self.api,
            provider=self.config.critic_provider or self.config.provider,
            scenario=self.config.critic_scenario,
        )
        # The registry is the switch. `critic: "off"` has to mean no reviewer
        # runs, and folding the old `revision_review` flag into an item left
        # nothing holding that door: the item fired and spawned a blocking
        # reviewer whatever the config said. Withholding the action is the
        # whole of the fix, and it needs no second knob to express.
        self.actions = {"inject": InjectAction()}
        if self.config.critic == "on":
            self.actions["review"] = ReviewAction(
                reviewer=self._reviewer,
                prompt_for=self._revision_prompt,
            )
        items = self._installable(items)
        if self.config.critic == "on":
            # plan_mode looks its approver up by name and accepts anything with
            # a `review`; registering over its AutoApprover is how the critic
            # reaches the one checkpoint that precedes the code. Tree-scoped so
            # a child inherits it, matching how plan_mode scopes its own.
            self.api.services.register(
                PLAN_APPROVER_SERVICE,
                PlanCritic(critic=self._reviewer, task_for=self._first_user_message),
                scope="tree",
            )
        self.triggers = TriggerEngine(
            items=items, budget=Budget(total=self.config.max_interventions)
        )
        if self._pg is None:
            self._pg = PgQuerySource(dsn, self.session_id)
        self._restore()
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
        return True

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
            "policy_engine: submitted after {}/{} intervention(s), {} review round(s)",
            self.triggers.budget.spent if self.triggers else 0,
            self.triggers.budget.total if self.triggers else 0,
            self._review_rounds,
        )
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="Submitted.")]),
            reason="policy:submitted",
        )

    async def _review(self, args: dict[str, JsonValue]) -> CriticVerdict | None:
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
        self._fact_cache.clear()

    async def _on_run_end(self, event: RunEndEvent) -> None:
        """Tag whatever is still buffered when the run stops.

        Batching means turns can be in hand but unread when the session ends —
        including the last stretch before submit, which is where a session
        tends to be most worth reading. Nothing acts on these tags, but they
        are what the offline analysis and the next checklist revision see.
        The refresh queue is drained for the same reason: the final turn's
        edits are exactly the ones a later reader asks about.
        """
        await self._flush_symbol_refreshes()
        await self._flush_tagger()

    # -- detect + intervene ----------------------------------------------------

    async def _on_decide(self, event: DecideEvent) -> LoopAction | None:
        # Nothing on this path reads the symbol index — triggers read the tag
        # set — so indexing runs alongside the tagger rather than in front of
        # it. Serialised, its ast-grep and PG time was pure added latency on
        # every turn.
        indexing = asyncio.create_task(self._flush_symbol_refreshes())
        if self.triggers is None:
            await indexing
            return None
        self._record_turn(event)
        stopping = isinstance(event.observation.default_action, Stop)

        # Wrapping up is a decision point, so the buffer is flushed there
        # regardless of how few turns it holds.
        pending = len(self._turns) - self._tagged
        if stopping or pending >= self.config.tagger_interval:
            await self._flush_tagger()
        await indexing

        moment = self._moment(event, stopping=stopping)
        if stopping and self._submitted:
            return None

        item = self.triggers.next_triggered(moment, fact_check=self._fact_check)
        if item is None:
            return self._finish_reminder(moment)

        action = self.actions.get(item.deliver)
        if action is None:
            logger.warning(
                "policy_engine: {} names action {!r}, which is not installed",
                item.item_id,
                item.deliver,
            )
            return None

        # Recorded whether or not the action produces anything. The acceptance
        # reviewer reads these at submit as concerns the run raised, and one
        # that fired and was silently satisfied is still a concern that was
        # raised.
        self._concerns.append(item.check)
        logger.info(
            "policy_engine: {} via {} at turn {} ({}/{})",
            item.item_id,
            item.deliver,
            moment.turn_index,
            self.triggers.budget.spent,
            self.triggers.budget.total,
        )
        return await action.deliver(item, moment)

    def _installable(self, items: dict[str, ChecklistItem]) -> dict[str, ChecklistItem]:
        """Drop items whose action is not installed, once and loudly.

        Dropped here rather than skipped at fire time so the budget is never
        charged for an item that cannot be delivered, and so a checklist naming
        an action this host does not have says so at install instead of on
        whichever turn it first matches.
        """
        keep = {k: v for k, v in items.items() if v.deliver in self.actions}
        for item_id, item in items.items():
            if item_id not in keep:
                logger.warning(
                    "policy_engine: {} wants action {!r}, not installed here; "
                    "item disabled",
                    item_id,
                    item.deliver,
                )
        return keep

    def _revision_prompt(self, moment: Moment) -> str:
        """What a review-delivering item sends the critic mid-task."""
        return build_revision_prompt(
            task=self._first_user_message(), events=self._turns
        )

    def _moment(self, event: DecideEvent, *, stopping: bool) -> Moment:
        """This turn as the gates see it: the tagger's tags plus the run's own.

        The stream predicates are recomputed every turn and never accumulate.
        ``reworked_own_work`` is true of the turn that reworked something and
        false of the next one, which is the whole reason it could not simply be
        added to the tag set.

        The detector observes unconditionally, even when nothing can fire. It
        recognises a rework by matching an edit against what this run wrote
        earlier, so a turn it does not see is a turn missing from what the next
        one is compared against.
        """
        turn_index = event.observation.turn_index
        predicates = set(self._active_tags)
        if stopping:
            predicates.add(STOPPING)
        if self._revisions.observe(turn_index, self._current_turn_calls):
            predicates.add(REWORKED)
        return Moment(turn_index=turn_index, predicates=frozenset(predicates))

    def _finish_reminder(self, moment: Moment) -> LoopAction | None:
        """The exit contract, which is engine behaviour rather than an item.

        The agent only learns ``submit`` exists from this message, so it cannot
        depend on a checklist matching -- leaving it to one made review a
        coincidence: three recorded sessions, one review. For the same reason
        it is not itself a checklist item: a deployment shipping its own
        checklist would drop it, and the agent would have no way to finish.

        Charged to the same budget as everything else, which is what bounds it.
        Twice, because an agent that wraps up in prose a second time has not
        heard it; and if the budget is gone the session simply ends without a
        review, which is the fail-open this whole subsystem already takes.
        """
        if not moment.stopping or self.config.critic != "on" or self._submitted:
            return None
        if self.triggers is None or not self.triggers.budget.may_spend(
            _FINISH_REMINDER, _MAX_FINISH_REMINDERS
        ):
            return None
        self.triggers.budget.spend(_FINISH_REMINDER)
        logger.info("policy_engine: finish reminder, no item matched")
        return build_injection(render_finish_reminder())

    def _fact_check(self, sql: str) -> bool:
        """Whether a precondition holds for this session, right now.

        False on any failure. A precondition that cannot be evaluated has not
        been shown to hold, and firing a check on an unanswered question is the
        thing preconditions exist to stop.
        """
        cached = self._fact_cache.get(sql)
        if cached is not None:
            return cached
        if self._pg is None:
            return False
        schema = self.config.trajectory_schema or os.environ.get(
            "AGENTM_TRAJECTORY_SCHEMA", ""
        )
        if not schema:
            logger.warning("policy_engine: no trajectory schema; preconditions unmet")
            return False
        try:
            rows = self._pg.query(
                facts.expand(sql, schema), {"session_id": self.session_id}
            )
        except Exception as exc:  # noqa: BLE001 - a bad precondition is not fatal
            logger.warning("policy_engine: precondition failed: {}", exc)
            return False
        held = bool(rows)
        # Only a real answer is cached. A failure is not knowledge, and the next
        # turn should ask again rather than inherit a false negative.
        self._fact_cache[sql] = held
        return held

    def _restore(self) -> None:
        """Rebuild what a fork does not carry, from the trajectory that does.

        Atom state is memory, so a session forked mid-run starts with an empty
        event log, nothing spent, no memory of what it wrote, and no tags — and
        the acceptance reviewer would then read an empty run. All of it is
        already durable: turns in the trajectory, delivered checks in each
        turn's injected messages, tags in policy.turn_annotations under the
        sessions this one descends from.

        One kind of firing leaves nothing to find. A review that looked and
        accepted spends from the budget and injects no message, so it cannot be
        charged back here, and a forked run's allowance is that many
        interruptions richer than the source's was.
        """
        for turn in self.api.get_turns():
            calls = [
                _ToolCallRecord(
                    name=record.call.name,
                    arguments=dict(record.call.arguments),
                    result_text=_content_text(record.result, 1500),
                    is_error=record.result.is_error,
                )
                for record in turn.tool_results
            ]
            self._turns.append(
                render_turn(
                    turn.index,
                    _content_text(turn.response, 3000),
                    [
                        {
                            "name": call.name,
                            "arguments": dict(call.arguments),
                            "result_text": call.result_text,
                            "is_error": call.is_error,
                        }
                        for call in calls
                    ],
                )
            )
            # Replayed through the detector rather than skipped: it answers
            # whether an edit reworks something this run wrote, and the writes
            # it never saw are missing from every later comparison. Skipped,
            # the first rework after a fork read as first-time work — and the
            # turns just after a fork are the ones the fork exists to watch.
            self._revisions.observe(turn.index, calls)
            for message in turn.outcome.injected:
                text = _content_text(message, 4000)
                self._concerns.append(text)
                self._recharge(text)
        self._tagged = len(self._turns)
        self.turn = len(self._turns)
        self._restore_tags()
        if self._turns:
            logger.info(
                "policy_engine: restored {} turn(s), {} concern(s), {} tag(s), "
                "{}/{} of the budget already spent",
                len(self._turns),
                len(self._concerns),
                len(self._active_tags),
                self.triggers.budget.spent if self.triggers else 0,
                self.triggers.budget.total if self.triggers else 0,
            )

    def _recharge(self, message: str) -> None:
        """Charge the budget for one intervention the source run delivered.

        The pool and each item's own fire limit are memory, so a fork that did
        not rebuild them started on a fresh full allowance and let an item with
        ``max_fires: 1`` say the same thing a second time — to an agent that
        had already answered it. What went out is recoverable because every
        check carries the item's own words.

        A finding is in the reviewer's words rather than an item's, so it
        cannot name the item that paid for it. It is still charged to the pool:
        the agent was interrupted, and the pool is what rations that.
        """
        if self.triggers is None:
            return
        for item_id, item in self.triggers.items.items():
            if item.check and item.check in message:
                self.triggers.budget.spend(item_id)
                return
        if _FINISH_REMINDER_MARK and _FINISH_REMINDER_MARK in message:
            self.triggers.budget.spend(_FINISH_REMINDER)
            return
        self.triggers.budget.spend(_RESTORED_INTERVENTION)

    def _restore_tags(self) -> None:
        """Tags carry down the fork chain: a fork is the same run under a new id.

        The chain rather than the root: A→B→C reaches A and C by root alone,
        and everything B learned in between belongs to C as much as A's does.
        Sessions that were never forked have a chain of one, and the root stays
        in the seed set for runs recorded before the source was written down.
        """
        if self._pg is None:
            return
        schema = self._schema()
        ids = [self.session_id, self.api.ctx.root_session_id]
        if schema:
            rows = self._pg.query(
                "WITH RECURSIVE ancestry(id, source_id) AS ("
                f"  SELECT id, meta_json->'config'->>'fork_source_session_id' "
                f"  FROM {schema}.agentm_trajectory_sessions "
                "   WHERE id = %(session_id)s"
                "  UNION ALL"
                f"  SELECT s.id, s.meta_json->'config'->>'fork_source_session_id' "
                f"  FROM {schema}.agentm_trajectory_sessions s "
                "   JOIN ancestry a ON s.id = a.source_id"
                ") SELECT id FROM ancestry"
            )
            ids.extend(str(row[0]) for row in rows if row[0])
        rows = self._pg.query(
            "SELECT DISTINCT unnest(tags) FROM policy.turn_annotations "
            "WHERE session_id = ANY(%(ids)s)",
            {"ids": sorted(set(ids))},
        )
        self._active_tags.update(str(row[0]) for row in rows if row[0])

    def _schema(self) -> str:
        """The trajectory schema this run writes to, or empty when unset."""
        return self.config.trajectory_schema or os.environ.get(
            "AGENTM_TRAJECTORY_SCHEMA", ""
        )

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
        system = tagger_system_prompt()
        if system is None:
            if not self._provider_missing_logged:
                self._provider_missing_logged = True
                logger.warning("policy_engine: no vocabulary; tagging off")
            return None
        logger.info("policy_engine: tagging on provider {}", provider.name)
        self._tagger = TaggerConversation(
            session_id=self.session_id,
            stream_fn=provider.stream_fn,
            model=provider.model,
            system=system,
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
