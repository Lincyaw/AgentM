# code-health: ignore-file[AM025] -- vendor LLM adapters normalize untyped provider SDK payloads
"""Durable LLM-backed context compaction policy.

Compacted prefixes become structured, turn-addressed checkpoints.  Original
user messages remain verbatim in projected context, while assistant/tool
history is recoverable through the ``[Turn N]`` references consumed by the
``read_history`` atom.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import cast

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.compaction import (
    CompactionPublisher,
    CompactionRequest,
    CompactionResult,
    CompactionSourceAnchor,
    ContextCompactionService,
    ProjectionReport,
    SessionCompactor,
    TurnRange,
)
from agentm.core.abi.context import (
    BindableContextPolicy,
    ContextTransformCancelled,
    PolicyContext,
    turn_to_messages,
)
from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    JsonValue,
    OpaqueThinkingBlock,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    UserMessage,
    synthetic_user_message,
    thaw_json,
)
from agentm.core.abi.resource import ResourceRef, ResourceStore
from agentm.core.abi.roles import (
    COMPACTION_PUBLISHER_SERVICE,
    CONTEXT_COMPACTION_SERVICE,
    RESOURCE_STORE_SERVICE,
    SESSION_COMPACTOR_SERVICE,
    TRAJECTORY_STORE_SERVICE,
)
from agentm.core.abi.session_api import AgentSessionConfig, AtomAPI, LoopConfig
from agentm.core.abi.store import (
    SessionMeta,
    TrajectoryCompactionCommit,
    TrajectoryNodeQuery,
    TrajectoryStore,
)
from agentm.core.abi.trajectory import (
    ContentReplacementState,
    TrajectoryHead,
    TrajectoryHeadAdvance,
    TrajectoryNode,
    Turn,
)
from agentm.core.abi.trigger import CompactTrigger, TriggerRenderer
from agentm.core.lib.async_cancel import (
    OperationCancelledBySignal,
    await_known_outcome,
)
from agentm.core.lib.tokens import truncate_text_tokens
from agentm.extensions import ExtensionManifest


_SUMMARY_SYSTEM_PROMPT = (
    "You are a context compaction assistant for an AI agent session. "
    "Read the supplied conversation transcript and produce a structured "
    "checkpoint summary that contains everything a successor agent needs to "
    "continue the work.\n\n"
    "Do NOT continue the conversation. Do NOT answer questions from the "
    "conversation. ONLY output the structured summary."
)

_SUMMARY_PROMPT = """\
The conversation above is an old prefix that will be replaced by your
checkpoint. Before writing, silently:
- resolve contradictions by preferring the most recent reliable source and
  noting the conflict;
- order events chronologically, treating the latest state as authoritative;
- avoid invention and mark unverified information as UNVERIFIED.

Use this structure:

## Session Identity
Record visible session metadata.

## Goal
State the user's goal and acceptance criteria. Preserve structured goal
conditions and verification methods verbatim when present.

## Task Specification
The original user messages from the compacted prefix remain visible verbatim
beside this checkpoint. Do not reproduce or paraphrase them. Cite the turns
that contain task specifications and record only derived constraints that are
not obvious from the user text.

## Constraints & Preferences
Preserve requirements, operational boundaries, rejected approaches, and why
they were rejected. Write "(none)" when empty.

## Progress
### Done
- [x] Completed work, with exact locations and outcomes

### In Progress
- [ ] Current work

### Blocked
- Current blockers, if any

## Key Decisions
- **Decision**: rationale, alternatives considered, and why rejected

## Files & Artifacts
### Read
- `path` — purpose and key takeaway

### Modified
- `path` — exact change and relevant identifiers

### Created
- `path` — purpose and contents

## Tool Trace Summary
Preserve significant tool names, purposes, key inputs, conclusions, paths,
IDs, error codes, success/failure, and impact. Drop redundant logs, verbose
terminal output, and repeated search results.

## Errors & Debugging
Preserve exact errors, stack traces, failed approaches, successful fixes,
commands, exit codes, and test outcomes.

## Next Steps
1. Ordered next actions

## Recovery Pointers
Point to intentionally omitted detail using `[Turn N]` and the `read_history`
tool, or name files that should be re-read.

Rules:
- Cite originating turns for specific work using `[Turn N]`.
- Preserve exact paths, function names, commands, identifiers, and factual
  values; specificity matters more than prose.
- Do not pad sections. Write "(none)" when a section has no content.
- Do not add commentary outside the checkpoint."""

_UPDATE_SUMMARY_PROMPT = """\
The conversation above contains NEW turns to merge into the checkpoint in
<previous-summary> tags. Silently resolve contradictions and superseded state:
prefer newer reliable information and record meaningful changes.

Rewrite the checkpoint using its existing section structure:
- original user messages remain visible verbatim, so do not reproduce or
  paraphrase them;
- preserve all still-relevant information from the previous checkpoint;
- add new progress, decisions, files, errors, and tool outcomes;
- move completed work from In Progress to Done and remove resolved blockers;
- update Next Steps to reflect the latest state;
- preserve Session Identity, exact identifiers, and existing `[Turn N]`
  citations; cite newly incorporated work with its turn marker;
- do not repeat obsolete or early detail merely because it appeared in the
  previous checkpoint;
- point to `read_history` when exact historical detail can be recovered there.

Output only the complete rewritten checkpoint."""


class LlmCompactionConfig(BaseModel):
    """Configuration for durable context summarization."""

    model_config = ConfigDict(extra="forbid")

    keep_last_turns: int = Field(ge=0)
    tool_result_max_tokens: int = Field(
        default=8_000,
        gt=0,
        description="Per-tool-result cap when rendering compaction input.",
    )
    state_key: str = Field(default="llm_compaction", min_length=1)
    summary_system_prompt: str = Field(
        default=_SUMMARY_SYSTEM_PROMPT,
        min_length=1,
    )
    summary_prompt: str = Field(
        default=_SUMMARY_PROMPT,
        min_length=1,
    )
    update_summary_prompt: str = Field(
        default=_UPDATE_SUMMARY_PROMPT,
        min_length=1,
    )
    custom_instructions: str | None = Field(
        default=None,
        min_length=1,
        description="Optional scenario-specific focus appended to summary prompts.",
    )


MANIFEST = ExtensionManifest(
    name="llm_compaction",
    description="Compact old context into durable provider-generated summaries.",
    registers=(
        "context_policy:llm_compaction",
        f"service:{CONTEXT_COMPACTION_SERVICE}",
        f"service:{SESSION_COMPACTOR_SERVICE}",
        f"service:{COMPACTION_PUBLISHER_SERVICE}",
    ),
    config_schema=LlmCompactionConfig,
    requires=(
        "service:resource_store",
        "service:trajectory_store",
    ),
    priority=AtomInstallPriority.CONTEXT,
)


@dataclass(frozen=True, slots=True)
class _ActiveSummary:
    covered_position: int
    text: str
    message: AgentMessage
    ref: ResourceRef
    compaction_session_id: str | None


class LlmCompactionPolicy(BindableContextPolicy):
    """Replace an old committed prefix with one durable summary message."""

    def __init__(self, api: AtomAPI, config: LlmCompactionConfig) -> None:
        self._api = api
        self._config = config
        self._session_id = ""
        self._parent_session_id: str | None = None
        self._resource_store: ResourceStore | None = None
        self._trajectory_store: TrajectoryStore | None = None
        self._renderers: dict[str, TriggerRenderer] = {}
        self._last_report = ProjectionReport(
            metadata={"policy": "llm_compaction", "decision": "not_run"}
        )

    def bind(self, ctx: PolicyContext) -> None:
        self._session_id = ctx.session_id
        self._parent_session_id = ctx.parent_session_id
        services = ctx.services or {}
        resource_candidate = services.get(RESOURCE_STORE_SERVICE)
        if isinstance(resource_candidate, ResourceStore):
            self._resource_store = resource_candidate
        trajectory_candidate = services.get(TRAJECTORY_STORE_SERVICE)
        if isinstance(trajectory_candidate, TrajectoryStore):
            self._trajectory_store = trajectory_candidate
        self._renderers = dict(ctx.trigger_renderers or {})

    async def transform(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
    ) -> list[AgentMessage]:
        return await self._transform(messages, turns, signal=None, force=False)

    async def transform_with_signal(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
        *,
        signal: CancelSignal,
    ) -> list[AgentMessage]:
        return await self._transform(messages, turns, signal=signal, force=False)

    async def compact(
        self,
        turns: Sequence[Turn],
        *,
        signal: CancelSignal | None = None,
    ) -> ProjectionReport:
        """Force compaction independently of the configured threshold."""

        if not turns:
            raise ValueError("cannot compact a session with no committed turns")
        if signal is not None and signal.is_set():
            raise ContextTransformCancelled
        try:
            await self._transform(
                self._render_turns(turns),
                turns,
                signal=signal,
                force=True,
            )
        except BaseException as exc:
            if signal is not None and signal.is_set():
                raise ContextTransformCancelled from exc
            raise
        return self.explain()

    async def _transform(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
        *,
        signal: CancelSignal | None,
        force: bool,
    ) -> list[AgentMessage]:
        if not turns:
            return messages

        state = await self._load_state(turns)
        active = await self._active_summary(state, turns)

        if active is not None:
            tail = self._render_turns(turns[active.covered_position + 1 :])
            self._require_suffix(messages, tail)

        if not force:
            if active is not None:
                self._record_report(
                    turns,
                    active.covered_position,
                    active.ref,
                    decision="reuse",
                    compaction_session_id=active.compaction_session_id,
                )
                return self._projected_messages(
                    active.message,
                    turns,
                    active.covered_position,
                )
            self._last_report = ProjectionReport(
                kept=_turn_range(turns),
                metadata={
                    "policy": "llm_compaction",
                    "decision": "pass_through",
                },
            )
            return messages

        previous_covered = active.covered_position if active is not None else -1
        if active is not None and previous_covered >= len(turns) - 1:
            self._record_report(
                turns,
                previous_covered,
                active.ref,
                decision="reuse",
                compaction_session_id=active.compaction_session_id,
            )
            return self._projected_messages(
                active.message,
                turns,
                previous_covered,
            )

        self._require_dependencies()
        rendered = self._render_turns(turns)
        if rendered != messages:
            raise RuntimeError(
                "llm_compaction requires an upstream projection that preserves "
                "the committed turn sequence"
            )

        target = max(
            previous_covered + 1,
            len(turns) - self._config.keep_last_turns - 1,
            0,
        )

        covered_turn = turns[target]
        compactor, publisher = self._require_compaction_ports()
        result = await compactor.compact(
            CompactionRequest(
                source_session_id=self._session_id,
                through_turn_id=covered_turn.id,
                start_after_turn_id=(
                    turns[active.covered_position].id if active is not None else None
                ),
                previous_summary=(active.text if active is not None else None),
                options=self._config.model_dump(mode="json"),
            ),
            signal=signal,
        )
        if (
            result.source_session_id != self._session_id
            or result.covered_through_turn_id != covered_turn.id
        ):
            raise RuntimeError(
                "session compactor returned an artifact for the wrong source range"
            )
        published = await publisher.publish(result, signal=signal)
        if published.resource_ref is None:
            raise RuntimeError("compaction publisher returned no resource_ref")
        ref = ResourceRef.parse(published.resource_ref)
        summary_text = published.summary
        compaction_session_id = _compaction_session_id(published)

        self._record_report(
            turns,
            target,
            ref,
            decision="compact",
            compaction_session_id=compaction_session_id,
        )
        return self._projected_messages(
            _summary_message(
                summary_text,
                ref=ref,
                state_key=self._config.state_key,
                covered_turn_id=covered_turn.id,
                compaction_session_id=compaction_session_id,
            ),
            turns,
            target,
        )

    def explain(self) -> ProjectionReport:
        return self._last_report

    async def _load_state(
        self,
        turns: Sequence[Turn],
    ) -> ContentReplacementState | None:
        if self._trajectory_store is None or not self._session_id:
            return None
        state = await asyncio.to_thread(
            self._trajectory_store.load_content_replacement_state,
            self._session_id,
            self._config.state_key,
        )
        if state is not None or self._parent_session_id is None:
            return state

        head = await asyncio.to_thread(
            self._trajectory_store.get_head,
            self._session_id,
        )
        if head is None or head.logical_parent_id is None:
            return None
        parent_state = await asyncio.to_thread(
            self._trajectory_store.load_content_replacement_state,
            self._parent_session_id,
            self._config.state_key,
        )
        if (
            parent_state is None
            or parent_state.leaf_node_id != head.logical_parent_id
            or _covered_position(parent_state, turns) is None
        ):
            return None
        return await await_known_outcome(
            asyncio.to_thread(
                self._trajectory_store.clone_content_replacement_state,
                source_session_id=self._parent_session_id,
                target_session_id=self._session_id,
                state_key=self._config.state_key,
                target_leaf_id=head.node_id or head.logical_parent_id,
            )
        )

    async def _active_summary(
        self,
        state: ContentReplacementState | None,
        turns: Sequence[Turn],
    ) -> _ActiveSummary | None:
        if state is None:
            return None
        covered_position = _covered_position(state, turns)
        raw_ref = state.metadata.get("active_summary_ref")
        if covered_position is None or not isinstance(raw_ref, str):
            raise RuntimeError("llm_compaction state has no valid active summary")
        resource_store = self._resource_store
        if resource_store is None:
            raise RuntimeError("llm_compaction requires a registered ResourceStore")
        ref = ResourceRef.parse(raw_ref)
        if not await resource_store.exists_ref(ref):
            raise RuntimeError(
                f"llm_compaction summary resource is missing: {ref.uri()}"
            )
        summary_bytes = await resource_store.read_ref(ref)
        expected_sha256 = state.metadata.get("summary_sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise RuntimeError(
                "llm_compaction state has no summary_sha256 integrity record"
            )
        actual_sha256 = hashlib.sha256(summary_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "llm_compaction summary integrity check failed: "
                f"{actual_sha256} != {expected_sha256}"
            )
        summary_text = summary_bytes.decode("utf-8")
        raw_compaction_session_id = state.metadata.get("compaction_session_id")
        compaction_session_id = (
            raw_compaction_session_id
            if isinstance(raw_compaction_session_id, str) and raw_compaction_session_id
            else None
        )
        return _ActiveSummary(
            covered_position=covered_position,
            text=summary_text,
            message=_summary_message(
                summary_text,
                ref=ref,
                state_key=self._config.state_key,
                covered_turn_id=turns[covered_position].id,
                compaction_session_id=compaction_session_id,
            ),
            ref=ref,
            compaction_session_id=compaction_session_id,
        )

    def _require_dependencies(
        self,
    ) -> None:
        if self._resource_store is None:
            raise RuntimeError("llm_compaction requires a registered ResourceStore")
        if self._trajectory_store is None:
            raise RuntimeError("llm_compaction requires a TrajectoryStore")

    def _require_compaction_ports(
        self,
    ) -> tuple[SessionCompactor, CompactionPublisher]:
        compactor = self._api.services.get(
            SESSION_COMPACTOR_SERVICE,
            cast(type[SessionCompactor], SessionCompactor),
        )
        if compactor is None:
            raise RuntimeError("llm_compaction requires a registered SessionCompactor")
        publisher = self._api.services.get(
            COMPACTION_PUBLISHER_SERVICE,
            cast(type[CompactionPublisher], CompactionPublisher),
        )
        if publisher is None:
            raise RuntimeError(
                "llm_compaction requires a registered CompactionPublisher"
            )
        return compactor, publisher

    def _render_turns(self, turns: Sequence[Turn]) -> list[AgentMessage]:
        return [
            message
            for turn in turns
            for message in turn_to_messages(turn, self._renderers)
        ]

    def _projected_messages(
        self,
        summary_message: AgentMessage,
        turns: Sequence[Turn],
        covered_position: int,
    ) -> list[AgentMessage]:
        return [
            summary_message,
            *self._preserved_user_messages(turns[: covered_position + 1]),
            *self._render_turns(turns[covered_position + 1 :]),
        ]

    def _preserved_user_messages(
        self,
        turns: Sequence[Turn],
    ) -> list[UserMessage]:
        return [
            message
            for message in self._render_turns(turns)
            if isinstance(message, UserMessage) and not message.meta.synthetic
        ]

    @staticmethod
    def _require_suffix(
        messages: Sequence[AgentMessage],
        suffix: Sequence[AgentMessage],
    ) -> None:
        if suffix and list(messages[-len(suffix) :]) != list(suffix):
            raise RuntimeError(
                "llm_compaction cannot compose with an upstream projection "
                "that changes the uncompacted turn suffix"
            )

    def _record_report(
        self,
        turns: Sequence[Turn],
        covered_position: int,
        ref: ResourceRef,
        *,
        decision: str,
        compaction_session_id: str | None,
    ) -> None:
        self._last_report = ProjectionReport(
            summarized=(
                TurnRange(
                    start=turns[0].index,
                    end=turns[covered_position].index,
                ),
            ),
            kept=_turn_range(turns[covered_position + 1 :]),
            content_refs=(ref.uri(),),
            synthetic_message_count=1,
            metadata={
                "policy": "llm_compaction",
                "decision": decision,
                "covered_through_turn_id": turns[covered_position].id,
                "preserved_user_message_count": len(
                    self._preserved_user_messages(turns[: covered_position + 1])
                ),
                **(
                    {"compaction_session_id": compaction_session_id}
                    if compaction_session_id is not None
                    else {}
                ),
            },
        )


class LlmCompactionService:
    """Coalesce requests and wake the driver without interrupting active work."""

    __slots__ = ("_api", "_pending", "_policy")

    def __init__(self, api: AtomAPI, policy: LlmCompactionPolicy) -> None:
        self._api = api
        self._policy = policy
        self._pending = False

    @property
    def pending(self) -> bool:
        return self._pending

    def request(self) -> None:
        if self._pending:
            return
        self._api.push_trigger(
            CompactTrigger(),
            priority="next",
            origin="compaction",
            mode="compact",
            is_meta=True,
        )
        self._pending = True

    async def execute(
        self,
        turns: Sequence[Turn],
        *,
        signal: CancelSignal | None = None,
    ) -> ProjectionReport | None:
        if not self._pending:
            return None
        self._pending = False
        return await self._policy.compact(turns, signal=signal)


def install(api: AtomAPI, config: LlmCompactionConfig) -> None:
    policy = LlmCompactionPolicy(api, config)
    api.register_context_policy(policy, priority=400)
    api.services.register(
        CONTEXT_COMPACTION_SERVICE,
        LlmCompactionService(api, policy),
        ContextCompactionService,
        scope="session",
    )

    store = api.services.get(
        TRAJECTORY_STORE_SERVICE,
        cast(type[TrajectoryStore], TrajectoryStore),
    )
    resource_store = api.services.get(
        RESOURCE_STORE_SERVICE,
        cast(type[ResourceStore], ResourceStore),
    )
    if store is not None and not api.services.has(SESSION_COMPACTOR_SERVICE):
        api.services.register(
            SESSION_COMPACTOR_SERVICE,
            AgentSessionCompactor(store=store),
            SessionCompactor,
            scope="tree",
        )
    if (
        store is not None
        and resource_store is not None
        and not api.services.has(COMPACTION_PUBLISHER_SERVICE)
    ):
        api.services.register(
            COMPACTION_PUBLISHER_SERVICE,
            TrajectoryCompactionPublisher(store=store, resource_store=resource_store),
            CompactionPublisher,
            scope="tree",
        )


# ---------------------------------------------------------------------------
# SessionCompactor / CompactionPublisher implementations
# ---------------------------------------------------------------------------


class AgentSessionCompactor:
    """Generate a summary artifact via a one-turn child session."""

    __slots__ = ("_store",)

    def __init__(self, *, store: TrajectoryStore) -> None:
        self._store = store

    async def compact(
        self,
        request: CompactionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> CompactionResult:
        if signal is not None and signal.is_set():
            raise OperationCancelledBySignal
        if request.strategy != "llm_structured_checkpoint":
            raise ValueError(
                "AgentSessionCompactor only supports the "
                "'llm_structured_checkpoint' strategy"
            )
        return await self._compact_with_store(request, store=self._store, signal=signal)

    async def _compact_with_store(
        self,
        request: CompactionRequest,
        *,
        store: TrajectoryStore,
        signal: CancelSignal | None,
    ) -> CompactionResult:
        meta, turns, source_head, source_nodes = await _load_source_snapshot(
            store,
            session_id=request.source_session_id,
        )
        if not turns:
            raise ValueError("cannot compact a session with no committed turns")
        strategy_config = _strategy_config(request.options)
        start, target = _selected_positions(
            turns,
            request=request,
            keep_last_turns=strategy_config.keep_last_turns,
        )
        selected_turns = turns[start : target + 1]

        root_session_id = meta.config.get("root_session_id")
        if not isinstance(root_session_id, str) or not root_session_id:
            root_session_id = request.source_session_id

        from agentm.sdk import AgentSession

        child_config = AgentSessionConfig(
            scenario="empty",
            extensions=[],
            system=strategy_config.summary_system_prompt,
            trajectory_store=store,
            purpose="context_compaction",
            loop_config=LoopConfig(max_turns=1, max_tool_calls=0),
            session_id=None,
            root_session_id=root_session_id,
            parent_session_id=request.source_session_id,
            cancel_signal=signal,
            parent_cancellation="independent",
        )

        child = await AgentSession.create(child_config)
        try:
            model = child.model
            if model is None:
                raise RuntimeError(
                    f"compaction session {child.session_id} has no active provider"
                )
            messages_by_turn = _persisted_messages_by_turn(
                source_nodes,
                turns=selected_turns,
            )
            conversation = _serialize_compaction_input(
                selected_turns,
                messages_by_turn=messages_by_turn,
                config=strategy_config,
                model_id=model.id,
            )
            prompt = _summary_prompt(
                source_session_id=request.source_session_id,
                source_parent_session_id=meta.parent_id,
                conversation=conversation,
                previous_summary=request.previous_summary,
                config=strategy_config,
            )
            messages = await child.run(prompt)
        finally:
            await await_known_outcome(child.shutdown())

        summary = _final_assistant_text(messages)
        if not summary:
            raise RuntimeError(
                f"compaction AgentSession returned an empty summary: {child.session_id}"
            )
        covered_turn = turns[target]
        return CompactionResult(
            source=CompactionSourceAnchor(
                head=source_head,
                last_turn_id=turns[-1].id,
                last_turn_index=turns[-1].index,
            ),
            covered=TurnRange(start=turns[0].index, end=covered_turn.index),
            covered_through_turn_id=covered_turn.id,
            summary=summary,
            producer_ref=f"session:{child.session_id}",
            metadata={
                "strategy": request.strategy,
                "state_key": strategy_config.state_key,
                "compaction_session_id": child.session_id,
                "covered_through_turn_index": covered_turn.index,
                "source_turn_count": len(turns),
                "source_start_turn_index": selected_turns[0].index,
            },
        )


class TrajectoryCompactionPublisher:
    """Atomically adopt an immutable artifact at its exact source head."""

    __slots__ = ("_resource_store", "_store")

    def __init__(
        self,
        *,
        store: TrajectoryStore,
        resource_store: ResourceStore,
    ) -> None:
        self._store = store
        self._resource_store = resource_store

    async def publish(
        self,
        result: CompactionResult,
        *,
        signal: CancelSignal | None = None,
    ) -> CompactionResult:
        if signal is not None and signal.is_set():
            raise OperationCancelledBySignal
        state_key = result.metadata.get("state_key")
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("compaction result metadata has no state_key")

        _, turns = await asyncio.to_thread(
            self._store.load,
            result.source_session_id,
        )
        if not turns:
            raise RuntimeError(
                f"session {result.source_session_id}: "
                "cannot publish compaction without committed turns"
            )
        source = result.source
        latest_turn = turns[-1]
        if (
            latest_turn.id != source.last_turn_id
            or latest_turn.index != source.last_turn_index
        ):
            raise RuntimeError(
                "refusing to publish compaction after source history advanced"
            )
        head = await asyncio.to_thread(
            self._store.get_head,
            result.source_session_id,
            head_id=source.head.head_id,
            branch_id=source.head.branch_id,
            agent_id=source.head.agent_id,
            is_sidechain=source.head.is_sidechain,
        )
        if head != source.head:
            raise RuntimeError(
                "refusing to publish compaction after source head changed"
            )
        covered_position = next(
            (
                position
                for position, turn in enumerate(turns)
                if turn.id == result.covered_through_turn_id
            ),
            None,
        )
        if covered_position is None:
            raise ValueError(
                "compaction result is not anchored to committed source history"
            )
        existing = await asyncio.to_thread(
            self._store.load_content_replacement_state,
            result.source_session_id,
            state_key,
        )
        existing_position = (
            _state_covered_position(existing, turns) if existing is not None else None
        )
        if existing_position is not None and existing_position > covered_position:
            raise RuntimeError(
                "refusing to replace newer compaction state with an older artifact"
            )

        summary_bytes = result.summary.encode("utf-8")
        ref = _summary_ref(
            state_key=state_key,
            session_id=result.source_session_id,
            turn_id=result.covered_through_turn_id,
            summary=result.summary,
        )
        await await_known_outcome(
            self._resource_store.write_ref(
                ref,
                summary_bytes,
                rationale=(
                    "Persist store-driven context summary generated by "
                    f"{result.producer_ref}"
                ),
            )
        )
        latest_nodes = await asyncio.to_thread(
            self._store.query_nodes,
            TrajectoryNodeQuery(
                session_id=result.source_session_id,
                sort="desc",
                limit=1,
            ),
        )

        resource_uri = ref.uri()
        timestamp = time.time()
        boundary = TrajectoryNode(
            id=_boundary_id(
                state_key=state_key,
                session_id=result.source_session_id,
                turn_id=source.last_turn_id,
                resource_uri=resource_uri,
                source_leaf_id=head.node_id or head.logical_parent_id,
            ),
            session_id=result.source_session_id,
            seq=latest_nodes[0].seq + 1 if latest_nodes else 0,
            kind="compact_boundary",
            root_session_id=head.root_session_id,
            parent_session_id=head.parent_session_id,
            branch_id=head.branch_id,
            head_id=head.head_id,
            role="control",
            logical_parent_id=head.node_id or head.logical_parent_id,
            turn_id=source.last_turn_id,
            turn_index=source.last_turn_index,
            agent_id=head.agent_id,
            is_sidechain=head.is_sidechain,
            content_ref=resource_uri,
            visibility="replay_only",
            payload={
                "state_key": state_key,
                "covered_through_turn_id": result.covered_through_turn_id,
                "covered_through_turn_index": turns[covered_position].index,
                "producer_ref": result.producer_ref,
            },
            timestamp=timestamp,
        )
        replacements = dict(existing.replacements) if existing is not None else {}
        replacements[f"through:{result.covered_through_turn_id}"] = resource_uri
        metadata: dict[str, object] = (
            dict(existing.metadata) if existing is not None else {}
        )
        metadata.update(
            {
                "active_summary_ref": resource_uri,
                "covered_through_turn_id": result.covered_through_turn_id,
                "covered_through_turn_index": turns[covered_position].index,
                "producer_ref": result.producer_ref,
                "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            }
        )
        compaction_session_id = result.metadata.get("compaction_session_id")
        if isinstance(compaction_session_id, str) and compaction_session_id:
            metadata["compaction_session_id"] = compaction_session_id
        state = ContentReplacementState(
            state_key=state_key,
            seen_tool_call_ids=(
                existing.seen_tool_call_ids if existing is not None else ()
            ),
            replacements=replacements,
            source_session_id=(
                existing.source_session_id if existing is not None else None
            ),
            source_leaf_id=(existing.source_leaf_id if existing is not None else None),
            leaf_node_id=boundary.id,
            branch_id=head.branch_id,
            head_id=head.head_id,
            metadata=metadata,
        )
        try:
            await await_known_outcome(
                asyncio.to_thread(
                    self._store.commit_compaction,
                    result.source_session_id,
                    TrajectoryCompactionCommit(
                        boundary=boundary,
                        advance_head=TrajectoryHeadAdvance(
                            session_id=result.source_session_id,
                            node_id=boundary.id,
                            seq=boundary.seq,
                            previous_node_id=head.node_id,
                            head_id=head.head_id,
                            branch_id=head.branch_id,
                            root_session_id=head.root_session_id,
                            parent_session_id=head.parent_session_id,
                            agent_id=head.agent_id,
                            is_sidechain=head.is_sidechain,
                            updated_at=timestamp,
                        ),
                        content_replacement_state=state,
                    ),
                )
            )
        except Exception:
            logger.warning(
                "compaction commit failed after resource was persisted; "
                "orphan_resource={} session={} expected_head_node={} "
                "producer={}",
                resource_uri,
                result.source_session_id,
                head.node_id,
                result.producer_ref,
            )
            raise
        return replace(result, resource_ref=resource_uri)


# ---------------------------------------------------------------------------
# Compaction helpers
# ---------------------------------------------------------------------------


def _strategy_config(options: Mapping[str, JsonValue]) -> LlmCompactionConfig:
    raw = thaw_json(options)
    if not isinstance(raw, Mapping):  # code-health: ignore[AM025] -- ABI JSON boundary
        raise TypeError("llm compaction options must be an object")
    return LlmCompactionConfig.model_validate(raw)


def _selected_positions(
    turns: Sequence[Turn],
    *,
    request: CompactionRequest,
    keep_last_turns: int,
) -> tuple[int, int]:
    target = (
        _turn_position(turns, request.through_turn_id)
        if request.through_turn_id is not None
        else max(len(turns) - keep_last_turns - 1, 0)
    )
    start = (
        _turn_position(turns, request.start_after_turn_id) + 1
        if request.start_after_turn_id is not None
        else 0
    )
    if start > target:
        raise ValueError("compaction range is empty or reversed")
    return start, target


def _turn_position(turns: Sequence[Turn], turn_id: str) -> int:
    position = next(
        (index for index, turn in enumerate(turns) if turn.id == turn_id),
        None,
    )
    if position is None:
        raise ValueError(f"unknown compaction turn id: {turn_id}")
    return position


async def _load_source_snapshot(
    store: TrajectoryStore,
    *,
    session_id: str,
) -> tuple[SessionMeta, list[Turn], TrajectoryHead, list[TrajectoryNode]]:
    for _attempt in range(2):
        meta, turns = await asyncio.to_thread(store.load, session_id)
        head = await asyncio.to_thread(store.get_head, session_id)
        if head is None:
            raise RuntimeError(
                f"session {session_id}: cannot compact without an active head"
            )
        leaf_node_id = head.node_id or head.logical_parent_id
        nodes = (
            await asyncio.to_thread(
                store.load_chain,
                session_id,
                leaf_node_id,
                include_logical_parent=True,
            )
            if leaf_node_id is not None
            else []
        )
        _, current_turns = await asyncio.to_thread(store.load, session_id)
        current_head = await asyncio.to_thread(store.get_head, session_id)
        if current_head == head and current_turns == turns:
            return meta, turns, head, nodes
    raise RuntimeError(
        f"session {session_id}: source trajectory changed while taking "
        "compaction snapshot"
    )


def _persisted_messages_by_turn(
    nodes: Sequence[TrajectoryNode],
    *,
    turns: Sequence[Turn],
) -> Mapping[str, tuple[AgentMessage, ...]]:
    selected_ids = {turn.id for turn in turns}
    grouped: dict[str, list[AgentMessage]] = {}
    for node in nodes:
        if node.turn_id not in selected_ids or node.message is None:
            continue
        grouped.setdefault(node.turn_id, []).append(node.message)
    return {turn_id: tuple(messages) for turn_id, messages in grouped.items()}


def _boundary_id(
    *,
    state_key: str,
    session_id: str,
    turn_id: str,
    resource_uri: str,
    source_leaf_id: str | None,
) -> str:
    material = "\0".join(
        (state_key, session_id, turn_id, resource_uri, source_leaf_id or "")
    ).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"session:{session_id}:compact:{digest}"


def _serialize_compaction_input(
    turns: Sequence[Turn],
    *,
    messages_by_turn: Mapping[str, tuple[AgentMessage, ...]],
    config: LlmCompactionConfig,
    model_id: str,
) -> str:
    blocks: list[str] = []
    for turn in turns:
        messages = messages_by_turn.get(turn.id, ())
        body = "\n\n".join(
            section
            for message in messages
            for section in _serialize_message_for_summary(
                message,
                tool_result_max_tokens=config.tool_result_max_tokens,
                model_id=model_id,
            )
        )
        blocks.append(f"[Turn {turn.index + 1}]\n{body or '(no persisted messages)'}")
    return "\n\n".join(blocks)


def _summary_prompt(
    *,
    source_session_id: str,
    source_parent_session_id: str | None,
    conversation: str,
    previous_summary: str | None,
    config: LlmCompactionConfig,
) -> str:
    prompt = (
        "<session-metadata>\n"
        f"session_id: {source_session_id}\n"
        f"parent_session_id: {source_parent_session_id or '(none)'}\n"
        "</session-metadata>\n\n"
        f"<conversation>\n{conversation}\n</conversation>\n\n"
    )
    if previous_summary is not None:
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        prompt += config.update_summary_prompt
    else:
        prompt += config.summary_prompt
    if config.custom_instructions is not None:
        prompt += "\n\nAdditional focus:\n" + config.custom_instructions
    return prompt


def _final_assistant_text(messages: Sequence[AgentMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        text = "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        ).strip()
        if text:
            return text
    return ""


def _state_covered_position(
    state: ContentReplacementState,
    turns: Sequence[Turn],
) -> int | None:
    turn_id = state.metadata.get("covered_through_turn_id")
    if not isinstance(turn_id, str):
        return None
    return next(
        (index for index, turn in enumerate(turns) if turn.id == turn_id),
        None,
    )


def _summary_message(
    summary: str,
    *,
    ref: ResourceRef,
    state_key: str,
    covered_turn_id: str,
    compaction_session_id: str | None,
) -> AgentMessage:
    return synthetic_user_message(
        f"<conversation-summary>\n{summary}\n</conversation-summary>",
        kind="conversation_summary",
        origin="compaction",
        tags={
            "content_ref": ref.uri(),
            "content_replacement_state_key": state_key,
            "covered_through_turn_id": covered_turn_id,
            **(
                {"compaction_session_id": compaction_session_id}
                if compaction_session_id is not None
                else {}
            ),
        },
    )


def _compaction_session_id(result: CompactionResult) -> str | None:
    value = result.metadata.get("compaction_session_id")
    if isinstance(value, str) and value:
        return value
    prefix = "session:"
    if result.producer_ref.startswith(prefix):
        return result.producer_ref[len(prefix) :] or None
    return None


def _covered_position(
    state: ContentReplacementState,
    turns: Sequence[Turn],
) -> int | None:
    covered_turn_id = state.metadata.get("covered_through_turn_id")
    if not isinstance(covered_turn_id, str):
        return None
    return next(
        (position for position, turn in enumerate(turns) if turn.id == covered_turn_id),
        None,
    )


def _turn_range(turns: Sequence[Turn]) -> tuple[TurnRange, ...]:
    if not turns:
        return ()
    return (TurnRange(start=turns[0].index, end=turns[-1].index),)


def _serialize_message_for_summary(
    message: AgentMessage,
    *,
    tool_result_max_tokens: int,
    model_id: str,
) -> list[str]:
    if isinstance(message, UserMessage):
        sections: list[str] = []
        text = "\n".join(
            user_block.text
            for user_block in message.content
            if isinstance(user_block, TextContent)
        )
        if text:
            sections.append(f"[User]: {text}")
        sections.extend(
            f"[User image]: mime_type={user_block.mime_type}; "
            f"bytes={len(user_block.data)}"
            for user_block in message.content
            if isinstance(user_block, ImageContent)
        )
        return sections or ["[User]: (empty)"]

    if isinstance(message, AssistantMessage):
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        opaque_thinking: list[str] = []
        tool_calls: list[str] = []
        for assistant_block in message.content:
            if isinstance(assistant_block, TextContent):
                text_parts.append(assistant_block.text)
            elif isinstance(assistant_block, ThinkingBlock):
                thinking_parts.append(assistant_block.text)
            elif isinstance(assistant_block, OpaqueThinkingBlock):
                opaque_thinking.append(assistant_block.provider)
            elif isinstance(assistant_block, ToolCallBlock):
                arguments = json.dumps(
                    thaw_json(assistant_block.arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                tool_calls.append(
                    f"{assistant_block.name}({arguments}) "
                    f"[tool_call_id={assistant_block.id}]"
                )
        sections = []
        if thinking_parts:
            sections.append("[Assistant thinking]: " + "\n".join(thinking_parts))
        if opaque_thinking:
            sections.append(
                "[Assistant opaque thinking]: provider=" + ",".join(opaque_thinking)
            )
        if text_parts:
            sections.append("[Assistant]: " + "\n".join(text_parts))
        if tool_calls:
            sections.append("[Assistant tool calls]: " + "; ".join(tool_calls))
        return sections or ["[Assistant]: (empty)"]

    sections = []
    for result_block in message.content:
        content_parts = [
            part.text for part in result_block.content if isinstance(part, TextContent)
        ]
        content_parts.extend(
            f"[image mime_type={part.mime_type}; bytes={len(part.data)}]"
            for part in result_block.content
            if isinstance(part, ImageContent)
        )
        content = "\n".join(content_parts) or "(empty)"
        truncated = truncate_text_tokens(
            content,
            tool_result_max_tokens,
            model=model_id,
        )
        if truncated.was_truncated:
            content = (
                truncated.text
                + f"\n[... {truncated.truncated_tokens} more tokens truncated]"
            )
        status = "error" if result_block.is_error else "success"
        sections.append(
            f"[Tool result {status}; tool_call_id={result_block.tool_call_id}]: "
            f"{content}"
        )
    return sections or ["[Tool result]: (empty)"]


def _summary_ref(
    *,
    state_key: str,
    session_id: str,
    turn_id: str,
    summary: str,
) -> ResourceRef:
    digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    path = (
        f"{_path_token(state_key)}/{_path_token(session_id)}/"
        f"{_path_token(turn_id)}-{digest[:16]}.txt"
    )
    return ResourceRef(namespace="summary", path=path)


def _path_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


__all__ = [
    "AgentSessionCompactor",
    "LlmCompactionConfig",
    "LlmCompactionPolicy",
    "LlmCompactionService",
    "MANIFEST",
    "TrajectoryCompactionPublisher",
    "install",
]
