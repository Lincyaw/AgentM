# code-health: ignore-file[AM025] -- vendor LLM adapters normalize untyped provider SDK payloads
"""Durable LLM-backed context compaction policy.

Compacted prefixes become structured, turn-addressed checkpoints.  Original
user messages remain verbatim in projected context, while assistant/tool
history is recoverable through the ``[Turn N]`` references consumed by the
``read_history`` atom.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.compaction import (
    CompactionPublisher,
    CompactionRequest,
    CompactionResult,
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
    UserMessage,
    synthetic_user_message,
)
from agentm.core.abi.resource import ResourceRef, ResourceStore
from agentm.core.abi.roles import (
    COMPACTION_PUBLISHER_SERVICE,
    CONTEXT_COMPACTION_SERVICE,
    RESOURCE_STORE_SERVICE,
    SESSION_COMPACTOR_SERVICE,
    TRAJECTORY_STORE_SERVICE,
)
from agentm.core.abi.session_api import AtomAPI
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import ContentReplacementState, Turn
from agentm.core.abi.trigger import CompactTrigger, TriggerRenderer
from agentm.core.lib.async_cancel import await_known_outcome
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
            raise RuntimeError(
                "llm_compaction requires a host-provided SessionCompactor"
            )
        publisher = self._api.services.get(
            COMPACTION_PUBLISHER_SERVICE,
            cast(type[CompactionPublisher], CompactionPublisher),
        )
        if publisher is None:
            raise RuntimeError(
                "llm_compaction requires a host-provided CompactionPublisher"
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


__all__ = [
    "LlmCompactionConfig",
    "LlmCompactionPolicy",
    "LlmCompactionService",
    "MANIFEST",
    "install",
]
