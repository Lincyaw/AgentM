# code-health: ignore-file[AM025] -- vendor LLM adapters normalize untyped provider SDK payloads
"""Durable LLM-backed context compaction policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Sequence

from pydantic import BaseModel, Field

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.codec import serialize_message
from agentm.core.abi.compaction import ProjectionReport, TurnRange
from agentm.core.abi.context import (
    BindableContextPolicy,
    PolicyContext,
    turn_to_messages,
)
from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    synthetic_user_message,
)
from agentm.core.abi.resource import ResourceRef, ResourceStore
from agentm.core.abi.roles import (
    RESOURCE_STORE_SERVICE,
    TRAJECTORY_STORE_SERVICE,
)
from agentm.core.abi.session_api import AtomAPI
from agentm.core.abi.store import (
    TrajectoryCompactionCommit,
    TrajectoryNodeQuery,
    TrajectoryStore,
)
from agentm.core.abi.stream import MessageEnd, Model, StreamFn, TextDelta
from agentm.core.abi.trajectory import (
    ContentReplacementState,
    TrajectoryHeadAdvance,
    TrajectoryNode,
    Turn,
)
from agentm.core.abi.trigger import TriggerRenderer
from agentm.core.lib.async_cancel import await_known_outcome
from agentm.core.lib.tokens import count_text_tokens
from agentm.extensions import ExtensionManifest


class LlmCompactionConfig(BaseModel):
    """Configuration for durable context summarization."""

    max_messages: int | None = Field(
        default=None,
        ge=2,
        description="Explicit override; by default the model token budget is used.",
    )
    keep_last_turns: int = Field(default=4, ge=0)
    state_key: str = Field(default="llm_compaction", min_length=1)
    summary_system_prompt: str = (
        "Summarize the supplied conversation prefix for another model that "
        "will continue the same task. Preserve decisions, constraints, user "
        "intent, unresolved work, tool outcomes, identifiers, and exact facts "
        "needed to continue. Do not add commentary or invent information."
    )


MANIFEST = ExtensionManifest(
    name="llm_compaction",
    description="Compact old context into durable provider-generated summaries.",
    registers=("context_policy:llm_compaction",),
    config_schema=LlmCompactionConfig,
    requires=(
        "service:resource_store",
        "service:trajectory_store",
    ),
    priority=AtomInstallPriority.CONTEXT,
)


class LlmCompactionPolicy(BindableContextPolicy):
    """Replace an old committed prefix with one durable summary message."""

    def __init__(self, config: LlmCompactionConfig) -> None:
        self._config = config
        self._session_id = ""
        self._parent_session_id: str | None = None
        self._resource_store: ResourceStore | None = None
        self._trajectory_store: TrajectoryStore | None = None
        self._stream_fn: StreamFn | None = None
        self._model: Model | None = None
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
        self._stream_fn = ctx.stream_fn
        self._model = ctx.model
        self._renderers = dict(ctx.trigger_renderers or {})

    async def transform(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
    ) -> list[AgentMessage]:
        return await self._transform(messages, turns, signal=None)

    async def transform_with_signal(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
        *,
        signal: CancelSignal,
    ) -> list[AgentMessage]:
        return await self._transform(messages, turns, signal=signal)

    async def _transform(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
        *,
        signal: CancelSignal | None,
    ) -> list[AgentMessage]:
        if not turns:
            return messages

        state = await self._load_state(turns)
        active = await self._active_summary(state, turns)
        if active is not None:
            covered_position, summary_message, ref = active
            tail = self._render_turns(turns[covered_position + 1 :])
            self._require_suffix(messages, tail)
            projected = [summary_message, *tail]
            within_budget, budget_metadata = self._within_budget(projected, turns)
            if within_budget:
                self._record_report(
                    turns,
                    covered_position,
                    ref,
                    decision="reuse",
                    budget_metadata=budget_metadata,
                )
                return projected

        within_budget, budget_metadata = self._within_budget(messages, turns)
        if within_budget:
            self._last_report = ProjectionReport(
                kept=_turn_range(turns),
                metadata={
                    "policy": "llm_compaction",
                    "decision": "within_budget",
                    **budget_metadata,
                },
            )
            return messages

        resource_store, trajectory_store, stream_fn, model = (
            self._require_dependencies()
        )
        rendered = self._render_turns(turns)
        if rendered != messages:
            raise RuntimeError(
                "llm_compaction requires an upstream projection that preserves "
                "the committed turn sequence"
            )

        previous_covered = active[0] if active is not None else -1
        target = max(
            previous_covered + 1,
            len(turns) - self._config.keep_last_turns - 1,
            0,
        )
        max_messages = self._config.max_messages
        if max_messages is not None:
            while (
                target < len(turns) - 1
                and 1 + len(self._render_turns(turns[target + 1 :])) > max_messages
            ):
                target += 1

        summary_input: list[AgentMessage]
        if active is None:
            summary_input = self._render_turns(turns[: target + 1])
        else:
            summary_input = [
                active[1],
                *self._render_turns(turns[active[0] + 1 : target + 1]),
            ]
        summary_text = await self._summarize(
            stream_fn=stream_fn,
            model=model,
            messages=summary_input,
            signal=signal,
        )
        covered_turn = turns[target]
        ref = _summary_ref(
            state_key=self._config.state_key,
            session_id=self._session_id,
            turn_id=covered_turn.id,
            summary=summary_text,
        )
        await resource_store.write_ref(
            ref,
            summary_text.encode("utf-8"),
            rationale=(
                "Persist provider-generated context summary through "
                f"turn {covered_turn.id}"
            ),
        )

        head, latest_nodes = await asyncio.gather(
            asyncio.to_thread(
                trajectory_store.get_head,
                self._session_id,
            ),
            asyncio.to_thread(
                trajectory_store.query_nodes,
                TrajectoryNodeQuery(
                    session_id=self._session_id,
                    sort="desc",
                    limit=1,
                ),
            ),
        )
        if head is None:
            raise RuntimeError(
                "llm_compaction cannot persist state without an active trajectory head"
            )
        anchored_turn = turns[-1]
        boundary_id = _boundary_id(
            state_key=self._config.state_key,
            session_id=self._session_id,
            turn_id=anchored_turn.id,
            summary_ref=ref,
        )
        boundary = TrajectoryNode(
            id=boundary_id,
            session_id=self._session_id,
            seq=latest_nodes[0].seq + 1 if latest_nodes else 0,
            kind="compact_boundary",
            root_session_id=head.root_session_id,
            parent_session_id=head.parent_session_id,
            branch_id=head.branch_id,
            head_id=head.head_id,
            role="control",
            logical_parent_id=head.node_id or head.logical_parent_id,
            turn_id=anchored_turn.id,
            turn_index=anchored_turn.index,
            agent_id=head.agent_id,
            is_sidechain=head.is_sidechain,
            content_ref=ref.uri(),
            visibility="replay_only",
            payload={
                "state_key": self._config.state_key,
                "covered_through_turn_id": covered_turn.id,
                "covered_through_turn_index": covered_turn.index,
            },
            timestamp=time.time(),
        )
        replacements = dict(state.replacements) if state is not None else {}
        replacements[f"through:{covered_turn.id}"] = ref.uri()
        persisted = ContentReplacementState(
            state_key=self._config.state_key,
            seen_tool_call_ids=(state.seen_tool_call_ids if state is not None else ()),
            replacements=replacements,
            source_session_id=(state.source_session_id if state is not None else None),
            source_leaf_id=state.source_leaf_id if state is not None else None,
            leaf_node_id=boundary.id,
            branch_id=head.branch_id,
            head_id=head.head_id,
            metadata={
                **(dict(state.metadata) if state is not None else {}),
                "active_summary_ref": ref.uri(),
                "covered_through_turn_id": covered_turn.id,
                "covered_through_turn_index": covered_turn.index,
                "summary_sha256": hashlib.sha256(
                    summary_text.encode("utf-8")
                ).hexdigest(),
            },
        )
        await await_known_outcome(
            asyncio.to_thread(
                trajectory_store.commit_compaction,
                self._session_id,
                TrajectoryCompactionCommit(
                    boundary=boundary,
                    advance_head=TrajectoryHeadAdvance(
                        session_id=self._session_id,
                        node_id=boundary.id,
                        seq=boundary.seq,
                        previous_node_id=head.node_id,
                        head_id=head.head_id,
                        branch_id=head.branch_id,
                        root_session_id=head.root_session_id,
                        parent_session_id=head.parent_session_id,
                        agent_id=head.agent_id,
                        is_sidechain=head.is_sidechain,
                        updated_at=boundary.timestamp,
                    ),
                    content_replacement_state=persisted,
                ),
            )
        )

        self._record_report(
            turns,
            target,
            ref,
            decision="compact",
            budget_metadata=budget_metadata,
        )
        return [
            _summary_message(
                summary_text,
                ref=ref,
                state_key=self._config.state_key,
                covered_turn_id=covered_turn.id,
            ),
            *self._render_turns(turns[target + 1 :]),
        ]

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
    ) -> tuple[int, AgentMessage, ResourceRef] | None:
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
        return (
            covered_position,
            _summary_message(
                summary_text,
                ref=ref,
                state_key=self._config.state_key,
                covered_turn_id=turns[covered_position].id,
            ),
            ref,
        )

    def _require_dependencies(
        self,
    ) -> tuple[ResourceStore, TrajectoryStore, StreamFn, Model]:
        if self._resource_store is None:
            raise RuntimeError("llm_compaction requires a registered ResourceStore")
        if self._trajectory_store is None:
            raise RuntimeError("llm_compaction requires a TrajectoryStore")
        if self._stream_fn is None or self._model is None:
            raise RuntimeError("llm_compaction requires an active provider")
        return (
            self._resource_store,
            self._trajectory_store,
            self._stream_fn,
            self._model,
        )

    def _render_turns(self, turns: Sequence[Turn]) -> list[AgentMessage]:
        return [
            message
            for turn in turns
            for message in turn_to_messages(turn, self._renderers)
        ]

    def _within_budget(
        self,
        messages: Sequence[AgentMessage],
        turns: Sequence[Turn],
    ) -> tuple[bool, dict[str, str | int]]:
        max_messages = self._config.max_messages
        if max_messages is not None:
            message_count = len(messages)
            return (
                message_count <= max_messages,
                {
                    "budget_kind": "messages",
                    "message_count": message_count,
                    "message_limit": max_messages,
                },
            )

        model = self._model
        if model is None:
            raise RuntimeError("llm_compaction requires an active provider")
        input_limit = max(0, model.context_window - model.max_output_tokens)
        estimated_input = self._estimate_next_input_tokens(messages, turns, model)
        return (
            estimated_input <= input_limit,
            {
                "budget_kind": "tokens",
                "estimated_input_tokens": estimated_input,
                "input_limit_tokens": input_limit,
                "remaining_input_tokens": input_limit - estimated_input,
            },
        )

    def _estimate_next_input_tokens(
        self,
        messages: Sequence[AgentMessage],
        turns: Sequence[Turn],
        model: Model,
    ) -> int:
        """Combine observed provider usage with the newly committed completion."""

        projected_estimate = _estimate_message_tokens(messages, model.id)
        latest = turns[-1]
        if latest.meta.total_input_tokens <= 0:
            return projected_estimate

        completion = _turn_completion_messages(latest, self._renderers)
        observed_estimate = latest.meta.total_input_tokens + _estimate_message_tokens(
            completion,
            model.id,
        )
        return max(projected_estimate, observed_estimate)

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

    async def _summarize(
        self,
        *,
        stream_fn: StreamFn,
        model: Model,
        messages: list[AgentMessage],
        signal: CancelSignal | None,
    ) -> str:
        final: AssistantMessage | None = None
        deltas: list[str] = []
        async for event in stream_fn(
            messages=messages,
            model=model,
            tools=[],
            system=self._config.summary_system_prompt,
            signal=signal,
            thinking="off",
        ):
            if isinstance(event, TextDelta):
                deltas.append(event.text)
            elif isinstance(event, MessageEnd):
                final = event.message
        text = (
            "".join(
                block.text for block in final.content if isinstance(block, TextContent)
            )
            if final is not None
            else "".join(deltas)
        ).strip()
        if not text:
            raise RuntimeError("llm_compaction provider returned an empty summary")
        return text

    def _record_report(
        self,
        turns: Sequence[Turn],
        covered_position: int,
        ref: ResourceRef,
        *,
        decision: str,
        budget_metadata: dict[str, str | int],
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
                **budget_metadata,
            },
        )


def install(api: AtomAPI, config: LlmCompactionConfig) -> None:
    api.register_context_policy(LlmCompactionPolicy(config), priority=400)


def _summary_message(
    summary: str,
    *,
    ref: ResourceRef,
    state_key: str,
    covered_turn_id: str,
) -> AgentMessage:
    return synthetic_user_message(
        f"<conversation-summary>\n{summary}\n</conversation-summary>",
        kind="conversation_summary",
        origin="compaction",
        tags={
            "content_ref": ref.uri(),
            "content_replacement_state_key": state_key,
            "covered_through_turn_id": covered_turn_id,
        },
    )


def _turn_completion_messages(
    turn: Turn,
    renderers: dict[str, TriggerRenderer],
) -> list[AgentMessage]:
    rendered = turn_to_messages(turn, renderers)
    if turn.response is None:
        return list(turn.outcome.injected)
    response_position = next(
        (index for index, message in enumerate(rendered) if message is turn.response),
        len(rendered),
    )
    return rendered[response_position:]


def _estimate_message_tokens(
    messages: Sequence[AgentMessage],
    model_id: str,
) -> int:
    provider_shape = []
    for message in messages:
        encoded = serialize_message(message)
        provider_shape.append(
            {
                "role": encoded["role"],
                "content": encoded["content"],
            }
        )
    return count_text_tokens(
        json.dumps(
            provider_shape,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        model=model_id,
    )


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


def _boundary_id(
    *,
    state_key: str,
    session_id: str,
    turn_id: str,
    summary_ref: ResourceRef,
) -> str:
    material = "\0".join((state_key, session_id, turn_id, summary_ref.uri())).encode(
        "utf-8"
    )
    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"session:{session_id}:compact:{digest}"


def _path_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _turn_range(turns: Sequence[Turn]) -> tuple[TurnRange, ...]:
    if not turns:
        return ()
    return (TurnRange(start=turns[0].index, end=turns[-1].index),)


__all__ = [
    "LlmCompactionConfig",
    "LlmCompactionPolicy",
    "MANIFEST",
    "install",
]
