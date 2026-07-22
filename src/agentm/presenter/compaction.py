"""Presenter-side, store-driven session compaction orchestration."""

# code-health: ignore-file[AM025] -- validates persisted metadata/message unions

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import time

from loguru import logger

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.compaction import (
    CompactionRequest,
    CompactionResult,
    CompactionSourceAnchor,
    TurnRange,
)
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    JsonValue,
    TextContent,
    thaw_json,
)
from agentm.core.abi.resource import ResourceRef, ResourceStore
from agentm.core.abi.session_api import AgentSessionConfig, LoopConfig
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
from agentm.core.lib.async_cancel import (
    OperationCancelledBySignal,
    await_known_outcome,
)
from agentm.core.lib.tokens import truncate_text_tokens
from agentm.extensions.builtin.llm_compaction import LlmCompactionConfig
from agentm.storage.trajectory.resolve import (
    ResolvedTrajectoryStore,
    resolve_trajectory_store_or_create,
)


class AgentSessionCompactor:
    """Default compactor: one persisted source snapshot, one audit session."""

    __slots__ = ("_config",)

    def __init__(self, config: AgentSessionConfig) -> None:
        self._config = config

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

        resolved: ResolvedTrajectoryStore | None = None
        store = self._config.trajectory_store
        if store is None:
            resolved = resolve_trajectory_store_or_create(self._config.cwd or None)
            store = resolved.store
        try:
            return await self._compact_with_store(request, store=store, signal=signal)
        finally:
            if resolved is not None:
                await asyncio.to_thread(resolved.close)

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
        child_config = replace(
            self._config,
            scenario="empty",
            extensions=[],
            extra_extensions=[],
            extra_tools=[],
            atom_config_overrides={},
            system=strategy_config.summary_system_prompt,
            trajectory_store=store,
            environment_operations=None,
            resource_writer=None,
            tool_allowlist=[],
            purpose="context_compaction",
            loop_config=LoopConfig(max_turns=1, max_tool_calls=0),
            experiment=None,
            session_id=None,
            root_session_id=root_session_id,
            parent_session_id=request.source_session_id,
            initial_turns=[],
            cancel_signal=signal,
            parent_cancellation="independent",
            session_compactor=None,
            compaction_publisher=None,
        )

        from agentm.sdk import AgentSession

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


def _serialize_message_for_summary(
    message: AgentMessage,
    *,
    tool_result_max_tokens: int,
    model_id: str,
) -> list[str]:
    from agentm.core.abi.messages import (
        ImageContent,
        OpaqueThinkingBlock,
        ThinkingBlock,
        ToolCallBlock,
        UserMessage,
    )

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
    "TrajectoryCompactionPublisher",
]
