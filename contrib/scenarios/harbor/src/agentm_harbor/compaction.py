# code-health: ignore-file[AM025] -- validates persisted message unions and model output
"""Harbor-specific chronological map-reduce context compaction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from agentm.core.abi import (
    SESSION_COMPACTOR_SERVICE,
    AgentSessionConfig,
    AtomAPI,
    AtomInstallPriority,
    CancelSignal,
    CompactionRequest,
    CompactionResult,
    CompactionSourceAnchor,
    FunctionTool,
    ImageContent,
    LoopConfig,
    SessionCompactor,
    TextContent,
    ThinkingBlock,
    ToolResult,
    ToolTerminate,
    TrajectoryHead,
    TrajectoryNode,
    Turn,
    TurnRange,
    UserMessage,
)
from agentm.core.abi.messages import JsonValue, thaw_json
from agentm.core.abi.store import TrajectoryStore
from agentm.core.lib.async_cancel import OperationCancelledBySignal
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_CHECKPOINT_PREFIX = "# Harbor Recovery Checkpoint\n\n"

_CHUNK_SYSTEM = """\
You compress one contiguous chronological slice of a software-engineering agent
trajectory. Produce faithful continuation memory, not a diagnosis or review.

Keep only information whose omission could make the next agent misunderstand the task,
repeat material work, or lose the current workspace state. Preserve the local sequence
between assistant intent, tool invocation, observed result, and subsequent reaction.
Cite every retained item with its exact [Turn N] marker. Keep exact paths, identifiers,
important invocation arguments, assertions, errors, and observed outputs when they
change the meaning of what happened.

Do not classify events into a taxonomy. Do not infer a root cause, decide which turn was
wrong, propose a fix, or broaden a command result beyond what it actually exercised.
Assistant reasoning is a hypothesis, not an observation. Bash search output only shows
what was searched or seen; it does not establish file or symbol state. Bash execution
may record the behavior of the exact command and assertions that ran. Structured file
operations and resource mutations establish workspace changes.

Drop repeated searches, repeated reads, superseded planning, conversational filler,
and long logs that add no new information. Preserve unresolved contradictions and
failed approaches briefly when forgetting them would cause repetition.

You must finish by calling submit_chunk_summary exactly once.
"""


class HarborCompactionConfig(BaseModel):
    """Input bounds and concurrency for Harbor compaction."""

    model_config = ConfigDict(extra="forbid")

    chunk_max_chars: int = Field(default=60_000, gt=0)
    block_max_chars: int = Field(default=6_000, gt=0)
    max_parallel_chunks: int = Field(default=4, gt=0)


MANIFEST = ExtensionManifest(
    name="harbor_compaction",
    description=("Harbor-only chronological chunk session compactor."),
    registers=(f"service:{SESSION_COMPACTOR_SERVICE}",),
    config_schema=HarborCompactionConfig,
    requires=("service:trajectory_store",),
    priority=AtomInstallPriority.SERVICE,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ChunkDigest(_StrictModel):
    summary: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _SelectedHistory:
    all_turns: Sequence[Turn]
    selected_turns: Sequence[Turn]
    target: Turn
    source_head: TrajectoryHead
    nodes: Sequence[TrajectoryNode]


@dataclass(frozen=True, slots=True)
class _HistoryChunk:
    start_turn: int
    end_turn: int
    text: str


@dataclass(frozen=True, slots=True)
class _PriorCheckpoint:
    text: str
    reliability: Literal["none", "harbor_checkpoint", "untrusted"]


@dataclass(frozen=True, slots=True)
class _StageResult:
    value: BaseModel
    session_id: str


class _StageCapture:
    """Capture one schema-validated terminal tool submission."""

    __slots__ = ("_model", "_tool_name", "value")

    def __init__(self, tool_name: str, model: type[BaseModel]) -> None:
        self._tool_name = tool_name
        self._model = model
        self.value: BaseModel | None = None

    def tool(self) -> FunctionTool:
        return FunctionTool(
            name=self._tool_name,
            description="Submit the complete schema-valid result and terminate this stage.",
            parameters=self._model,
            fn=self.submit,
        )

    async def submit(
        self,
        args: dict[str, object],
    ) -> ToolTerminate | ToolResult:
        try:
            value = self._model.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Schema validation failed; correct and retry: {exc}",
                    )
                ],
                is_error=True,
            )
        self.value = value
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text=value.model_dump_json())]),
            reason=f"harbor_compaction:{self._tool_name}",
        )


class HarborSessionCompactor:
    """Compact a committed Harbor trajectory through chronological map-reduce."""

    __slots__ = ("_api", "_config")

    def __init__(self, api: AtomAPI, config: HarborCompactionConfig) -> None:
        self._api = api
        self._config = config

    async def compact(
        self,
        request: CompactionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> CompactionResult:
        _check_signal(signal)
        store = self._api.store
        if store is None:
            raise RuntimeError("harbor_compaction requires a TrajectoryStore")
        if self._api.model is None:
            raise RuntimeError("harbor_compaction requires an active provider")

        history = await _select_history(store, request)
        task = _task_view(
            history.nodes,
            history.target.index,
        )
        chunks = _history_chunks(history.selected_turns, config=self._config)
        prior_checkpoint = _prior_checkpoint(request)
        chunk_results = await self._summarize_chunks(
            chunks,
            task=task,
            signal=signal,
        )

        digests = [cast(_ChunkDigest, result.value) for result in chunk_results]
        producer_session_id = chunk_results[-1].session_id

        state_key = request.options.get("state_key", "llm_compaction")
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("Harbor compaction request has an invalid state_key")
        selected_start = history.selected_turns[0]
        return CompactionResult(
            source=CompactionSourceAnchor(
                head=history.source_head,
                last_turn_id=history.all_turns[-1].id,
                last_turn_index=history.all_turns[-1].index,
            ),
            covered=TurnRange(
                start=history.all_turns[0].index,
                end=history.target.index,
            ),
            covered_through_turn_id=history.target.id,
            summary=_render_checkpoint(
                task=task,
                chunks=chunks,
                digests=digests,
                mutation_anchors=_mutation_anchors(history.selected_turns),
                prior_checkpoint=prior_checkpoint,
            ),
            producer_ref=f"session:{producer_session_id}",
            metadata={
                "strategy": "harbor_chronological_chunks",
                "requested_strategy": request.strategy,
                "state_key": state_key,
                "compaction_session_id": producer_session_id,
                "chunk_summary_session_ids": tuple(result.session_id for result in chunk_results),
                "chunk_count": len(chunks),
                "covered_through_turn_index": history.target.index,
                "source_turn_count": len(history.all_turns),
                "source_start_turn_index": selected_start.index,
                "incremental": request.previous_summary is not None,
            },
        )

    async def _summarize_chunks(
        self,
        chunks: Sequence[_HistoryChunk],
        *,
        task: str,
        signal: CancelSignal | None,
    ) -> list[_StageResult]:
        semaphore = asyncio.Semaphore(self._config.max_parallel_chunks)

        async def summarize(chunk: _HistoryChunk) -> _StageResult:
            async with semaphore:
                return await self._run_stage(
                    purpose=(f"harbor_compaction:chunk:{chunk.start_turn}-{chunk.end_turn}"),
                    system=_CHUNK_SYSTEM,
                    prompt=_chunk_prompt(task=task, chunk=chunk),
                    tool_name="submit_chunk_summary",
                    output_model=_ChunkDigest,
                    signal=signal,
                )

        tasks: list[asyncio.Task[_StageResult]] = []
        async with asyncio.TaskGroup() as group:
            for chunk in chunks:
                tasks.append(group.create_task(summarize(chunk)))
        return [task.result() for task in tasks]

    async def _run_stage(
        self,
        *,
        purpose: str,
        system: str,
        prompt: str,
        tool_name: str,
        output_model: type[BaseModel],
        signal: CancelSignal | None,
    ) -> _StageResult:
        _check_signal(signal)
        capture = _StageCapture(tool_name, output_model)
        child = await self._api.spawn_child_session(
            AgentSessionConfig(
                cwd=self._api.ctx.cwd,
                extensions=[],
                extra_tools=[capture.tool()],
                model=self._api.model,
                system=system,
                tool_allowlist=[tool_name],
                purpose=purpose,
                loop_config=LoopConfig(max_turns=2, max_tool_calls=2),
                cancel_signal=signal,
                parent_cancellation="inherit",
            )
        )
        try:
            await child.run(prompt)
        finally:
            await child.shutdown()
        if capture.value is None:
            raise RuntimeError(
                f"Harbor compaction stage {purpose!r} did not call {tool_name!r}; "
                f"child session: {child.session_id}"
            )
        return _StageResult(value=capture.value, session_id=child.session_id)


def _check_signal(signal: CancelSignal | None) -> None:
    if signal is not None and signal.is_set():
        raise OperationCancelledBySignal


async def _select_history(
    store: TrajectoryStore,
    request: CompactionRequest,
) -> _SelectedHistory:
    for _attempt in range(2):
        _, turns = await asyncio.to_thread(store.load, request.source_session_id)
        source_head = await asyncio.to_thread(
            store.get_head,
            request.source_session_id,
        )
        if source_head is None:
            raise RuntimeError("cannot compact a session without an active head")
        leaf_node_id = source_head.node_id or source_head.logical_parent_id
        nodes = (
            await asyncio.to_thread(
                store.load_chain,
                request.source_session_id,
                leaf_node_id,
                include_logical_parent=True,
            )
            if leaf_node_id is not None
            else []
        )
        _, current_turns = await asyncio.to_thread(
            store.load,
            request.source_session_id,
        )
        current_head = await asyncio.to_thread(
            store.get_head,
            request.source_session_id,
        )
        if current_head == source_head and current_turns == turns:
            break
    else:
        raise RuntimeError("source trajectory changed while taking compaction snapshot")
    if not turns:
        raise ValueError("cannot compact a session with no committed turns")
    target_position = (
        _turn_position(turns, request.through_turn_id)
        if request.through_turn_id is not None
        else len(turns) - 1
    )
    start_position = (
        _turn_position(turns, request.start_after_turn_id) + 1
        if request.start_after_turn_id is not None
        else 0
    )
    if start_position > target_position:
        raise ValueError("compaction range is empty or reversed")
    return _SelectedHistory(
        all_turns=turns,
        selected_turns=turns[start_position : target_position + 1],
        target=turns[target_position],
        source_head=source_head,
        nodes=nodes,
    )


def _turn_position(turns: Sequence[Turn], turn_id: str) -> int:
    position = next(
        (index for index, turn in enumerate(turns) if turn.id == turn_id),
        None,
    )
    if position is None:
        raise ValueError(f"unknown compaction turn id: {turn_id}")
    return position


def _task_view(
    nodes: Sequence[TrajectoryNode],
    target_turn: int,
) -> str:
    parts: list[str] = []
    for node in nodes:
        message = node.message
        if (
            node.turn_index is None
            or node.turn_index > target_turn
            or not isinstance(message, UserMessage)
            or message.meta.synthetic
        ):
            continue
        text = _user_message_text(message)
        if text:
            parts.append(f"[Turn {node.turn_index}]\n{text}")
    return "\n\n".join(parts) or "(no non-synthetic user messages found)"


def _history_chunks(
    turns: Sequence[Turn],
    *,
    config: HarborCompactionConfig,
) -> list[_HistoryChunk]:
    chunks: list[_HistoryChunk] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for turn in turns:
        rendered = _render_turn(turn, config.block_max_chars)
        if current and current_chars + len(rendered) > config.chunk_max_chars:
            chunks.append(_make_chunk(current))
            current = []
            current_chars = 0
        current.append((turn.index, rendered))
        current_chars += len(rendered)
    if current:
        chunks.append(_make_chunk(current))
    if not chunks:
        raise ValueError("compaction range contains no turns")
    return chunks


def _make_chunk(items: Sequence[tuple[int, str]]) -> _HistoryChunk:
    return _HistoryChunk(
        start_turn=items[0][0],
        end_turn=items[-1][0],
        text="\n\n".join(text for _, text in items),
    )


def _render_turn(turn: Turn, block_limit: int) -> str:
    parts = [f"[Turn {turn.index}]"]
    for mutation in turn.meta.resource_mutations:
        parts.append(
            "RESOURCE_MUTATION "
            f"{mutation.op} {mutation.ref.uri()} "
            f"before={mutation.before_version or '(none)'} "
            f"after={mutation.after_version or '(none)'}"
        )

    response = turn.response
    if response is not None:
        for block in response.content:
            if isinstance(block, TextContent):
                text = _clip_edges(block.text, block_limit)
                if text:
                    parts.append("ASSISTANT_TEXT\n" + text)
            elif isinstance(block, ThinkingBlock):
                text = _clip_edges(block.text, block_limit)
                if text:
                    parts.append("ASSISTANT_REASONING_HYPOTHESIS\n" + text)
            elif isinstance(block, ImageContent):
                parts.append(f"ASSISTANT_IMAGE {block.mime_type} {len(block.data)} bytes")

    for record in turn.tool_results:
        arguments = _json_text(record.call.arguments, block_limit)
        result = _tool_result_text(record.result.content, block_limit)
        parts.append(
            f"TOOL_CALL {record.call.name} args={arguments}\n"
            f"TOOL_RESULT error={record.result.is_error}\n{result}"
        )
    return "\n".join(parts)


def _mutation_anchors(turns: Sequence[Turn]) -> str:
    anchors: dict[str, str] = {}
    for turn in turns:
        for mutation in turn.meta.resource_mutations:
            key = mutation.ref.uri()
            anchors[key] = (
                f"[Turn {turn.index}] RESOURCE_MUTATION {mutation.op} {key} "
                f"before={mutation.before_version or '(none)'} "
                f"after={mutation.after_version or '(none)'}"
            )
        for record in turn.tool_results:
            if record.call.name not in {"write", "edit"} or record.result.is_error:
                continue
            path = record.call.arguments.get("path")
            if isinstance(path, str) and path:
                anchors[path] = (
                    f"[Turn {turn.index}] successful structured {record.call.name} path={path}"
                )
    return "\n".join(anchors.values()) or "(no structured file mutations in this range)"


def _chunk_prompt(*, task: str, chunk: _HistoryChunk) -> str:
    return f"""\
<authoritative-user-messages>
{task}
</authoritative-user-messages>

<chronological-trajectory-slice turns="{chunk.start_turn}-{chunk.end_turn}">
{chunk.text}
</chronological-trajectory-slice>
"""


def _render_checkpoint(
    *,
    task: str,
    chunks: Sequence[_HistoryChunk],
    digests: Sequence[_ChunkDigest],
    mutation_anchors: str,
    prior_checkpoint: _PriorCheckpoint,
) -> str:
    summaries = "\n\n".join(
        f"### Turns {chunk.start_turn}-{chunk.end_turn}\n\n{digest.summary}"
        for chunk, digest in zip(chunks, digests, strict=True)
    )
    blocks = [
        _CHECKPOINT_PREFIX.rstrip(),
        (
            "This is chronological continuation memory. Generated slice summaries are "
            "lossy; their cited source turns remain authoritative."
        ),
        "## Authoritative user messages",
        task,
    ]
    if prior_checkpoint.reliability != "none":
        blocks.extend(
            (
                f"## Prior checkpoint ({prior_checkpoint.reliability})",
                prior_checkpoint.text,
            )
        )
    blocks.extend(
        (
            "## Chronological slice summaries",
            summaries,
            "## Structured mutation anchors",
            mutation_anchors,
        )
    )
    return "\n\n".join(blocks)


def _user_message_text(message: UserMessage) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[image: {block.mime_type}, {len(block.data)} bytes]")
    return "\n".join(parts)


def _tool_result_text(content: Sequence[TextContent | ImageContent], limit: int) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(f"[image: {block.mime_type}, {len(block.data)} bytes]")
    return _clip_edges("\n".join(parts) or "(empty tool result)", limit)


def _json_text(value: Mapping[str, JsonValue], limit: int) -> str:
    return _clip_edges(
        json.dumps(thaw_json(value), ensure_ascii=False, sort_keys=True),
        limit,
    )


def _clip_edges(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    omitted = len(text) - head - tail
    return text[:head] + f"\n...[omitted {omitted} chars]...\n" + text[-tail:]


def _prior_checkpoint(request: CompactionRequest) -> _PriorCheckpoint:
    if request.previous_summary is None:
        return _PriorCheckpoint(
            text="(none; this is the first compaction)",
            reliability="none",
        )
    reliability: Literal["harbor_checkpoint", "untrusted"] = (
        "harbor_checkpoint"
        if request.previous_summary.startswith(_CHECKPOINT_PREFIX)
        else "untrusted"
    )
    return _PriorCheckpoint(
        text=request.previous_summary,
        reliability=reliability,
    )


def install(api: AtomAPI, config: HarborCompactionConfig) -> None:
    api.services.register(
        SESSION_COMPACTOR_SERVICE,
        HarborSessionCompactor(api, config),
        SessionCompactor,
        scope="session",
    )


__all__ = ("HarborCompactionConfig", "HarborSessionCompactor")
