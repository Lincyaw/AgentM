"""Spawn foreground or detached child sessions through the SDK child port.

``dispatch_agent`` has two explicit cancellation domains:

* foreground children inherit parent cancellation and return their result from
  the tool call;
* background children are independent of parent interruption, return a task
  ticket immediately, and later publish one ``SubagentResult`` trigger.

Task stop and session shutdown always remain enforceable. The atom depends
only on the typed ``SpawnedSession`` surface and does not inspect child
internals, gateway state, or physical artifact layouts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Literal
import uuid

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    AtomAPI,
    AtomInstallPriority,
    EventCancelSource,
    FunctionTool,
    JsonValue,
    LOOP_BUDGET_SERVICE,
    LoopConfig,
    SessionShutdownEvent,
    SpawnedSession,
    SubagentResult,
    TextContent,
    ToolCallBlock,
    ToolResult,
    UserInput,
)
from agentm.core.lib import (
    BackgroundTask,
    BackgroundTaskRegistry,
    SlotLimitReached,
)
from agentm.core.lib.async_cancel import await_known_outcome
from agentm.extensions import ExtensionManifest

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ABORTED: Literal["aborted"] = "aborted"
_ERROR: Literal["error"] = "error"
_Status = Literal["running", "completed", "aborted", "error"]
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0


class SubAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_workers: int = Field(default=4, ge=1)
    shutdown_grace_seconds: float = Field(
        default=_DEFAULT_SHUTDOWN_GRACE_SECONDS,
        gt=0,
        allow_inf_nan=False,
    )


MANIFEST = ExtensionManifest(
    name="sub_agent",
    description=(
        "Dispatch foreground or background child sessions with explicit "
        "cancellation ownership."
    ),
    registers=(
        "tool:dispatch_agent",
        "tool:inject_instruction",
        "tool:abort_task",
        "tool:check_agent",
        "event:session_shutdown",
    ),
    config_schema=SubAgentConfig,
    requires=("atom:system_prompt",),
    priority=AtomInstallPriority.SERVICE,
)


@dataclass(slots=True, kw_only=True)
class _ChildTask(BackgroundTask):
    purpose: str
    session: SpawnedSession
    detached: bool
    status: _Status = _RUNNING
    final_messages: list[AgentMessage] = field(default_factory=list)
    summary: str | None = None
    error: str | None = None
    superseded_by: str | None = None


def _tool_result(
    payload: Mapping[str, JsonValue],
    *,
    is_error: bool = False,
) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=True, sort_keys=True),
            )
        ],
        is_error=is_error,
        extras=payload,
    )


def _final_assistant_text(messages: Sequence[AgentMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for block in reversed(message.content):
            if not isinstance(block, ToolCallBlock):
                continue
            if block.name != "return_response":
                continue
            text = block.arguments.get("text")
            if isinstance(text, str) and text.strip():
                return text
        chunks = [
            block.text for block in message.content if isinstance(block, TextContent)
        ]
        if chunks:
            return "\n".join(chunks)
    return None


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _subagent_result_text(state: _ChildTask) -> str:
    attrs = [
        f'task_id="{_xml_attr(state.task_id)}"',
        f'child_session_id="{_xml_attr(state.session.session_id)}"',
        f'purpose="{_xml_attr(state.purpose)}"',
        f'status="{state.status}"',
    ]
    if state.superseded_by is not None:
        attrs.extend(
            (
                'stale="true"',
                f'superseded_by="{_xml_attr(state.superseded_by)}"',
            )
        )
    lines = [f"<subagent_result {' '.join(attrs)}>"]
    if state.superseded_by is not None:
        lines.append(
            "  <note>This result is stale because a newer dispatch "
            f"({_xml_attr(state.superseded_by)}) superseded it.</note>"
        )
    if state.summary:
        lines.append(f"  <summary>{_xml_attr(state.summary)}</summary>")
    if state.error:
        lines.append(f"  <error>{_xml_attr(state.error)}</error>")
    lines.append("</subagent_result>")
    return "\n".join(lines)


def _task_payload(state: _ChildTask) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "task_id": state.task_id,
        "child_session_id": state.session.session_id,
        "status": state.status,
        "purpose": state.purpose,
        "background": state.detached,
    }
    if state.summary:
        payload["summary"] = state.summary
    if state.error:
        payload["error"] = state.error
    if state.superseded_by is not None:
        payload["superseded_by"] = state.superseded_by
        payload["stale"] = True
    return payload


class _DispatchAgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(
        default="subagent",
        min_length=1,
        description="Short purpose recorded in child trajectory lineage.",
    )
    prompt: str = Field(
        min_length=1,
        description="Initial instruction for the child.",
    )
    background: bool = Field(
        default=False,
        description=(
            "Run independently and return a task ticket immediately. Parent "
            "interruption does not stop a background child."
        ),
    )
    scenario: str | None = Field(
        default=None,
        description=(
            "Optional host-resolvable scenario. Omit to inherit the current "
            "session composition."
        ),
    )
    supersedes: str | None = Field(
        default=None,
        description=(
            "Earlier task id this dispatch replaces. A late earlier result is "
            "delivered as stale."
        ),
    )


class _InjectInstructionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class _AbortTaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)


class _CheckAgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)


class _ChildTaskManager:
    def __init__(
        self,
        *,
        api: AtomAPI,
        max_workers: int,
        shutdown_grace_seconds: float,
    ) -> None:
        self._api = api
        self._max_workers = max_workers
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._registry: BackgroundTaskRegistry[_ChildTask] = BackgroundTaskRegistry(
            max_workers=max_workers
        )
        self._shutting_down = False

    def _parent_loop_config(self) -> LoopConfig:
        configured = self._api.services.get(
            LOOP_BUDGET_SERVICE,
            LoopConfig,
        )
        return configured if configured is not None else LoopConfig()

    async def dispatch(self, args: dict[str, object]) -> ToolResult:
        try:
            request = _DispatchAgentParams.model_validate(args)
        except ValidationError as exc:
            return _tool_result({"error": str(exc)}, is_error=True)
        if self._shutting_down:
            return _tool_result(
                {"error": "session is shutting down; dispatch refused"},
                is_error=True,
            )

        superseded: _ChildTask | None = None
        if request.supersedes is not None:
            async with self._registry.lock:
                superseded = self._registry.get(request.supersedes)
            if superseded is None:
                return _tool_result(
                    {"error": (f"unknown supersedes task_id: {request.supersedes}")},
                    is_error=True,
                )

        try:
            await self._registry.reserve_slot()
        except SlotLimitReached:
            return _tool_result(
                {
                    "error": (
                        f"max_workers limit reached ({self._max_workers}); "
                        "dispatch refused"
                    )
                },
                is_error=True,
            )

        cancel_source = EventCancelSource()
        child_config = AgentSessionConfig(
            cwd=self._api.ctx.cwd,
            scenario=request.scenario,
            loop_config=self._parent_loop_config(),
            purpose=request.purpose,
            cancel_signal=cancel_source,
            parent_cancellation=("independent" if request.background else "inherit"),
        )
        try:
            child = await self._api.spawn_child_session(child_config)
        except BaseException as spawn_error:
            await await_known_outcome(self._registry.release_slot())
            if not isinstance(spawn_error, Exception):
                raise
            logger.warning("sub_agent child creation failed: {}", spawn_error)
            return _tool_result(
                {"error": f"child creation failed: {spawn_error}"},
                is_error=True,
            )

        task_id = uuid.uuid4().hex
        state = _ChildTask(
            task_id=task_id,
            abort_signal=cancel_source,
            purpose=request.purpose,
            session=child,
            detached=request.background,
        )
        state.task = asyncio.create_task(
            self._run_child(state, request.prompt),
            name=f"agentm-child-{task_id}",
        )
        try:
            await self._registry.register(state)
        except BaseException:
            cancel_source.set("unknown")
            child.interrupt("unknown")
            state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)
            await await_known_outcome(self._registry.release_slot())
            raise

        if superseded is not None:
            superseded.superseded_by = task_id

        if request.background:
            return _tool_result(
                {
                    "task_id": task_id,
                    "child_session_id": child.session_id,
                    "status": _RUNNING,
                    "background": True,
                }
            )

        await state.task
        return _tool_result(
            _task_payload(state),
            is_error=state.status in {_ABORTED, _ERROR},
        )

    async def _run_child(
        self,
        state: _ChildTask,
        prompt: str,
    ) -> None:
        if state.detached:
            with self._api.track_background():
                await self._run_child_inner(state, prompt)
            return
        await self._run_child_inner(state, prompt)

    async def _run_child_inner(
        self,
        state: _ChildTask,
        prompt: str,
    ) -> None:
        try:
            await state.session.run(prompt)
            await state.session.idle()
            state.final_messages = state.session.get_messages()
            state.status = _ABORTED if state.abort_signal.is_set() else _COMPLETED
        except asyncio.CancelledError:
            state.abort_signal.set(
                state.abort_signal.reason or "task_stop",
            )
            state.session.interrupt(state.abort_signal.reason or "task_stop")
            state.final_messages = state.session.get_messages()
            state.status = _ABORTED
            state.error = "child task cancelled"
            await self._finish_child(state)
            raise
        except Exception as exc:  # noqa: BLE001
            state.final_messages = state.session.get_messages()
            state.status = _ERROR
            state.error = str(exc) or type(exc).__name__
            logger.warning(
                "sub_agent child {} failed: {}",
                state.session.session_id,
                state.error,
            )
        await self._finish_child(state)

    async def _finish_child(self, state: _ChildTask) -> None:
        state.summary = _final_assistant_text(state.final_messages)
        try:
            await state.session.shutdown()
        except Exception as exc:  # noqa: BLE001
            cleanup_error = str(exc) or type(exc).__name__
            state.status = _ERROR
            state.error = (
                f"{state.error}; shutdown failed: {cleanup_error}"
                if state.error
                else f"shutdown failed: {cleanup_error}"
            )
            logger.warning(
                "sub_agent child {} shutdown failed: {}",
                state.session.session_id,
                cleanup_error,
            )
        if state.detached and not self._shutting_down:
            self._api.push_trigger(
                SubagentResult(
                    child_session_id=state.session.session_id,
                    payload=_subagent_result_text(state),
                    terminal=False,
                ),
                origin="subagent",
            )

    async def inject_instruction(
        self,
        args: dict[str, object],
    ) -> ToolResult:
        try:
            request = _InjectInstructionParams.model_validate(args)
        except ValidationError as exc:
            return _tool_result({"error": str(exc)}, is_error=True)
        async with self._registry.lock:
            state = self._registry.get(request.task_id)
            if state is None:
                return _tool_result(
                    {"error": f"unknown task_id: {request.task_id}"},
                    is_error=True,
                )
            if state.status != _RUNNING:
                return _tool_result(
                    {
                        "error": (
                            f"task {request.task_id} is {state.status}; "
                            "instruction rejected"
                        )
                    },
                    is_error=True,
                )
            state.session.push_trigger(
                UserInput(content=(TextContent(type="text", text=request.message),)),
                priority="now",
                origin="subagent",
            )
        return _tool_result({"task_id": request.task_id, "status": _RUNNING})

    async def abort(self, args: dict[str, object]) -> ToolResult:
        try:
            request = _AbortTaskParams.model_validate(args)
        except ValidationError as exc:
            return _tool_result({"error": str(exc)}, is_error=True)
        async with self._registry.lock:
            state = self._registry.get(request.task_id)
            if state is None:
                return _tool_result(
                    {"error": f"unknown task_id: {request.task_id}"},
                    is_error=True,
                )
            if state.status != _RUNNING:
                return _tool_result(
                    {"error": (f"task {request.task_id} is already {state.status}")},
                    is_error=True,
                )
            state.abort_signal.set("task_stop")
            state.session.interrupt("task_stop")
            if state.task is not None:
                state.task.cancel()
        return _tool_result({"task_id": request.task_id, "status": "cancelling"})

    async def check_agent(
        self,
        args: dict[str, object],
    ) -> ToolResult:
        try:
            request = _CheckAgentParams.model_validate(args)
        except ValidationError as exc:
            return _tool_result({"error": str(exc)}, is_error=True)
        async with self._registry.lock:
            state = self._registry.get(request.task_id)
        if state is None:
            return _tool_result(
                {"error": f"unknown task_id: {request.task_id}"},
                is_error=True,
            )
        return _tool_result(_task_payload(state))

    async def on_session_shutdown(
        self,
        _event: SessionShutdownEvent,
    ) -> None:
        self._shutting_down = True
        async with self._registry.lock:
            states = self._registry.values()
            running = [state for state in states if state.status == _RUNNING]
            for state in running:
                state.abort_signal.set("shutdown")
                state.session.interrupt("shutdown")
        tasks = [state.task for state in running if state.task is not None]
        if not tasks:
            return
        _, pending = await asyncio.wait(
            tasks,
            timeout=self._shutdown_grace_seconds,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _SubAgentRuntime:
    def __init__(self, api: AtomAPI, config: SubAgentConfig) -> None:
        self._api = api
        self._manager = _ChildTaskManager(
            api=api,
            max_workers=config.max_workers,
            shutdown_grace_seconds=config.shutdown_grace_seconds,
        )

    def install(self) -> None:
        self._api.on(
            SessionShutdownEvent.CHANNEL,
            self._manager.on_session_shutdown,
        )
        if self._api.ctx.parent_session_id is not None:
            return
        self._api.register_tool(
            FunctionTool(
                name="dispatch_agent",
                description=(
                    "Run a child agent. Foreground mode waits and inherits "
                    "parent cancellation. Background mode returns a task "
                    "ticket immediately, survives parent interruption, and "
                    "delivers its result automatically."
                ),
                parameters=_DispatchAgentParams,
                fn=self._manager.dispatch,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="inject_instruction",
                description=(
                    "Interrupt a running background child with an additional "
                    "user instruction."
                ),
                parameters=_InjectInstructionParams,
                fn=self._manager.inject_instruction,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="abort_task",
                description=(
                    "Stop a running child task. Cancellation is scoped to that "
                    "child and does not interrupt the parent."
                ),
                parameters=_AbortTaskParams,
                fn=self._manager.abort,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="check_agent",
                description=(
                    "Return the current status and retained summary for a "
                    "dispatched child task."
                ),
                parameters=_CheckAgentParams,
                fn=self._manager.check_agent,
            )
        )


def install(api: AtomAPI, config: SubAgentConfig) -> None:
    _SubAgentRuntime(api, config).install()


__all__ = (
    "MANIFEST",
    "SubAgentConfig",
    "install",
)
