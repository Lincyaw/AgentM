from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal, cast

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.harness.events import (
    ChildSessionEndEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.harness.extension import ExtensionAPI, ProviderConfig
from agentm.extensions import ExtensionManifest

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ABORTED: Literal["aborted"] = "aborted"
_ERROR: Literal["error"] = "error"
_Status = Literal["running", "completed", "aborted", "error"]
_DEFAULT_INHERIT_EXTENSIONS = ["permission", "dedup", "trajectory"]
_SHUTDOWN_GRACE_SECONDS = 5.0

MANIFEST = ExtensionManifest(
    name="sub_agent",
    description="Spawn nested AgentSession workers without core support.",
    registers=(
        "tool:dispatch_agent",
        "tool:check_tasks",
        "tool:inject_instruction",
        "tool:abort_task",
        "event:session_shutdown",
        "event:session_ready",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "inherit_extensions": {
                "type": "array",
                "items": {"type": "string"},
                "default": _DEFAULT_INHERIT_EXTENSIONS,
            },
            "available_inherited_extensions": {
                "type": "object",
                "additionalProperties": True,
            },
            "max_workers": {"type": "integer", "minimum": 1, "default": 4},
        },
        "additionalProperties": False,
    },
)


@dataclass(slots=True)
class _ChildTask:
    task_id: str
    purpose: str
    session: Any
    task: asyncio.Task[list[Any] | None]
    abort_signal: asyncio.Event
    status: _Status = _RUNNING
    pending_instructions: list[str] = field(default_factory=list)
    final_messages: list[Any] | None = None
    error: str | None = None


class _ChildAborted(RuntimeError):
    pass


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, default=str))],
        is_error=is_error,
        details=payload,
    )


def _normalize_extension_spec(spec: Any) -> tuple[str, dict[str, Any]]:
    if (
        isinstance(spec, Sequence)
        and not isinstance(spec, (str, bytes))
        and len(spec) == 2
        and isinstance(spec[0], str)
        and isinstance(spec[1], dict)
    ):
        return spec[0], dict(spec[1])
    if isinstance(spec, dict) and isinstance(spec.get("module"), str):
        raw_cfg = spec.get("config", {})
        if isinstance(raw_cfg, dict):
            return cast(str, spec["module"]), dict(raw_cfg)
    raise ValueError(
        "extension entries must be [module, config] pairs or "
        "{'module': str, 'config': dict} objects"
    )


def _coerce_extension_specs(raw_specs: Any) -> list[tuple[str, dict[str, Any]]]:
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise ValueError("extensions must be a list")
    return [_normalize_extension_spec(spec) for spec in raw_specs]


def _resolve_inherited_extensions(
    names: list[str],
    available: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    resolved: list[tuple[str, dict[str, Any]]] = []
    for name in names:
        raw_spec = available.get(name)
        if raw_spec is None:
            continue
        resolved.append(_normalize_extension_spec(raw_spec))
    return resolved


def _get_active_provider(api: ExtensionAPI) -> ProviderConfig:
    provider = api.provider
    if provider is None:
        raise RuntimeError("sub_agent requires an active provider")
    return provider


def _load_session_types() -> tuple[Any, Any]:
    session_mod = import_module("agentm.harness.session")
    return session_mod.AgentSession, session_mod.AgentSessionConfig


async def _shutdown_child_with_error(
    child: Any,
    *,
    parent_bus: Any,
    parent_session_id: str,
    error: str | None,
) -> None:
    await child.bus.emit("session_shutdown", SessionShutdownEvent(cwd=child.cwd))
    await parent_bus.emit(
        "child_session_end",
        ChildSessionEndEvent(
            child_session_id=child.session_id,
            parent_session_id=parent_session_id,
            final_message_count=len(child.session_manager.get_messages()),
            error=error,
        ),
    )
    child.bus.clear()


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    bridge_provider = config.get("_bridge_provider")
    if isinstance(bridge_provider, ProviderConfig):
        api.register_provider(bridge_provider.name, bridge_provider)
        return

    inherit_extensions = list(
        config.get("inherit_extensions", _DEFAULT_INHERIT_EXTENSIONS)
    )
    available_inherited = dict(config.get("available_inherited_extensions", {}))
    max_workers = int(config.get("max_workers", 4))
    registry: dict[str, _ChildTask] = {}
    registry_lock = asyncio.Lock()
    parent_session_id = "unknown"
    reserved_slots = 0

    async def _drain_instructions(state: _ChildTask) -> str | None:
        async with registry_lock:
            if not state.pending_instructions:
                return None
            batched = "\n\n".join(state.pending_instructions)
            state.pending_instructions.clear()
            return batched

    async def _finalize_state(
        state: _ChildTask,
        *,
        status: _Status,
        final_messages: list[Any] | None,
        error: str | None,
    ) -> None:
        state.status = status
        state.final_messages = final_messages
        state.error = error
        if error is None:
            await state.session.shutdown()
        else:
            await _shutdown_child_with_error(
                state.session,
                parent_bus=api.events,
                parent_session_id=parent_session_id,
                error=error,
            )

    async def _run_child(
        *,
        state: _ChildTask,
        initial_prompt: str,
    ) -> list[Any] | None:
        next_prompt: str | None = initial_prompt
        final_messages: list[Any] | None = None
        try:
            while True:
                if next_prompt is None:
                    break
                final_messages = await state.session.prompt(
                    next_prompt,
                    signal=state.abort_signal,
                )
                if state.abort_signal.is_set():
                    raise _ChildAborted()
                next_prompt = await _drain_instructions(state)
            await _finalize_state(
                state,
                status=_COMPLETED,
                final_messages=final_messages,
                error=None,
            )
            return final_messages
        except _ChildAborted:
            await _finalize_state(
                state,
                status=_ABORTED,
                final_messages=state.session.session_manager.get_messages(),
                error="aborted",
            )
            return state.final_messages
        except Exception as exc:  # noqa: BLE001
            await _finalize_state(
                state,
                status=_ERROR,
                final_messages=state.session.session_manager.get_messages(),
                error=str(exc) or exc.__class__.__name__,
            )
            return state.final_messages

    async def _dispatch_agent(args: dict[str, Any]) -> ToolResult:
        nonlocal reserved_slots
        purpose = str(args.get("purpose", "subagent"))
        prompt = str(args.get("prompt", ""))
        child_extensions = _coerce_extension_specs(args.get("extensions"))
        inherited_extensions = _resolve_inherited_extensions(
            inherit_extensions,
            available_inherited,
        )
        provider = _get_active_provider(api)
        task_id = uuid.uuid4().hex
        session_cls, session_config_cls = _load_session_types()

        async with registry_lock:
            running_children = sum(
                1 for child in registry.values() if child.status == _RUNNING
            )
            if running_children + reserved_slots >= max_workers:
                return _tool_result(
                    {
                        "error": (
                            f"max_workers limit reached ({max_workers}); "
                            "refusing to dispatch another child"
                        )
                    },
                    is_error=True,
                )
            reserved_slots += 1

        child_config = session_config_cls(
            cwd=api.cwd,
            extensions=child_extensions + inherited_extensions,
            provider=(__name__, {"_bridge_provider": provider}),
            parent_bus=api.events,
            parent_session_id=parent_session_id,
            purpose=purpose,
        )
        try:
            child = await session_cls.create(child_config)
        except Exception as exc:  # noqa: BLE001
            async with registry_lock:
                reserved_slots -= 1
            return _tool_result(
                {
                    "error": (
                        f"failed to create child session for purpose {purpose!r}: {exc}"
                    )
                },
                is_error=True,
            )

        abort_signal = asyncio.Event()
        state = _ChildTask(
            task_id=task_id,
            purpose=purpose,
            session=child,
            task=asyncio.create_task(asyncio.sleep(0)),
            abort_signal=abort_signal,
        )
        state.task = asyncio.create_task(_run_child(state=state, initial_prompt=prompt))
        async with registry_lock:
            reserved_slots -= 1
            registry[task_id] = state
        return _tool_result(
            {"task_id": task_id, "status": _RUNNING, "purpose": purpose}
        )

    async def _check_tasks(_args: dict[str, Any]) -> ToolResult:
        async with registry_lock:
            tasks = list(registry.values())
        payload = {
            "tasks": [
                {
                    "task_id": state.task_id,
                    "purpose": state.purpose,
                    "status": state.status,
                    "error": state.error,
                    "final_message_count": (
                        len(state.final_messages)
                        if state.final_messages is not None
                        else None
                    ),
                }
                for state in tasks
            ]
        }
        return _tool_result(payload)

    async def _inject_instruction(args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        message = str(args.get("message", ""))
        async with registry_lock:
            state = registry.get(task_id)
            if state is None:
                return _tool_result(
                    {"error": f"unknown task_id: {task_id}"}, is_error=True
                )
            if state.status != _RUNNING:
                return _tool_result(
                    {
                        "error": (
                            f"task {task_id} is {state.status}; "
                            "instructions can only be injected into running children"
                        )
                    },
                    is_error=True,
                )
            state.pending_instructions.append(message)
        return _tool_result({"task_id": task_id, "status": _RUNNING})

    async def _abort_task(args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        async with registry_lock:
            state = registry.get(task_id)
            if state is None:
                return _tool_result(
                    {"error": f"unknown task_id: {task_id}"}, is_error=True
                )
            if state.status != _RUNNING:
                return _tool_result(
                    {
                        "error": (
                            f"task {task_id} is already {state.status}; "
                            "cannot abort it again"
                        )
                    },
                    is_error=True,
                )
            state.abort_signal.set()
        return _tool_result({"task_id": task_id, "status": _ABORTED})

    async def _on_session_ready(event: SessionReadyEvent) -> None:
        nonlocal parent_session_id
        parent_session_id = event.session_id

    async def _on_session_shutdown(_event: SessionShutdownEvent) -> None:
        async with registry_lock:
            children = list(registry.values())
        pending = [child for child in children if child.status == _RUNNING]
        if not pending:
            return
        done, still_running = await asyncio.wait(
            [child.task for child in pending],
            timeout=_SHUTDOWN_GRACE_SECONDS,
        )
        _ = done
        if still_running:
            for child in pending:
                if child.task in still_running:
                    child.abort_signal.set()
            await asyncio.gather(*still_running, return_exceptions=True)

    api.on("session_ready", _on_session_ready)
    api.on("session_shutdown", _on_session_shutdown)
    api.register_tool(
        FunctionTool(
            name="dispatch_agent",
            description="Spawn a child AgentSession and return its task id immediately.",
            parameters={
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "prompt": {"type": "string"},
                    "extensions": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "prefixItems": [
                                {"type": "string"},
                                {"type": "object"},
                            ],
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "required": ["purpose", "prompt"],
                "additionalProperties": False,
            },
            fn=_dispatch_agent,
        )
    )
    api.register_tool(
        FunctionTool(
            name="check_tasks",
            description="List active and completed child tasks.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=_check_tasks,
        )
    )
    api.register_tool(
        FunctionTool(
            name="inject_instruction",
            description="Queue an instruction for the child's next prompt turn.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["task_id", "message"],
                "additionalProperties": False,
            },
            fn=_inject_instruction,
        )
    )
    api.register_tool(
        FunctionTool(
            name="abort_task",
            description="Abort a running child session.",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            fn=_abort_task,
        )
    )
