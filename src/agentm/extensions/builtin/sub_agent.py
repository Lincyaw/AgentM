"""Builtin ``sub_agent`` atom: spawn nested ``AgentSession`` workers.

Architecture:
- Module-level helpers handle config-shape parsing and JSON-payload building.
- :class:`_ChildTaskManager` owns the long-lived state (worker registry,
  registry lock, reserved-slot counter, parent session id, shutdown grace
  logic). Pulling this out of the closure-heavy ``install`` body keeps
  ``install`` itself a thin "wire-up the manager" entry point per
  the extension-as-scenario §4 dispatcher rule.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.harness.events import (
    ChildSessionEndEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.harness.extension import ExtensionAPI, ExtensionLoadError, ProviderConfig
from agentm.harness.session_config import AgentSessionConfig
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
                "description": (
                    "Names of parent-side extensions the child session may "
                    "inherit. Each name listed here MUST appear as a key in "
                    "``available_inherited_extensions`` or installation fails "
                    "fast with ExtensionLoadError."
                ),
            },
            "available_inherited_extensions": {
                "type": "object",
                "additionalProperties": True,
                "description": (
                    "Resolution map the parent supplies to translate inherited "
                    "names into concrete extension specs. Each value is either "
                    "a ``[module_path, config_dict]`` pair or "
                    "``{'module': str, 'config': dict}``. Parents must populate "
                    "this for every name in ``inherit_extensions``; otherwise "
                    "inheritance silently fails — which is why this atom now "
                    "fast-fails on missing keys."
                ),
            },
            "max_workers": {"type": "integer", "minimum": 1, "default": 4},
        },
        "required": ["available_inherited_extensions"],
        "additionalProperties": True,
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


class _ChildTaskManager:
    """Per-session registry + lifecycle for dispatched child agents.

    Holds:
    * the running ``task_id → _ChildTask`` registry and a single
      ``asyncio.Lock`` guarding mutations.
    * ``reserved_slots``, a counter that tracks in-flight ``dispatch_agent``
      calls so we can enforce ``max_workers`` against not-yet-registered
      tasks. Without it, two parallel dispatches could both observe
      ``running_children == 0`` and bypass the limit.
    * the parent session id, captured from ``session_ready`` so shutdown
      events on this manager's children can reference the parent.

    Public entry points (one per registered tool / lifecycle event):
    ``dispatch``, ``check_tasks``, ``inject_instruction``, ``abort``,
    ``on_session_ready``, ``on_session_shutdown``.
    """

    def __init__(
        self,
        *,
        api: ExtensionAPI,
        inherit_extensions: list[str],
        available_inherited: dict[str, Any],
        max_workers: int,
    ) -> None:
        self._api = api
        self._inherit_extensions = inherit_extensions
        self._available_inherited = available_inherited
        self._max_workers = max_workers
        self._registry: dict[str, _ChildTask] = {}
        self._registry_lock = asyncio.Lock()
        self._reserved_slots = 0
        self._parent_session_id = "unknown"

    async def _drain_instructions(self, state: _ChildTask) -> str | None:
        async with self._registry_lock:
            if not state.pending_instructions:
                return None
            batched = "\n\n".join(state.pending_instructions)
            state.pending_instructions.clear()
            return batched

    async def _finalize_state(
        self,
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
                parent_bus=self._api.events,
                parent_session_id=self._parent_session_id,
                error=error,
            )

    async def _run_child(
        self, *, state: _ChildTask, initial_prompt: str
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
                next_prompt = await self._drain_instructions(state)
            await self._finalize_state(
                state,
                status=_COMPLETED,
                final_messages=final_messages,
                error=None,
            )
            return final_messages
        except _ChildAborted:
            await self._finalize_state(
                state,
                status=_ABORTED,
                final_messages=state.session.session_manager.get_messages(),
                error="aborted",
            )
            return state.final_messages
        except Exception as exc:  # noqa: BLE001
            await self._finalize_state(
                state,
                status=_ERROR,
                final_messages=state.session.session_manager.get_messages(),
                error=str(exc) or exc.__class__.__name__,
            )
            return state.final_messages

    async def dispatch(self, args: dict[str, Any]) -> ToolResult:
        purpose = str(args.get("purpose", "subagent"))
        prompt = str(args.get("prompt", ""))
        child_extensions = _coerce_extension_specs(args.get("extensions"))
        inherited_extensions = _resolve_inherited_extensions(
            self._inherit_extensions,
            self._available_inherited,
        )
        provider = _get_active_provider(self._api)
        task_id = uuid.uuid4().hex

        async with self._registry_lock:
            running_children = sum(
                1
                for child in self._registry.values()
                if child.status == _RUNNING
            )
            if running_children + self._reserved_slots >= self._max_workers:
                return _tool_result(
                    {
                        "error": (
                            f"max_workers limit reached ({self._max_workers}); "
                            "refusing to dispatch another child"
                        )
                    },
                    is_error=True,
                )
            self._reserved_slots += 1

        # ``parent_bus`` / ``parent_session_id`` are overridden by the harness
        # inside ``api.spawn_child_session``; we leave them at the dataclass
        # default so the override is unambiguous.
        child_config = AgentSessionConfig(
            cwd=self._api.cwd,
            extensions=child_extensions + inherited_extensions,
            provider=(__name__, {"_bridge_provider": provider}),
            purpose=purpose,
        )
        try:
            child = await self._api.spawn_child_session(child_config)
        except Exception as exc:  # noqa: BLE001
            async with self._registry_lock:
                self._reserved_slots -= 1
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
        state.task = asyncio.create_task(
            self._run_child(state=state, initial_prompt=prompt)
        )
        async with self._registry_lock:
            self._reserved_slots -= 1
            self._registry[task_id] = state
        return _tool_result(
            {"task_id": task_id, "status": _RUNNING, "purpose": purpose}
        )

    async def check_tasks(self, _args: dict[str, Any]) -> ToolResult:
        async with self._registry_lock:
            tasks = list(self._registry.values())
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

    async def inject_instruction(self, args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        message = str(args.get("message", ""))
        async with self._registry_lock:
            state = self._registry.get(task_id)
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

    async def abort(self, args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        async with self._registry_lock:
            state = self._registry.get(task_id)
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

    async def on_session_ready(self, event: SessionReadyEvent) -> None:
        self._parent_session_id = event.session_id

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        async with self._registry_lock:
            children = list(self._registry.values())
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


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Bridge mode: the harness loads sub_agent as the *provider* extension
    # for a freshly-spawned child session, passing ``_bridge_provider``.
    # We just hand the parent's provider through so the child can stream.
    bridge_provider = config.get("_bridge_provider")
    if isinstance(bridge_provider, ProviderConfig):
        api.register_provider(bridge_provider.name, bridge_provider)
        return

    # Dispatch mode: parent populates ``available_inherited_extensions`` so
    # children can inherit selected parent atoms. The discovery filter
    # already skips this atom when config is ``{}`` (the key is in
    # MANIFEST.config_schema.required), so we can assume it is present.
    inherit_extensions = list(
        config.get("inherit_extensions", _DEFAULT_INHERIT_EXTENSIONS)
    )
    available_inherited = dict(config["available_inherited_extensions"])
    missing = [name for name in inherit_extensions if name not in available_inherited]
    if missing:
        # Fast-fail: silently dropping inherited extensions hides subtle child
        # misbehaviour (e.g. permission policy not applied). Match design
        # §10b.4 ordering errors and surface this at install time.
        raise ExtensionLoadError(
            __name__,
            ValueError(
                "sub_agent.inherit_extensions references "
                f"{missing!r} but available_inherited_extensions does not "
                "supply them; parent must populate the resolution map for "
                "every inherited name."
            ),
        )

    manager = _ChildTaskManager(
        api=api,
        inherit_extensions=inherit_extensions,
        available_inherited=available_inherited,
        max_workers=int(config.get("max_workers", 4)),
    )

    api.on("session_ready", manager.on_session_ready)
    api.on("session_shutdown", manager.on_session_shutdown)
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
            fn=manager.dispatch,
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
            fn=manager.check_tasks,
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
            fn=manager.inject_instruction,
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
            fn=manager.abort,
        )
    )
