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
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from xml.sax.saxutils import escape, quoteattr

from agentm.core.abi import (
    BeforeAgentEndEvent,
    FunctionTool,
    TextContent,
    ToolResult,
    UserMessage,
)
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
        "tool:wait_subagent",
        "tool:inject_instruction",
        "tool:abort_task",
        "event:before_agent_end",
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
    read: bool = False


class _ChildAborted(RuntimeError):
    pass


def _final_assistant_text(messages: list[Any] | None) -> str | None:
    """Pull the last assistant text block out of a child session's final
    messages. Returns ``None`` while the child is still running or produced
    no text output."""
    if not messages:
        return None
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        if role != "assistant":
            continue
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if not isinstance(content, list):
            continue
        chunks: list[str] = []
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "text":
                continue
            text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if isinstance(text, str):
                chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    return None


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, default=str))],
        is_error=is_error,
        details=payload,
    )


def _is_terminal(status: _Status) -> bool:
    return status in {_COMPLETED, _ABORTED, _ERROR}


def _task_payload(state: _ChildTask) -> dict[str, Any]:
    return {
        "task_id": state.task_id,
        "purpose": state.purpose,
        "status": state.status,
        "error": state.error,
        "final_message_count": (
            len(state.final_messages) if state.final_messages is not None else None
        ),
        "final_text": _final_assistant_text(state.final_messages),
    }


def _last_assistant_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if getattr(last, "role", None) != "assistant":
        return ""
    content = getattr(last, "content", None)
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


def _notification_message(
    *,
    pending: list[_ChildTask],
    running: list[_ChildTask],
) -> UserMessage:
    parts: list[str] = []
    for state in pending:
        final_text = escape(_notification_text(state))
        parts.append(
            "<subagent_result"
            f" task_id={quoteattr(state.task_id)}"
            f" purpose={quoteattr(state.purpose)}>"
            f"{final_text}"
            "</subagent_result>"
        )
    for state in running:
        parts.append(
            "<subagent_pending"
            f" task_id={quoteattr(state.task_id)}"
            f" purpose={quoteattr(state.purpose)} />"
        )
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text="\n\n".join(parts))],
        timestamp=time.time(),
    )


def _notification_text(state: _ChildTask) -> str:
    final_text = _final_assistant_text(state.final_messages)
    if final_text:
        return final_text
    if state.status == _ABORTED:
        return "Task aborted before producing final text."
    if state.status == _ERROR and state.error:
        return f"Task failed: {state.error}"
    if state.error:
        return state.error
    return ""


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
        self._running_only_cancels = 0

    async def _reset_running_only_cancels(self) -> None:
        async with self._registry_lock:
            self._running_only_cancels = 0

    async def _abort_running_states(
        self, running: list[_ChildTask]
    ) -> list[_ChildTask]:
        if not running:
            return []
        for state in running:
            # Reuse the public abort path so auto-aborts and explicit
            # ``abort_task`` calls share the same state transition logic.
            await self.abort({"task_id": state.task_id})
        await asyncio.wait(
            [state.task for state in running],
            timeout=_SHUTDOWN_GRACE_SECONDS,
        )
        async with self._registry_lock:
            terminal: list[_ChildTask] = []
            for state in running:
                if _is_terminal(state.status):
                    state.read = True
                    terminal.append(state)
            return terminal

    async def _resolve_subagent(self, name: str) -> dict[str, Any] | None:
        """Ask peer extensions to resolve ``name`` to a persona definition.

        A peer (e.g. a scenario-local extension) handles ``resolve_subagent``
        by returning ``{"body": str, "tools": list[str] | None}`` when the
        name is known, otherwise ``None``. The first non-None response wins.
        """
        responses = await self._api.events.emit(
            "resolve_subagent", {"name": name}
        )
        for response in responses:
            if isinstance(response, dict) and isinstance(response.get("body"), str):
                return response
        return None

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
                final_messages=(
                    final_messages
                    if final_messages is not None
                    else state.session.session_manager.get_messages()
                ),
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
        await self._reset_running_only_cancels()
        purpose = str(args.get("purpose", "subagent"))
        prompt = str(args.get("prompt", ""))
        subagent_type = args.get("subagent_type")
        child_extensions = _coerce_extension_specs(args.get("extensions"))
        inherited_extensions = _resolve_inherited_extensions(
            self._inherit_extensions,
            self._available_inherited,
        )
        persona_extensions: list[tuple[str, dict[str, Any]]] = []
        persona_tool_allowlist: list[str] | None = None
        if isinstance(subagent_type, str) and subagent_type.strip():
            persona = await self._resolve_subagent(subagent_type.strip())
            if persona is None:
                return _tool_result(
                    {
                        "error": (
                            f"unknown subagent_type {subagent_type!r}; no peer "
                            "extension resolved it via the 'resolve_subagent' "
                            "event"
                        )
                    },
                    is_error=True,
                )
            persona_extensions.append(
                (
                    "agentm.extensions.builtin.system_prompt",
                    {"prompt": persona["body"]},
                )
            )
            tools = persona.get("tools")
            if isinstance(tools, list) and tools:
                persona_tool_allowlist = [str(t) for t in tools]
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
            extensions=persona_extensions + child_extensions + inherited_extensions,
            provider=(__name__, {"_bridge_provider": provider}),
            purpose=purpose,
            tool_allowlist=persona_tool_allowlist,
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
        await self._reset_running_only_cancels()
        # Block until at least one running child changes state, so a single
        # ``check_tasks`` call always makes forward progress and the parent
        # does not burn LLM turns on no-op polls. Returns immediately when
        # there is nothing left to wait for.
        async with self._registry_lock:
            running = [
                child.task
                for child in self._registry.values()
                if child.status == _RUNNING
            ]
        if running:
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        async with self._registry_lock:
            tasks = list(self._registry.values())
            for state in tasks:
                if _is_terminal(state.status):
                    state.read = True
        payload = {
            "tasks": [_task_payload(state) for state in tasks]
        }
        return _tool_result(payload)

    async def wait_subagent(self, args: dict[str, Any]) -> ToolResult:
        await self._reset_running_only_cancels()
        task_id = str(args.get("task_id", ""))
        async with self._registry_lock:
            state = self._registry.get(task_id)
            if state is None:
                return _tool_result(
                    {"error": f"unknown task_id: {task_id}"}, is_error=True
                )
            task = state.task
        if state.status == _RUNNING:
            await task
        async with self._registry_lock:
            state = self._registry.get(task_id)
            assert state is not None
            if _is_terminal(state.status):
                state.read = True
            payload = _task_payload(state)
        return _tool_result(payload)

    async def inject_instruction(self, args: dict[str, Any]) -> ToolResult:
        await self._reset_running_only_cancels()
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
        await self._reset_running_only_cancels()
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

    async def before_agent_end(
        self, event: BeforeAgentEndEvent
    ) -> dict[str, Any] | None:
        last_text = _last_assistant_text(event.messages)
        should_auto_abort = False
        async with self._registry_lock:
            pending = [
                state
                for state in self._registry.values()
                if _is_terminal(state.status) and not state.read
            ]
            running = [
                state for state in self._registry.values() if state.status == _RUNNING
            ]
            if pending:
                for state in pending:
                    state.read = True
                self._running_only_cancels = 0
            elif not running:
                self._running_only_cancels = 0
            elif last_text:
                self._running_only_cancels = 0

            if event.stop_reason == "end_turn" and not pending and running:
                if self._running_only_cancels >= 1:
                    should_auto_abort = True
                    self._running_only_cancels = 0
                else:
                    self._running_only_cancels += 1

        if not pending and not running:
            return None

        if should_auto_abort:
            aborted = await self._abort_running_states(running)
            if not aborted:
                return None
            event.messages.append(
                _notification_message(pending=aborted, running=[])
            )
            return None

        message = _notification_message(pending=pending, running=running)
        return {"cancel": True, "append": [message]}

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
    api.on("before_agent_end", manager.before_agent_end)
    api.register_tool(
        FunctionTool(
            name="dispatch_agent",
            description=(
                "Spawn a child AgentSession and return its task id immediately. "
                "Pass ``subagent_type`` to launch a named persona (resolved by "
                "peer extensions via the ``resolve_subagent`` event); the "
                "persona's system prompt and tool allowlist are applied to the "
                "child."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"},
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
            name="wait_subagent",
            description="Wait for one child task to reach a terminal state.",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            fn=manager.wait_subagent,
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
