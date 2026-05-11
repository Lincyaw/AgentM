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

from agentm.core.abi.roles import SUB_AGENT_RUNTIME
from agentm.core.abi import (
    DecideTurnActionEvent,
    FunctionTool,
    Inject,
    LoopAction,
    LoopConfig,
    ModelEndTurn,
    Stop,
    TextContent,
    ToolResult,
    ToolTerminated,
    UserMessage,
)
from agentm.core.lib import to_jsonable
from agentm.core.lib.artifact_files import list_artifacts_for_task
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import discover_builtin
from agentm.core.abi.events import (
    ChildSessionEndEvent,
    ResolveSubagentEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.extension import ExtensionAPI, ExtensionLoadError, ProviderConfig

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ABORTED: Literal["aborted"] = "aborted"
_ERROR: Literal["error"] = "error"
_Status = Literal["running", "completed", "aborted", "error"]
_SHUTDOWN_GRACE_SECONDS = 5.0

MANIFEST = ExtensionManifest(
    name="sub_agent",
    description=(
        "Spawn nested AgentSession workers without core support. C18: keep this "
        "atom as one file until it reaches 1500 LOC; no split in issue #87."
    ),
    registers=(
        "tool:dispatch_agent",
        "tool:check_tasks",
        "tool:wait_subagent",
        "tool:inject_instruction",
        "tool:abort_task",
        "event:decide_turn_action",
        "event:session_shutdown",
        "event:session_ready",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "inherit_extensions": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
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
                    "``{'module': str, 'config': dict}``. When config is omitted "
                    "or empty for an inherited parent atom, the child inherits "
                    "that parent atom's resolved config by manifest name. Parents must populate "
                    "this for every name in ``inherit_extensions``; otherwise "
                    "inheritance silently fails — which is why this atom now "
                    "fast-fails on missing keys."
                ),
            },
            "max_workers": {"type": "integer", "minimum": 1, "default": 4},
        },
        "additionalProperties": True,
    },
    requires=("system_prompt",),
    provides_role=(SUB_AGENT_RUNTIME,),
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
    summary: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    artifact_refs: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    read: bool = False
    applied_budget: dict[str, int] = field(default_factory=dict)


class _ChildAborted(RuntimeError):
    pass


def _final_assistant_text(messages: list[Any] | None) -> str | None:
    """Pull the worker's terminal response text out of its final messages.

    Resolution order:

    1. The arguments of the most recent ``return_response`` tool call —
       the sanctioned termination tool installed by scenarios that need
       guaranteed worker output. Workers that take this path end on a
       tool_use turn with no assistant text, so the text-only fallback
       below would otherwise return ``None``.
    2. The most recent assistant message that contains text blocks —
       this preserves the legacy contract for scenarios where workers
       end with prose.

    Returns ``None`` while the child is still running or produced no
    output the parent can use.
    """
    if not messages:
        return None
    response = _extract_return_response_text(messages)
    if response is not None:
        return response
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


def _extract_return_response_text(messages: list[Any]) -> str | None:
    """Walk back through messages to find the last ``return_response``
    tool call and return its ``text`` argument."""
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
        for block in reversed(content):
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_call":
                continue
            name = getattr(block, "name", None) or (
                block.get("name") if isinstance(block, dict) else None
            )
            if name != "return_response":
                continue
            args = getattr(block, "arguments", None) or (
                block.get("arguments") if isinstance(block, dict) else None
            )
            if isinstance(args, dict):
                text = args.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    return None


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload)))],
        is_error=is_error,
        extras=payload,
    )


def _is_terminal(status: _Status) -> bool:
    return status in {_COMPLETED, _ABORTED, _ERROR}


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _summary_text(state: _ChildTask) -> str | None:
    if state.summary:
        return state.summary
    if state.status == _ABORTED:
        return "Task aborted before producing final text."
    if state.status == _ERROR and state.error:
        return f"Task failed: {state.error}"
    if state.error:
        return state.error
    return None


def _format_subagent_result(state: _ChildTask) -> str:
    lines = [
        (
            f"<subagent_result task_id={_xml_attr(state.task_id)} "
            f"purpose={_xml_attr(state.purpose)}>"
        )
    ]
    summary = _summary_text(state)
    if summary is not None:
        lines.append(f"  <summary>{_xml_attr(summary)}</summary>")
    if state.artifact_refs:
        lines.append("  <artifacts>")
        for ref in state.artifact_refs:
            lines.append(
                "    "
                f"<ref id={_xml_attr(ref['id'])} kind={_xml_attr(ref['kind'])} "
                f"title={_xml_attr(ref['title'])} />"
            )
        lines.append("  </artifacts>")
    lines.append("</subagent_result>")
    return "\n".join(lines)


def _notification_message(
    *,
    pending: list[_ChildTask],
    running: list[_ChildTask],
) -> UserMessage:
    parts: list[str] = []
    for state in pending:
        parts.append(_format_subagent_result(state))
    for state in running:
        parts.append(
            "<subagent_pending"
            f" task_id={_xml_attr(state.task_id)}"
            f" purpose={_xml_attr(state.purpose)} />"
        )
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text="\n\n".join(parts))],
        timestamp=time.time(),
    )


def _task_payload(state: _ChildTask) -> dict[str, Any]:
    return {
        "task_id": state.task_id,
        "purpose": state.purpose,
        "status": state.status,
        "error": state.error,
        "final_message_count": (
            len(state.final_messages) if state.final_messages is not None else None
        ),
        "final_text": _summary_text(state),
        "artifact_ids": list(state.artifact_ids),
        "budget": dict(state.applied_budget),
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
    loaded_by_name: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    resolved: list[tuple[str, dict[str, Any]]] = []
    for name in names:
        raw_spec = available.get(name)
        if raw_spec is None:
            continue
        module_path, config = _normalize_extension_spec(raw_spec)
        if config:
            resolved.append((module_path, config))
            continue
        loaded = loaded_by_name.get(name)
        if loaded is not None:
            config = dict(loaded)
        resolved.append((module_path, config))
    return resolved


def _persona_prompt_with_budget(
    *,
    body: str,
    applied_budget: dict[str, int],
) -> str:
    """Wrap the persona body with budget context so the worker sees its
    runway. The model has no other channel for this information — without
    it, the model burns through tool calls until force-stopped, never
    submitting a response."""
    if not applied_budget:
        return body
    parts = []
    if "max_turns" in applied_budget:
        parts.append(f"- max_turns: {applied_budget['max_turns']}")
    if "max_tool_calls" in applied_budget:
        parts.append(f"- max_tool_calls: {applied_budget['max_tool_calls']}")
    if not parts:
        return body
    block = (
        "<budget>\n"
        "Hard limits enforced by the harness — exceeding either ends the "
        "task with no chance to summarize:\n"
        + "\n".join(parts)
        + "\nPace yourself: leave at least one turn and one tool call to "
        "submit your response (e.g. via `return_response`).\n"
        "</budget>"
    )
    return f"{body}\n\n{block}" if body else block


def _coerce_budget(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    budget: dict[str, int] = {}
    for key in ("max_tool_calls", "max_turns"):
        value = raw.get(key)
        if isinstance(value, int) and value > 0:
            budget[key] = value
    return budget


def _resolve_child_loop_config(
    *,
    parent: LoopConfig,
    persona_budget: dict[str, int],
    dispatch_budget: dict[str, int],
) -> tuple[LoopConfig, dict[str, int]]:
    max_turns = dispatch_budget.get(
        "max_turns",
        persona_budget.get("max_turns", parent.max_turns),
    )
    max_tool_calls = dispatch_budget.get(
        "max_tool_calls",
        persona_budget.get("max_tool_calls", parent.max_tool_calls),
    )
    applied_budget = {"max_turns": max_turns}
    if max_tool_calls is not None:
        applied_budget["max_tool_calls"] = max_tool_calls
    return (
        LoopConfig(max_turns=max_turns, max_tool_calls=max_tool_calls),
        applied_budget,
    )


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
    await child.bus.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=child.cwd)
    )
    await parent_bus.emit(
        ChildSessionEndEvent.CHANNEL,
        ChildSessionEndEvent(
            child_session_id=child.session_id,
            parent_session_id=parent_session_id,
            final_message_count=len(child.session_manager.get_messages()),
            error=error,
        ),
    )
    child.bus.clear()


class _ChildTaskManager:
    """Per-session registry + lifecycle for dispatched child agents."""

    def __init__(
        self,
        *,
        api: ExtensionAPI,
        inherit_extensions: list[str],
        available_inherited: dict[str, Any],
        max_workers: int,
        system_prompt_module: str,
    ) -> None:
        self._api = api
        self._inherit_extensions = inherit_extensions
        self._available_inherited = available_inherited
        self._max_workers = max_workers
        self._system_prompt_module = system_prompt_module
        self._registry: dict[str, _ChildTask] = {}
        self._registry_lock = asyncio.Lock()
        self._reserved_slots = 0
        self._parent_session_id = "unknown"
        self._root_session_id = "unknown"
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
        responses = await self._api.events.emit(
            ResolveSubagentEvent.CHANNEL, ResolveSubagentEvent(name=name)
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
        state.summary = _final_assistant_text(final_messages)
        refs = list_artifacts_for_task(
            layout=self._api.get_project_layout(),
            root_session_id=self._root_session_id,
            task_id=state.task_id,
        )
        state.artifact_ids = [str(meta.get("artifact_id", "")) for meta in refs]
        state.artifact_refs = [
            {
                "id": str(meta.get("artifact_id", "")),
                "kind": str(meta.get("kind", "")),
                "title": str(meta.get("title", "")),
            }
            for meta in refs
            if str(meta.get("artifact_id", ""))
        ]
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
        dispatch_budget = _coerce_budget(args.get("budget"))
        child_extensions = _coerce_extension_specs(args.get("extensions"))
        inherited_extensions = _resolve_inherited_extensions(
            self._inherit_extensions,
            self._available_inherited,
            {
                atom.name: dict(getattr(atom, "config", None) or {})
                for atom in self._api.list_atoms()
            },
        )
        persona_extensions: list[tuple[str, dict[str, Any]]] = []
        persona_tool_allowlist: list[str] | None = None
        persona_budget: dict[str, int] = {}
        persona_name: str | None = None
        persona: dict[str, Any] | None = None
        if isinstance(subagent_type, str) and subagent_type.strip():
            persona_name = subagent_type.strip()
            persona = await self._resolve_subagent(persona_name)
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
            tools = persona.get("tools")
            if isinstance(tools, list) and tools:
                persona_tool_allowlist = [str(t) for t in tools]
            persona_budget = _coerce_budget(persona.get("budget_defaults"))
        # Validate parent has an active provider; the child config below
        # passes provider=None and lets spawn_child_session auto-wire the
        # inherit_provider builtin. We still pre-check here so the error
        # surfaces before the slot-reservation bookkeeping below.
        _get_active_provider(self._api)
        task_id = uuid.uuid4().hex
        parent_loop_config = self._api.session.get_loop_config()
        child_loop_config, applied_budget = _resolve_child_loop_config(
            parent=parent_loop_config,
            persona_budget=persona_budget,
            dispatch_budget=dispatch_budget,
        )
        if persona is not None:
            # Tell the worker how much runway it has so it can pace itself.
            # Without this, models tend to over-investigate and end up
            # force-stopped on budget exhaustion before submitting a
            # response.
            persona_extensions.append(
                (
                    self._system_prompt_module,
                    {
                        "prompt": _persona_prompt_with_budget(
                            body=persona["body"],
                            applied_budget=applied_budget,
                        ),
                    },
                )
            )

        async with self._registry_lock:
            running_children = sum(
                1 for child in self._registry.values() if child.status == _RUNNING
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

        # provider=None → spawn_child_session auto-wires the
        # inherit_provider builtin so the child re-uses the parent's
        # active ProviderConfig without re-authenticating.
        child_config = {
            "cwd": self._api.cwd,
            "extensions": persona_extensions + child_extensions + inherited_extensions,
            "provider": None,
            "loop_config": child_loop_config,
            "task_id": task_id,
            "persona": persona_name,
            "purpose": purpose,
            "tool_allowlist": persona_tool_allowlist,
        }
        try:
            child = await self._api.spawn_child_session(**child_config)
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
            applied_budget=applied_budget,
        )
        state.task = asyncio.create_task(
            self._run_child(state=state, initial_prompt=prompt)
        )
        async with self._registry_lock:
            self._reserved_slots -= 1
            self._registry[task_id] = state
        return _tool_result(
            {
                "task_id": task_id,
                "status": _RUNNING,
                "purpose": purpose,
                "budget": dict(applied_budget),
            }
        )

    async def check_tasks(self, _args: dict[str, Any]) -> ToolResult:
        await self._reset_running_only_cancels()
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
        return _tool_result({"tasks": [_task_payload(state) for state in tasks]})

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

    async def decide_turn_action(
        self, event: DecideTurnActionEvent
    ) -> LoopAction | None:
        """Floor: refuse to terminate while children have unread findings.

        Triggered on every turn, but only acts when the kernel default is a
        *voluntary* termination (``ModelEndTurn`` or ``ToolTerminated``). For
        kernel-imposed terminations (``MaxTurnsExhausted``, ``SignalAborted``,
        ``BudgetExhausted``) the cause is ``final`` and any override would be
        ignored anyway, so we return ``None``.

        Auto-abort path: the second consecutive running-only cancel triggers
        abort signals on every running child, then injects the resulting
        notification. The next turn's model will see the abort message and
        normally end immediately — costing one extra LLM call vs. the
        previous in-place ``event.messages.append`` mutation, but keeping the
        decision boundary clean.
        """

        default = event.observation.default_action
        # Only intercept voluntary terminations. ``Step`` (more tool calls
        # coming) and ``Inject`` (peer extension already overrode) are not
        # our concern. ``Stop`` with a non-voluntary cause is ``final`` and
        # cannot be overridden — the kernel will ignore us either way.
        if not isinstance(default, Stop):
            return None
        if not isinstance(default.cause, (ModelEndTurn, ToolTerminated)):
            return None

        last_text = (
            _last_assistant_text([event.observation.assistant_message])
            if event.observation.assistant_message is not None
            else ""
        )
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

            if isinstance(default.cause, ModelEndTurn) and not pending and running:
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
            return Inject(messages=[_notification_message(pending=aborted, running=[])])

        message = _notification_message(pending=pending, running=running)
        return Inject(messages=[message])

    async def on_session_ready(self, event: SessionReadyEvent) -> None:
        self._parent_session_id = event.session_id
        self._root_session_id = event.root_session_id

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
    inherit_extensions = list(config.get("inherit_extensions", []))
    available_inherited = dict(config.get("available_inherited_extensions", {}))
    missing = [name for name in inherit_extensions if name not in available_inherited]
    if missing:
        raise ExtensionLoadError(
            __name__,
            ValueError(
                "sub_agent.inherit_extensions references "
                f"{missing!r} but available_inherited_extensions does not "
                "supply them; parent must populate the resolution map for "
                "every inherited name."
            ),
        )

    builtins = discover_builtin()
    system_prompt = builtins.get("system_prompt")
    if system_prompt is None:
        raise ExtensionLoadError(
            __name__,
            ValueError("sub_agent requires the builtin system_prompt atom"),
        )

    manager = _ChildTaskManager(
        api=api,
        inherit_extensions=inherit_extensions,
        available_inherited=available_inherited,
        max_workers=int(config.get("max_workers", 4)),
        system_prompt_module=system_prompt.module_path,
    )

    api.on(SessionReadyEvent.CHANNEL, manager.on_session_ready)
    api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)
    api.on(DecideTurnActionEvent.CHANNEL, manager.decide_turn_action)
    api.register_tool(
        FunctionTool(
            name="dispatch_agent",
            description=(
                "Spawn a child AgentSession and return its task id immediately. "
                "Pass ``subagent_type`` to launch a named persona (resolved by "
                "peer extensions via the ``resolve_subagent`` event); the "
                "persona's system prompt, tool allowlist, and advisory budget "
                "defaults are applied to the child."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"},
                    "budget": {
                        "type": "object",
                        "properties": {
                            "max_tool_calls": {"type": "integer", "minimum": 1},
                            "max_turns": {"type": "integer", "minimum": 1},
                        },
                        "additionalProperties": False,
                    },
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
