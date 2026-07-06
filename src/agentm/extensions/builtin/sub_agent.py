"""Builtin ``sub_agent`` atom: spawn nested ``AgentSession`` workers.

Architecture:
- Module-level helpers handle config-shape parsing and JSON-payload building.
- :class:`_ChildTaskManager` owns the long-lived state (worker registry,
  registry lock, reserved-slot counter, parent session id, shutdown grace
  logic). Pulling this out of the closure-heavy ``install`` body keeps
  child lifecycle separate from install-time wiring.
- :class:`_SubAgentRuntime` owns install-time validation, manager construction,
  event subscription, and tool registration so ``install`` stays a thin
  dispatcher entry point.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from loguru import logger

from agentm.core.abi import (
    AgentSessionConfig,
    ChildSessionEndEvent,
    ExtensionAPI,
    ExtensionLoadError,
    ExtensionStaleError,
    FunctionTool,
    LoopConfig,
    ProviderConfig,
    ResolveSubagentEvent,
    SUB_AGENT_RUNTIME,
    SessionReadyEvent,
    SessionShutdownEvent,
    TextContent,
    ToolResult,
)
from agentm.core.lib import (
    forward_child_to_wire as _forward_child_to_wire,
    BackgroundTask,
    BackgroundTaskRegistry,
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    SlotLimitReached,
    list_artifacts_for_task,
    to_jsonable,
)
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField
from pydantic import ValidationError as PydanticValidationError

from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import discover_builtin

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ABORTED: Literal["aborted"] = "aborted"
_ERROR: Literal["error"] = "error"
_Status = Literal["running", "completed", "aborted", "error"]

class SubAgentConfig(PydanticBaseModel):
    inherit_extensions: list[str] = []
    available_inherited_extensions: dict[str, Any] = {}
    max_workers: int = 4
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS

MANIFEST = ExtensionManifest(
    name="sub_agent",
    description=(
        "Spawn nested AgentSession workers: dispatch_agent / "
        "inject_instruction / abort_task over background child sessions."
    ),
    registers=(
        "tool:dispatch_agent",
        "tool:inject_instruction",
        "tool:abort_task",
        "event:session_shutdown",
        "event:session_ready",
    ),
    config_schema=SubAgentConfig,
    requires=("system_prompt",),
    api_version=1,
    tier=1,
    provides_role=(SUB_AGENT_RUNTIME,),
)

@dataclass(slots=True, kw_only=True)
class _ChildTask(BackgroundTask):
    """A dispatched child session, carried as a :class:`BackgroundTask`.

    The generic asyncio bits (``task_id`` / ``task`` / ``abort_signal`` /
    ``status``) live on the base and are managed by the registry;
    everything below is child-session-specific and managed by this atom.
    """

    purpose: str
    session: Any
    status: _Status = _RUNNING
    pending_instructions: list[str] = field(default_factory=list)
    final_messages: list[Any] | None = None
    summary: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    artifact_refs: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
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

# Service the single-process gateway seeds so a spawned child can be registered
# as interactively addressable by its session id (the human can chat with it).
# Reached by name, never imported (§11) — its presence also signals interactive
# mode: when set, children are kept alive after their task instead of being
# torn down on finalize. See ``.claude/designs/interactive-subagent.md``.
_CHILD_SESSION_REGISTRY_SERVICE = "child_session_registry"


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload)))],
        is_error=is_error,
        extras=payload,
    )

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
) -> tuple[LoopConfig, dict[str, int]]:
    max_turns = persona_budget.get("max_turns", parent.max_turns)
    max_tool_calls = persona_budget.get("max_tool_calls", parent.max_tool_calls)
    # Only advertise caps that are actually enforced. An unbounded
    # ``max_turns`` (None — inherited from an uncapped parent) must not leak
    # into the worker's ``<budget>`` block as a fake "max_turns: None" limit.
    applied_budget: dict[str, int] = {}
    if max_turns is not None:
        applied_budget["max_turns"] = max_turns
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
        child_registry: Any | None = None,
        shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        self._api = api
        self._inherit_extensions = inherit_extensions
        self._available_inherited = available_inherited
        self._system_prompt_module = system_prompt_module
        self._max_workers = max_workers
        self._shutdown_grace_seconds = shutdown_grace_seconds
        # When set (interactive gateway), spawned children are registered here
        # for human addressing and kept alive after their task instead of being
        # torn down on finalize. None outside the gateway → legacy teardown.
        self._child_registry = child_registry
        self._registry: BackgroundTaskRegistry[_ChildTask] = BackgroundTaskRegistry(
            max_workers=max_workers
        )
        self._parent_session_id = "unknown"
        self._root_session_id = "unknown"

    async def _resolve_subagent(self, name: str) -> dict[str, Any] | None:
        responses = await self._api.events.emit(
            ResolveSubagentEvent.CHANNEL, ResolveSubagentEvent(name=name)
        )
        for response in responses:
            if isinstance(response, dict) and isinstance(response.get("body"), str):
                return response
        return None

    async def _drain_instructions(self, state: _ChildTask) -> str | None:
        async with self._registry.lock:
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
        # Post the per-child finding through the session inbox so the
        # parent's context-drain delivers it the same way background_exec
        # completions land. The parent session may park while the child runs;
        # this push is the wakeup and the delivery path.
        try:
            self._api.post_inbox(
                source="subagent",
                payload=_format_subagent_result(state),
                dedup_key=f"subagent-finding-{state.task_id}",
            )
        except ExtensionStaleError:
            # Atom reloaded between dispatch and finalize: the inbox we hold
            # is stale; nothing to deliver into. Same step-3 Major-3
            # discipline background_exec / monitor use.
            logger.debug(
                "sub_agent: inbox stale after reload; dropped finding for {}",
                state.task_id,
            )
        if self._child_registry is not None:
            # Interactive mode: leave the child's session alive and registered
            # so the human can keep chatting with it after its dispatched task
            # finishes (interactive-subagent §"continue after death"). The
            # parent already perceives completion via the inbox finding posted
            # above; the child's child_end marker fires only at real teardown
            # (on_session_shutdown), so the UI shows one lifecycle per child.
            return
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
        # #179: a child session is detached background work whose finding posts
        # to the parent inbox AFTER the parent's turn may have ended. Bracket
        # the whole run so a one-shot host (``agentm -p``) stays alive until the
        # finding has landed, rather than exiting on the parent's last turn and
        # dropping the child's result. The bracket ALWAYS exits (every branch
        # below finalizes), so the count cannot leak.
        try:
            bracket = self._api.track_background()
        except ExtensionStaleError:
            bracket = contextlib.nullcontext()
        with bracket:
            return await self._run_child_inner(
                state=state, initial_prompt=initial_prompt
            )

    async def _run_child_inner(
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
            logger.warning("sub_agent child session error: {}", exc)
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
        subagent_type = args.get("subagent_type")
        if "budget" in args:
            logger.debug(
                "sub_agent: ignoring caller-supplied budget for purpose {!r}; "
                "worker loop limits are controlled by persona/scenario/runtime config",
                purpose,
            )
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

        try:
            await self._registry.reserve_slot()
        except SlotLimitReached:
            return _tool_result(
                {
                    "error": (
                        f"max_workers limit reached ({self._max_workers}); "
                        "refusing to dispatch another child"
                    )
                },
                is_error=True,
            )

        # provider=None → spawn_child_session auto-wires the
        # inherit_provider builtin so the child re-uses the parent's
        # active ProviderConfig without re-authenticating.
        #
        # scenario = the parent's scenario so that a dispatch with neither a
        # persona nor an explicit ``extensions`` list (combined list empty)
        # inherits the parent's curated atom set instead of degrading. An
        # empty ``extensions`` is falsy, so ``_resolve_extensions`` skips the
        # extensions branch; without a scenario it then falls through to the
        # "auto-discover EVERY builtin with empty config" fallback, which both
        # spams install errors (cost_budget/inherit_provider/memory/wire_driver
        # can't load bare) and hands the worker a god-mode all-builtins set.
        # A non-empty combined list still wins (extensions branch precedes the
        # scenario branch), so explicit/persona dispatches are unaffected.
        child_extensions_combined = (
            persona_extensions + child_extensions + inherited_extensions
        )
        child_config = AgentSessionConfig(
            cwd=self._api.cwd,
            extensions=child_extensions_combined,
            scenario=self._api.scenario,
            provider=None,
            loop_config=child_loop_config,
            task_id=task_id,
            persona=persona_name,
            purpose=purpose,
            tool_allowlist=persona_tool_allowlist,
            lineage={
                "kind": "subagent",
                "parent_session_id": self._api.session_id,
                "root_session_id": self._api.root_session_id,
                "task_id": task_id,
                "persona": persona_name,
                "purpose": purpose,
            },
            experiment=self._api.experiment,
        )
        try:
            child = await self._api.spawn_child_session(child_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub_agent spawn_child_session failed: {}", exc)
            await self._registry.release_slot()
            return _tool_result(
                {
                    "error": (
                        f"failed to create child session for purpose {purpose!r}: {exc}"
                    )
                },
                is_error=True,
            )

        # Fan the child's own trajectory (stream/tool/turn/usage) onto the
        # parent's wire, stamped with the child's session id, so a chat peer
        # sees the sub-agent working live. No-op outside the gateway (the
        # wire_driver atom is the only thing that registers this service).
        _forward_child_to_wire(self._api, child)
        # Make the child interactively addressable by id (the human can chat
        # with it). No-op outside the interactive gateway. When present, this
        # also keeps the child alive after its task (see _finalize_state).
        if self._child_registry is not None:
            try:
                self._child_registry.register(child)
            except Exception as exc:  # noqa: BLE001
                logger.debug("sub_agent: child registry registration failed: {}", exc)

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
        await self._registry.register(state)
        return _tool_result(
            {
                "task_id": task_id,
                "child_session_id": child.session_id,
                "status": _RUNNING,
                "purpose": purpose,
            }
        )

    async def inject_instruction(self, args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        message = str(args.get("message", ""))
        async with self._registry.lock:
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
        # The two distinct error messages (unknown vs already-terminal) are
        # part of this tool's observable contract, so resolve the handle here
        # rather than via the registry's status-collapsing ``cancel``; the
        # abort itself is still ``abort_signal.set()`` under the registry lock.
        async with self._registry.lock:
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
        self._root_session_id = event.root_session_id

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        async with self._registry.lock:
            children = self._registry.values()
        pending = [child for child in children if child.status == _RUNNING]
        if pending:
            done, still_running = await asyncio.wait(
                [child.task for child in pending],
                timeout=self._shutdown_grace_seconds,
            )
            _ = done
            if still_running:
                for child in pending:
                    if child.task in still_running:
                        child.abort_signal.set()
                await asyncio.gather(*still_running, return_exceptions=True)
        # Interactive mode kept completed children alive for post-task chat;
        # the parent is now going away, so tear them all down (one child_end
        # each, emitted by AgentSession.shutdown) and drop their registry
        # entries so a stale id can never route an inbound to a dead session.
        if self._child_registry is None:
            return
        async with self._registry.lock:
            children = self._registry.values()
        for child in children:
            try:
                await child.session.shutdown()
            except Exception:
                logger.exception(
                    "sub_agent: interactive child shutdown failed for {}",
                    child.task_id,
                )
            child_sid = getattr(child.session, "session_id", None)
            if child_sid:
                self._child_registry.deregister(str(child_sid))

# Tool schemas (Pydantic -> JSON Schema via pydantic_to_tool_schema)
# ---------------------------------------------------------------------------

class _DispatchAgentParams(PydanticBaseModel):
    purpose: str
    prompt: str
    subagent_type: str | None = None
    extensions: list[list[Any]] | None = PydanticField(
        default=None,
        description="Each element is a [module_path, config] pair.",
    )

class _InjectInstructionParams(PydanticBaseModel):
    task_id: str
    message: str

class _AbortTaskParams(PydanticBaseModel):
    task_id: str

def _validate_available_inherited_configs(
    available_inherited: dict[str, Any],
) -> None:
    for name, entry in available_inherited.items():
        if not isinstance(entry, dict):
            raise ExtensionLoadError(
                __name__,
                ValueError(
                    "sub_agent.available_inherited_extensions"
                    f".{name} must be a mapping"
                ),
            )
        module_path = entry.get("module")
        if not isinstance(module_path, str) or not module_path:
            continue
        config = entry.get("config", {})
        if not isinstance(config, dict):
            raise ExtensionLoadError(
                __name__,
                ValueError(
                    "sub_agent.available_inherited_extensions"
                    f".{name}.config must be a mapping"
                ),
            )
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            raise ExtensionLoadError(module_path, exc) from exc
        manifest = getattr(module, "MANIFEST", None)
        schema_cls = getattr(manifest, "config_schema", None)
        if not (
            isinstance(schema_cls, type)
            and issubclass(schema_cls, PydanticBaseModel)
        ):
            continue
        try:
            schema_cls.model_validate(config)
        except PydanticValidationError as exc:
            raise ExtensionLoadError(
                __name__,
                ValueError(
                    "sub_agent.available_inherited_extensions"
                    f".{name}: {_format_config_validation_error(schema_cls, exc)}"
                ),
            ) from exc


def _format_config_validation_error(
    schema_cls: type[PydanticBaseModel],
    exc: PydanticValidationError,
) -> str:
    missing: list[str] = []
    for error in exc.errors():
        if error.get("type") != "missing":
            continue
        loc = error.get("loc")
        if isinstance(loc, (tuple, list)):
            missing.append(".".join(str(part) for part in loc))
        elif loc:
            missing.append(str(loc))
    if missing:
        return (
            f"config for {schema_cls.__name__} is missing required field(s): "
            + ", ".join(missing)
        )
    return f"config for {schema_cls.__name__} is invalid: {exc}"


class _SubAgentRuntime:
    """Install-time wiring for the sub-agent atom."""

    def __init__(self, api: ExtensionAPI, config: SubAgentConfig) -> None:
        self._api = api
        self._config = config

    async def install(self) -> None:
        inherit_extensions = list(self._config.inherit_extensions)
        available_inherited = dict(self._config.available_inherited_extensions)
        self._validate_inheritance_config(inherit_extensions, available_inherited)
        manager = _ChildTaskManager(
            api=self._api,
            inherit_extensions=inherit_extensions,
            available_inherited=available_inherited,
            max_workers=self._config.max_workers,
            system_prompt_module=self._system_prompt_module(),
            child_registry=self._child_registry(),
            shutdown_grace_seconds=self._config.shutdown_grace_seconds,
        )
        self._register_events(manager)
        self._register_tools(manager)

    def _validate_inheritance_config(
        self,
        inherit_extensions: list[str],
        available_inherited: dict[str, Any],
    ) -> None:
        _validate_available_inherited_configs(available_inherited)
        missing = [
            name for name in inherit_extensions if name not in available_inherited
        ]
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

    def _system_prompt_module(self) -> str:
        builtins = discover_builtin()
        system_prompt = builtins.get("system_prompt")
        if system_prompt is None:
            raise ExtensionLoadError(
                __name__,
                ValueError("sub_agent requires the builtin system_prompt atom"),
            )
        return system_prompt.module_path

    def _child_registry(self) -> Any | None:
        # Present only inside the interactive gateway: makes spawned children
        # human-addressable and keeps them alive after their task. ``getattr``
        # guards minimal stub APIs that omit the service registry.
        return getattr(self._api, "get_service", lambda _name: None)(
            _CHILD_SESSION_REGISTRY_SERVICE
        )

    def _register_events(self, manager: _ChildTaskManager) -> None:
        self._api.on(SessionReadyEvent.CHANNEL, manager.on_session_ready)
        self._api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)

    def _register_tools(self, manager: _ChildTaskManager) -> None:
        self._api.register_tool(
            FunctionTool(
                name="dispatch_agent",
                description=(
                    "Spawn a child AgentSession — returns {task_id, "
                    'child_session_id, status: "running", purpose} '
                    "immediately and the child runs in the background. Its "
                    "result arrives later in your inbox as a "
                    "<subagent_result> block containing the child's final "
                    "summary and any artifacts it produced; you are notified "
                    "automatically, so do not poll. subagent_type is "
                    "optional: omit it to spawn a child inheriting the "
                    "current scenario's atoms, or pass a known persona name "
                    "(system prompt + tool allowlist; unknown names error). "
                    "extensions optionally adds [module_path, config] atom "
                    "pairs on top of the inherited set. Use "
                    "inject_instruction to guide the child mid-run, "
                    "abort_task to stop it. Fails if max_workers children "
                    "are already running."
                ),
                parameters=pydantic_to_tool_schema(_DispatchAgentParams),
                fn=manager.dispatch,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="inject_instruction",
                description=(
                    "Queue a follow-up instruction for a running child (by its "
                    "dispatch_agent task_id); batched into the child's next prompt "
                    "turn. Errors if the task is unknown or already terminal."
                ),
                parameters=pydantic_to_tool_schema(_InjectInstructionParams),
                fn=manager.inject_instruction,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="abort_task",
                description=(
                    "Cooperatively abort a running child by its dispatch_agent "
                    "task_id. The child stops once it observes the signal. "
                    "Errors if unknown or already terminal."
                ),
                parameters=pydantic_to_tool_schema(_AbortTaskParams),
                fn=manager.abort,
            )
        )


async def install(api: ExtensionAPI, config: SubAgentConfig) -> None:
    await _SubAgentRuntime(api, config).install()


__all__ = (
    "MANIFEST",
    "SubAgentConfig",
    "install",
)
