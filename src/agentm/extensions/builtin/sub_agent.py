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
import contextlib
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from agentm.core.abi import (
    ChildSessionEndEvent,
    DecideTurnActionEvent,
    ExtensionAPI,
    ExtensionLoadError,
    ExtensionStaleError,
    FunctionTool,
    Inject,
    LoopAction,
    LoopConfig,
    ModelEndTurn,
    ProviderConfig,
    ResolveSubagentEvent,
    SUB_AGENT_RUNTIME,
    SessionReadyEvent,
    SessionShutdownEvent,
    Stop,
    TextContent,
    ToolCallEvent,
    ToolResult,
    ToolTerminated,
    UserMessage,
)
from agentm.core.lib import (
    BackgroundTask,
    BackgroundTaskRegistry,
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    SlotLimitReached,
    list_artifacts_for_task,
    to_jsonable,
)
from pydantic import BaseModel as PydanticBaseModel

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
        "event:tool_call",
    ),
    config_schema=SubAgentConfig,
    requires=("system_prompt",),
    provides_role=(SUB_AGENT_RUNTIME,),
)

@dataclass(slots=True, kw_only=True)
class _ChildTask(BackgroundTask):
    """A dispatched child session, carried as a :class:`BackgroundTask`.

    The generic asyncio bits (``task_id`` / ``task`` / ``abort_signal`` /
    ``status`` / ``read``) live on the base and are managed by the registry;
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

# Service registered by the ``wire_driver`` atom inside the gateway. Reached by
# name (not import) so this atom keeps the §11 contract: no atom-to-atom import.
_WIRE_CHILD_FORWARDER_SERVICE = "child_wire_forwarder"


def _forward_child_to_wire(api: ExtensionAPI, child: Any) -> None:
    """Hand a freshly spawned child to the wire forwarder if one is installed.

    Returns silently when running outside the gateway (no wire_driver, so no
    ``child_wire_forwarder`` service) — the child still runs, its trajectory
    just isn't streamed to a chat peer."""
    forwarder = api.get_service(_WIRE_CHILD_FORWARDER_SERVICE)
    if forwarder is None:
        return
    try:
        forwarder(child)
    except Exception:  # noqa: BLE001
        # Forwarding is observability, never load-bearing for the child's run.
        pass


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
        shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        self._api = api
        self._inherit_extensions = inherit_extensions
        self._available_inherited = available_inherited
        self._system_prompt_module = system_prompt_module
        self._max_workers = max_workers
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._registry: BackgroundTaskRegistry[_ChildTask] = BackgroundTaskRegistry(
            max_workers=max_workers
        )
        self._parent_session_id = "unknown"
        self._root_session_id = "unknown"
        # Auto-abort counter (see ``decide_turn_action``): increments on a
        # consecutive ``ModelEndTurn`` with empty text AND running children,
        # triggering an auto-abort on the second strike. Reset on ANY tool
        # call (sync handler subscribed to ``ToolCallEvent`` in ``install``):
        # if the agent is actively invoking tools it is engaged and should
        # not be auto-aborted out of its workflow. B3 boundary-review fix:
        # this was five per-tool ``await self._reset_running_only_cancels()``
        # callsites; the single bus subscription collapses the footgun
        # surface (a forgotten reset can no longer silently change
        # auto-abort behaviour).
        self._running_only_cancels = 0

    def _on_tool_call_reset_counter(self, _event: ToolCallEvent) -> None:
        """Reset the running-only-cancels counter on any tool invocation.

        Subscribed to ``ToolCallEvent.CHANNEL`` at ``install`` time. The
        kernel fires this BEFORE the tool executes (``loop.py``); a sync
        handler is sufficient because the reset is a single attribute
        assignment with no async needs. Behaviourally equivalent to the
        pre-B3 per-tool resets — the integration suites
        (``test_sub_agent_lifecycle`` / ``test_sub_agent_budgets``) prove
        the auto-abort semantics are unchanged.
        """

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
            timeout=self._shutdown_grace_seconds,
        )
        async with self._registry.lock:
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
        # Step 5b: post the per-child finding through the session inbox so
        # the parent's context-drain injects it the same way background_exec
        # completions land. The narrowed decide_turn_action floor only
        # checks ``status == _RUNNING`` (terminal states are not its
        # concern), so we do NOT set ``state.read`` here — that flag is
        # owned by the tools that surface findings directly to the model
        # (``check_tasks``, ``wait_subagent``).
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
            pass
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
        child_config = {
            "cwd": self._api.cwd,
            "extensions": child_extensions_combined,
            "scenario": self._api.scenario,
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
                "status": _RUNNING,
                "purpose": purpose,
                "budget": dict(applied_budget),
            }
        )

    async def check_tasks(self, _args: dict[str, Any]) -> ToolResult:
        await self._registry.poll_first_completed()
        async with self._registry.lock:
            tasks = self._registry.values()
            for state in tasks:
                if _is_terminal(state.status):
                    state.read = True
        return _tool_result({"tasks": [_task_payload(state) for state in tasks]})

    async def wait_subagent(self, args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        async with self._registry.lock:
            if self._registry.get(task_id) is None:
                return _tool_result(
                    {"error": f"unknown task_id: {task_id}"}, is_error=True
                )
        await self._registry.wait_one(task_id)
        async with self._registry.lock:
            state = self._registry.get(task_id)
            assert state is not None
            if _is_terminal(state.status):
                state.read = True
            payload = _task_payload(state)
        return _tool_result(payload)

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

    async def decide_turn_action(
        self, event: DecideTurnActionEvent
    ) -> LoopAction | None:
        """Floor (narrowed in step 5b): keep parent alive while children run.

        Step 5b removed the completed-unread inject branch — completed
        findings now ride through the session inbox via
        ``_finalize_state.post_inbox(source="subagent")`` and land via the
        runtime context-drain. The still-running branch survives because
        without it a parent that voluntarily ``Stop(ModelEndTurn)``s while
        children are detached would let those workers be stranded — the
        exact failure ``sub-agent-lifecycle.md`` was written to enforce.

        Only acts when the kernel default is a *voluntary* termination
        (``ModelEndTurn`` / ``ToolTerminated``). For kernel-imposed
        terminations (``MaxTurnsExhausted`` / ``SignalAborted`` /
        ``BudgetExhausted``) ``cause.final`` is True and any override is
        ignored anyway, so we return ``None``.

        Auto-abort path: the second consecutive running-only cancel triggers
        abort signals on every running child. Each aborted child's
        ``_finalize_state`` posts its finding through the session inbox
        before ``_abort_running_states`` returns; we then return ``None``
        so the runtime keep-alive floor (which sees the now-non-empty
        inbox) turns the parent's ``Stop`` into ``Step()`` and the next
        turn's context-drain delivers each ``<subagent_result>`` exactly
        once. (Earlier code Inject-ed the aborted set AND let the inbox
        drain redeliver, double-surfacing every finding — Major-2 fix on
        the step-5 review.)
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
        async with self._registry.lock:
            states = self._registry.values()
            running = [state for state in states if state.status == _RUNNING]
            if not running:
                self._running_only_cancels = 0
            elif last_text:
                self._running_only_cancels = 0

            if isinstance(default.cause, ModelEndTurn) and running:
                if self._running_only_cancels >= 1:
                    should_auto_abort = True
                    self._running_only_cancels = 0
                else:
                    self._running_only_cancels += 1

        if not running and not should_auto_abort:
            # No still-running children to keep alive for; nothing to inject.
            # Completed findings (if any) ride the inbox path and the
            # runtime keep-alive floor turns this Stop into another Step
            # whenever inbox is non-empty.
            return None

        if should_auto_abort:
            aborted = await self._abort_running_states(running)
            if not aborted:
                return None
            # Major-2 fix (option a, decided 2026-05-28): the aborted
            # children's findings are ALREADY queued on the inbox by their
            # ``_finalize_state.post_inbox`` (which ran inside the await of
            # ``_abort_running_states``). Returning ``None`` here lets the
            # runtime keep-alive floor see the non-empty inbox and turn the
            # parent's ``Stop(ModelEndTurn)`` into ``Step()`` — so the next
            # turn's ``context`` drain delivers each ``<subagent_result>``
            # EXACTLY ONCE. The previous code Inject-ed the same findings
            # this turn AND let the inbox deliver them next turn (the
            # dedup_key only dedupes inside the inbox, not across the two
            # delivery paths), compounding under multi-child fan-outs.
            return None

        # Still-running children: keep the parent alive by injecting a
        # <subagent_pending> notice so the model can choose to wait or
        # dispatch follow-up work rather than terminate the loop.
        message = _notification_message(pending=[], running=running)
        return Inject(messages=[message])

    async def on_session_ready(self, event: SessionReadyEvent) -> None:
        self._parent_session_id = event.session_id
        self._root_session_id = event.root_session_id

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        async with self._registry.lock:
            children = self._registry.values()
        pending = [child for child in children if child.status == _RUNNING]
        if not pending:
            return
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

async def install(api: ExtensionAPI, config: SubAgentConfig) -> None:
    inherit_extensions = list(config.inherit_extensions)
    available_inherited = dict(config.available_inherited_extensions)
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
        max_workers=config.max_workers,
        system_prompt_module=system_prompt.module_path,
        shutdown_grace_seconds=config.shutdown_grace_seconds,
    )

    api.on(SessionReadyEvent.CHANNEL, manager.on_session_ready)
    api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)
    api.on(DecideTurnActionEvent.CHANNEL, manager.decide_turn_action)
    # B3 boundary-review fix: encapsulate the counter reset behind the bus
    # so any tool invocation drives it (not just sub_agent's own tools).
    # ToolCallEvent fires per tool call BEFORE execute, sync handler is
    # sufficient (single int assignment).
    api.on(ToolCallEvent.CHANNEL, manager._on_tool_call_reset_counter)
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
                            "max_tool_calls": {"type": "integer"},
                            "max_turns": {"type": "integer"},
                        },
                    },
                    "extensions": {
                        "type": "array",
                        "description": "Each element is a [module_path, config] pair.",
                        "items": {
                            "type": "array",
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
