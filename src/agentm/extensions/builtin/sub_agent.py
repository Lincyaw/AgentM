"""Builtin ``sub_agent`` atom: spawn nested ``AgentSession`` workers.

Architecture:
- Module-level helpers handle config-shape parsing and JSON-payload building.
- :class:`_ChildTaskManager` owns the long-lived state (worker registry,
  registry lock, reserved-slot counter, parent session id, shutdown grace
  logic).
- :class:`_SubAgentRuntime` owns install-time validation, manager construction,
  event subscription, and tool registration.

Migration note: the v2-trajectory branch's spawn/child-session surface differs
substantially from main. ``AgentSessionConfig`` no longer carries
``task_id`` / ``persona`` / ``lineage``; ``SpawnedSession`` exposes
``run`` / ``prompt`` / ``get_messages`` / ``shutdown`` but no ``bus`` /
``session_manager`` / ``cwd``. Places where the mapping is not one-to-one are
flagged with ``# TODO(migration):``.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from loguru import logger

from agentm.core.abi import (
    AgentSessionConfig,
    AtomAPI,
    AtomInstallPriority,
    ChildSessionEndEvent,
    Event,
    ExtensionLoadError,
    FunctionTool,
    LOOP_BUDGET_SERVICE,
    LoopConfig,
    ProviderConfig,
    ServiceNotFound,
    SessionReadyEvent,
    SessionShutdownEvent,
    SubagentResult,
    TextContent,
    ToolResult,
)
from agentm.core.lib import (
    BackgroundTask,
    BackgroundTaskRegistry,
    SlotLimitReached,
    to_jsonable,
)
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField
from pydantic import ValidationError as PydanticValidationError

from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ABORTED: Literal["aborted"] = "aborted"
_ERROR: Literal["error"] = "error"
_Status = Literal["running", "completed", "aborted", "error"]

# TODO(migration): ``DEFAULT_SHUTDOWN_GRACE_SECONDS`` was a shared lib constant
# on main; inlined here until the constant is reintroduced to core.lib.
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0

# TODO(migration): main resolved the system_prompt atom's module path through
# ``agentm.extensions.discover.discover_builtin``. That discovery module does
# not exist on this branch; the builtin path is well-known, so it is hardcoded.
_SYSTEM_PROMPT_MODULE = "agentm.extensions.builtin.system_prompt"


@dataclass(frozen=True, slots=True)
class ResolveSubagentEvent(Event):
    """Ask peer atoms to resolve a named subagent persona.

    TODO(migration): main shipped this in ``core.abi``; inlined locally so the
    atom is self-contained until it is reintroduced to the ABI. Handlers return
    ``{"body": str, "tools": [...], "budget_defaults": {...}}`` dicts.
    """

    CHANNEL: ClassVar[str] = "resolve_subagent"
    name: str = ""


class SubAgentConfig(PydanticBaseModel):
    inherit_extensions: list[str] = []
    available_inherited_extensions: dict[str, Any] = {}
    max_workers: int = 4
    shutdown_grace_seconds: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS

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
    priority=AtomInstallPriority.SERVICE,
)

@dataclass(slots=True, kw_only=True)
class _ChildTask(BackgroundTask):
    """A dispatched child session, carried as a :class:`BackgroundTask`."""

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
    superseded_by: str | None = None

class _ChildAborted(RuntimeError):
    pass

def _final_assistant_text(messages: list[Any] | None) -> str | None:
    """Pull the worker's terminal response text out of its final messages."""
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
# Reached by name, never imported (§11).
_CHILD_SESSION_REGISTRY_SERVICE = "child_session_registry"


def _list_artifacts_for_task(
    *, cwd: str, root_session_id: str, task_id: str
) -> list[dict[str, Any]]:
    """Scan the cwd-local artifact store for a task's artifacts.

    TODO(migration): main used ``agentm.core.lib.list_artifacts_for_task`` with a
    ``ProjectLayout``. This branch has no layout; the path mirrors the
    ``artifact_store`` atom's ``<cwd>/.agentm/artifacts/<root>/`` layout.
    """
    artifacts_dir = Path(cwd or ".") / ".agentm" / "artifacts" / root_session_id
    if not artifacts_dir.exists():
        return []
    metas: list[dict[str, Any]] = []
    for meta_path in sorted(artifacts_dir.glob("*.meta.json")):
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        created_by = raw.get("created_by")
        if not isinstance(created_by, dict) or created_by.get("task_id") != task_id:
            continue
        metas.append(raw)
    metas.sort(key=lambda meta: float(meta.get("created_by", {}).get("timestamp", 0.0)))
    return metas


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
    stale_attrs = ""
    if state.superseded_by is not None:
        stale_attrs = (
            f' stale="true" superseded_by={_xml_attr(state.superseded_by)}'
        )
    lines = [
        (
            f"<subagent_result task_id={_xml_attr(state.task_id)} "
            f"purpose={_xml_attr(state.purpose)}{stale_attrs}>"
        )
    ]
    if state.superseded_by is not None:
        lines.append(
            "  <note>This task was replaced by a newer dispatch "
            f"(task {_xml_attr(state.superseded_by)}) before it finished. "
            "Its result is stale — do not let it override the replacing "
            "task's result.</note>"
        )
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
    runway."""
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
    applied_budget: dict[str, int] = {}
    if max_turns is not None:
        applied_budget["max_turns"] = max_turns
    if max_tool_calls is not None:
        applied_budget["max_tool_calls"] = max_tool_calls
    return (
        LoopConfig(max_turns=max_turns, max_tool_calls=max_tool_calls),
        applied_budget,
    )

def _get_active_provider(api: AtomAPI) -> ProviderConfig:
    provider = api.get_provider()
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
    # TODO(migration): main reached into ``child.bus`` / ``child.session_manager``.
    # The v2 SpawnedSession exposes only ``shutdown`` / ``get_messages``.
    try:
        message_count = len(child.get_messages())
    except Exception:  # noqa: BLE001
        message_count = 0
    await child.shutdown()
    await parent_bus.emit(
        ChildSessionEndEvent.CHANNEL,
        ChildSessionEndEvent(
            child_session_id=child.session_id,
            parent_session_id=parent_session_id,
            final_message_count=message_count,
            error=error,
        ),
    )

class _ChildTaskManager:
    """Per-session registry + lifecycle for dispatched child agents."""

    def __init__(
        self,
        *,
        api: AtomAPI,
        inherit_extensions: list[str],
        available_inherited: dict[str, Any],
        max_workers: int,
        system_prompt_module: str,
        child_registry: Any | None = None,
        shutdown_grace_seconds: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        self._api = api
        self._inherit_extensions = inherit_extensions
        self._available_inherited = available_inherited
        self._system_prompt_module = system_prompt_module
        self._max_workers = max_workers
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._child_registry = child_registry
        self._registry: BackgroundTaskRegistry[_ChildTask] = BackgroundTaskRegistry(
            max_workers=max_workers
        )
        self._parent_session_id = "unknown"
        self._root_session_id = "unknown"

    async def _resolve_subagent(self, name: str) -> dict[str, Any] | None:
        responses = await self._api.bus.emit(
            ResolveSubagentEvent.CHANNEL, ResolveSubagentEvent(name=name)
        )
        for response in responses:
            if isinstance(response, dict) and isinstance(response.get("body"), str):
                return response
        return None

    def _parent_loop_config(self) -> LoopConfig:
        # TODO(migration): main read ``api.session.get_loop_config()``. Here the
        # loop budget is resolved from the LOOP_BUDGET_SERVICE when present.
        try:
            service = self._api.services.get(LOOP_BUDGET_SERVICE)
        except ServiceNotFound:
            return LoopConfig()
        if isinstance(service, LoopConfig):
            return service
        max_turns = getattr(service, "max_turns", None)
        max_tool_calls = getattr(service, "max_tool_calls", None)
        return LoopConfig(max_turns=max_turns, max_tool_calls=max_tool_calls)

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
        refs = _list_artifacts_for_task(
            cwd=self._api.ctx.cwd,
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
        # Post the per-child finding through the trigger queue so the parent's
        # context-drain delivers it. main used ``api.post_inbox``; the v2 unified
        # input path is ``push_trigger`` with a ``SubagentResult`` trigger.
        try:
            self._api.push_trigger(
                SubagentResult(
                    child_session_id=getattr(state.session, "session_id", ""),
                    payload=_format_subagent_result(state),
                    terminal=True,
                ),
                origin="subagent",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "sub_agent: push_trigger failed; dropped finding for {}: {}",
                state.task_id,
                exc,
            )
        if self._child_registry is not None:
            # Interactive mode: leave the child alive and registered.
            return
        if error is None:
            await state.session.shutdown()
        else:
            await _shutdown_child_with_error(
                state.session,
                parent_bus=self._api.bus,
                parent_session_id=self._parent_session_id,
                error=error,
            )

    async def _run_child(
        self, *, state: _ChildTask, initial_prompt: str
    ) -> list[Any] | None:
        # A child session is detached background work whose finding posts to the
        # parent trigger queue AFTER the parent's turn may have ended. Bracket the
        # whole run so a one-shot host stays alive until the finding lands.
        with self._api.track_background():
            return await self._run_child_inner(
                state=state, initial_prompt=initial_prompt
            )

    async def _run_child_inner(
        self, *, state: _ChildTask, initial_prompt: str
    ) -> list[Any] | None:
        # TODO(migration): main drove the child via
        # ``state.session.prompt(next_prompt, signal=...)`` returning messages.
        # The v2 SpawnedSession has ``run(text) -> messages`` (start+prompt+wait)
        # and no per-prompt signal; abort is delivered via ``interrupt()`` from
        # ``abort``. Re-running the same session for injected instructions has
        # uncertain semantics and should be revisited.
        next_prompt: str | None = initial_prompt
        final_messages: list[Any] | None = None
        try:
            while next_prompt is not None:
                final_messages = await state.session.run(next_prompt)
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
                    else _safe_get_messages(state.session)
                ),
                error="aborted",
            )
            return state.final_messages
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub_agent child session error: {}", exc)
            await self._finalize_state(
                state,
                status=_ERROR,
                final_messages=_safe_get_messages(state.session),
                error=str(exc) or exc.__class__.__name__,
            )
            return state.final_messages

    async def dispatch(self, args: dict[str, Any]) -> ToolResult:
        purpose = str(args.get("purpose", "subagent"))
        prompt = str(args.get("prompt", ""))
        subagent_type = args.get("subagent_type")
        supersedes_raw = args.get("supersedes")
        superseded_state: _ChildTask | None = None
        if isinstance(supersedes_raw, str) and supersedes_raw.strip():
            supersedes_id = supersedes_raw.strip()
            superseded_state = self._registry.get(supersedes_id)
            if superseded_state is None:
                return _tool_result(
                    {
                        "error": (
                            f"unknown supersedes task_id: {supersedes_id}; "
                            "the dispatch was not performed. Check the "
                            "task_id of the attempt you meant to replace."
                        )
                    },
                    is_error=True,
                )
        if "budget" in args:
            logger.debug(
                "sub_agent: ignoring caller-supplied budget for purpose {!r}; "
                "worker loop limits are controlled by persona/scenario/runtime config",
                purpose,
            )
        child_extensions = _coerce_extension_specs(args.get("extensions"))
        # TODO(migration): main enumerated loaded atom configs via
        # ``api.list_atoms()`` to fill inherited-extension configs. That method is
        # absent on this branch; inherited configs fall back to declared values.
        inherited_extensions = _resolve_inherited_extensions(
            self._inherit_extensions,
            self._available_inherited,
            {},
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
        _get_active_provider(self._api)
        task_id = uuid.uuid4().hex
        parent_loop_config = self._parent_loop_config()
        child_loop_config, applied_budget = _resolve_child_loop_config(
            parent=parent_loop_config,
            persona_budget=persona_budget,
        )
        if persona is not None:
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

        child_extensions_combined = (
            persona_extensions + child_extensions + inherited_extensions
        )
        # TODO(migration): the v2 ``AgentSessionConfig`` dropped ``task_id`` /
        # ``persona`` / ``lineage``; the child no longer carries per-task
        # evolution identity. ``task_id`` is retained locally for artifact
        # attribution but is not threaded into the child config.
        child_config = AgentSessionConfig(
            cwd=self._api.ctx.cwd,
            extensions=child_extensions_combined or None,
            scenario=self._api.ctx.scenario,
            provider=None,
            loop_config=child_loop_config,
            purpose=purpose,
            tool_allowlist=persona_tool_allowlist,
            parent_session_id=self._api.ctx.session_id,
            root_session_id=self._api.ctx.root_session_id,
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

        # TODO(migration): main forwarded the child's trajectory onto the parent
        # wire via ``forward_child_to_wire`` (gateway-only). That helper is absent
        # on this branch; live child streaming is not wired up.
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
            abort_signal=cast(Any, abort_signal),
            applied_budget=applied_budget,
        )
        state.task = asyncio.create_task(
            self._run_child(state=state, initial_prompt=prompt)
        )
        await self._registry.register(state)

        if superseded_state is not None:
            superseded_state.superseded_by = task_id

        await state.task
        payload: dict[str, Any] = {
            "task_id": task_id,
            "child_session_id": child.session_id,
            "status": state.status,
            "purpose": purpose,
        }
        if state.summary:
            payload["summary"] = state.summary
        if state.error:
            payload["error"] = state.error
        if superseded_state is not None:
            payload["supersedes"] = superseded_state.task_id
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
            # v2: no per-prompt signal into ``run``; interrupt the child directly.
            try:
                state.session.interrupt()
            except Exception as exc:  # noqa: BLE001
                logger.debug("sub_agent: child interrupt failed for {}: {}", task_id, exc)
        return _tool_result({"task_id": task_id, "status": _ABORTED})

    async def check_agent(self, args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        async with self._registry.lock:
            state = self._registry.get(task_id)
        if state is None:
            return _tool_result(
                {"error": f"unknown task_id: {task_id}"}, is_error=True
            )
        child_sid = getattr(state.session, "session_id", None)
        result: dict[str, Any] = {
            "task_id": task_id,
            "status": state.status,
            "purpose": state.purpose,
        }
        if child_sid:
            result["child_session_id"] = child_sid
        if state.summary:
            result["summary"] = state.summary
        if state.error:
            result["error"] = state.error
        return _tool_result(result)

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


def _safe_get_messages(session: Any) -> list[Any]:
    try:
        return list(session.get_messages())
    except Exception:  # noqa: BLE001
        return []

# Tool schemas (Pydantic -> JSON Schema via pydantic_to_tool_schema)
# ---------------------------------------------------------------------------

class _DispatchAgentParams(PydanticBaseModel):
    purpose: str
    prompt: str
    extensions: list[list[Any]] | None = PydanticField(
        default=None,
        description="Each element is a [module_path, config] pair.",
    )
    supersedes: str | None = PydanticField(
        default=None,
        description=(
            "task_id of an earlier dispatch this one replaces (a retry of "
            "the same work). If the replaced task finishes later, its result "
            "arrives marked stale instead of looking current."
        ),
    )

class _InjectInstructionParams(PydanticBaseModel):
    task_id: str
    message: str

class _AbortTaskParams(PydanticBaseModel):
    task_id: str

class _CheckAgentParams(PydanticBaseModel):
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

    def __init__(self, api: AtomAPI, config: SubAgentConfig) -> None:
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
            system_prompt_module=_SYSTEM_PROMPT_MODULE,
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

    def _child_registry(self) -> Any | None:
        # Present only inside the interactive gateway.
        try:
            return self._api.services.get(_CHILD_SESSION_REGISTRY_SERVICE)
        except ServiceNotFound:
            return None

    def _register_events(self, manager: _ChildTaskManager) -> None:
        self._api.on(SessionReadyEvent.CHANNEL, manager.on_session_ready)
        self._api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)

    def _register_tools(self, manager: _ChildTaskManager) -> None:
        is_child = bool(self._api.ctx.parent_session_id)
        if is_child:
            return
        self._api.register_tool(
            FunctionTool(
                name="dispatch_agent",
                description=(
                    "Spawn a child AgentSession and block until it "
                    "finishes. Returns {task_id, child_session_id, "
                    "status, purpose, summary}. subagent_type is "
                    "optional: omit it to inherit the current scenario's "
                    "atoms. extensions optionally adds [module_path, "
                    "config] atom pairs. Use inject_instruction to guide "
                    "the child mid-run, abort_task to stop it, "
                    "check_agent to poll status."
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
        self._api.register_tool(
            FunctionTool(
                name="check_agent",
                description=(
                    "Check the status of a dispatched child agent by its "
                    "task_id. Returns status (running/completed/aborted/error), "
                    "purpose, child_session_id, and summary if available."
                ),
                parameters=pydantic_to_tool_schema(_CheckAgentParams),
                fn=manager.check_agent,
            )
        )


async def install(api: AtomAPI, config: SubAgentConfig) -> None:
    await _SubAgentRuntime(api, config).install()


__all__ = (
    "MANIFEST",
    "SubAgentConfig",
    "install",
)
