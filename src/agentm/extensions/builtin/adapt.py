"""Builtin ``adapt`` atom for online task self-modification.

The atom is intentionally small: it gives a single agent enough control
to discover runtime event hook points, scaffold simple observer atoms,
install/reload its own helper atoms, see lifecycle diagnostics, and keep
scenario-local atoms persistent when explicitly requested. Workspace helpers
and runtime extension atoms are separate adaptation mechanisms.
"""

from __future__ import annotations

import json
import inspect
from collections import deque
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import agentm.core.abi as abi
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    BeforeRunEvent,
    DiagnosticEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest


class AdaptConfig(BaseModel):
    max_events: int = 50
    inject_events: int = 8


MANIFEST = ExtensionManifest(
    name="adapt",
    description=(
        "Online self-adaptation tools for task agents: discover runtime event "
        "hooks, scaffold helper atoms, install/reload/unload agent-authored "
        "atoms, and inspect recent extension diagnostics."
    ),
    registers=(
        "tool:adapt_status",
        "tool:adapt_events",
        "tool:adapt_list_events",
        "tool:adapt_get_event",
        "tool:adapt_event_scaffold",
        "tool:adapt_install",
        "tool:adapt_install_file",
        "tool:adapt_reload",
        "tool:adapt_reload_file",
        "tool:adapt_unload",
        "event:before_agent_start",
        "event:diagnostic",
        "event:extension_install",
        "event:extension_reload",
    ),
    config_schema=AdaptConfig,
    requires=(),
)


class _StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EmptyParams(_StrictParams):
    pass


class _EventsParams(_StrictParams):
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum number of recent adaptation diagnostics/lifecycle records "
            "to return."
        ),
    )


class _ListEventsParams(_StrictParams):
    visibility: Literal["recommended", "advanced", "all"] = Field(
        default="recommended",
        description=(
            "recommended: start here for common self-observation hooks such as "
            "tool calls, tool results, diagnostics, and extension lifecycle; "
            "advanced: include lower-level hooks; all: include every discovered "
            "runtime event."
        ),
    )
    include_observed: bool = Field(
        default=True,
        description="Include per-channel counts observed in the current session.",
    )


class _GetEventParams(_StrictParams):
    channel: str = Field(
        description=(
            "Event channel to inspect after adapt_list_events, e.g. tool_result "
            "for observing tool outputs and errors."
        )
    )
    include_observed: bool = Field(
        default=True,
        description="Include per-channel counts observed in the current session.",
    )


class _EventScaffoldParams(_StrictParams):
    name: str = Field(
        description=(
            "Name for the new runtime atom; must be a Python identifier and "
            "will be used as MANIFEST.name."
        )
    )
    channel: str = Field(
        description="Event channel the generated observer atom should subscribe to."
    )
    goal: str = Field(
        description=(
            "Short description of the signal the atom should capture or summarize."
        )
    )
    tool_name: str | None = Field(
        default=None,
        description=(
            "Optional summary tool name. Defaults to '<name>_summary'. "
            "Use null to keep the default."
        ),
    )


class _InstallParams(_StrictParams):
    name: str = Field(
        description=(
            "Atom name to install; must be a valid identifier and match "
            "MANIFEST.name in the source."
        )
    )
    source: str = Field(
        description=(
            "Full Python source for the atom. Must define MANIFEST and "
            "install(session, config), and pass the runtime atom contract."
        )
    )
    rationale: str = Field(
        description=(
            "Why this runtime atom is needed instead of a normal workspace helper."
        )
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional config passed to install(session, config).",
    )
    scope: Literal["user", "scenario"] = Field(
        default="user",
        description=(
            "user: persist under <cwd>/.agentm/atoms and auto-load in future "
            "sessions. scenario: write <scenario_dir>/<name>.py and optionally "
            "pin it in the current scenario manifest."
        ),
    )
    pin_manifest: bool = Field(
        default=True,
        description=(
            "Only for scope=scenario. When true, append `local: <name>` to "
            "the current manifest if it is not already present."
        ),
    )


class _InstallFileParams(_StrictParams):
    name: str = Field(
        description=(
            "Atom name to install; must be a valid identifier and match "
            "MANIFEST.name in the source file."
        )
    )
    source_path: str = Field(
        description=(
            "Path to the atom source file in the current operations backend. "
            "The file is read through the same file access layer as workspace tools."
        )
    )
    rationale: str = Field(
        description=(
            "Why this runtime atom is needed instead of a normal workspace helper."
        )
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional config passed to install(session, config).",
    )
    scope: Literal["user", "scenario"] = Field(
        default="user",
        description="Where to persist the installed runtime-side atom.",
    )
    pin_manifest: bool = Field(
        default=True,
        description="Only for scope=scenario; append `local: <name>` to the manifest.",
    )


class _ReloadParams(_StrictParams):
    name: str = Field(description="Name of the already-loaded atom to replace.")
    source: str = Field(description="Full replacement Python source for the atom.")
    rationale: str = Field(
        description="What bug, missing signal, or behavior change this reload fixes."
    )


class _ReloadFileParams(_StrictParams):
    name: str = Field(description="Name of the already-loaded atom to replace.")
    source_path: str = Field(
        description=(
            "Path to replacement source in the current operations backend. "
            "The file is read through the same file access layer as workspace tools."
        )
    )
    rationale: str = Field(
        description="What bug, missing signal, or behavior change this reload fixes."
    )


class _UnloadParams(_StrictParams):
    name: str = Field(
        description="Loaded runtime atom name to remove from the running session."
    )
    rationale: str = Field(
        description="Why this atom is no longer useful or is actively harmful."
    )
    unpin_manifest: bool = Field(
        default=False,
        description=(
            "When true, remove `local: <name>` from the current scenario "
            "manifest if present. Only affects scenario-local pins."
        ),
    )


_Params = TypeVar("_Params", bound=BaseModel)


def _json_tool(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, indent=2, sort_keys=True),
            )
        ],
        is_error=is_error,
    )


def _parse_params(
    model_cls: type[_Params], args: dict[str, Any]
) -> _Params | ToolResult:
    try:
        return model_cls.model_validate(args)
    except ValidationError as exc:
        return _json_tool(
            {"ok": False, "error": "invalid tool arguments", "details": exc.errors()},
            is_error=True,
        )


def _result_payload(result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "ok",
        "name",
        "module_path",
        "target_path",
        "old_hash",
        "new_hash",
        "file_created",
        "rolled_back",
        "error",
    ):
        if hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                out[key] = value
    return out


def _atom_payload(atom: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("name", "tier", "api_version", "source_path", "current_hash"):
        if hasattr(atom, key):
            value = getattr(atom, key)
            if value is not None:
                payload[key] = value
    return payload


def _type_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return str(value).replace("typing.", "")


def _doc_summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return ""
    return "\n".join(line.strip() for line in doc.splitlines() if line.strip())


def _hook_payload(event_cls: type[Any], event_doc: str) -> dict[str, Any]:
    event_notes = [event_doc] if event_doc else []
    hook = getattr(event_cls, "HOOK", None)
    if hook is None:
        return {
            "visibility": "advanced",
            "effects": ["observe"],
            "return_contract": None,
            "mutation_contract": None,
            "mutable_fields": [],
            "handler": "sync_or_async",
            "notes": event_notes,
            "extra_notes": [],
        }
    extra_notes = list(getattr(hook, "notes", ()))
    return {
        "visibility": getattr(hook, "visibility", "advanced"),
        "effects": list(getattr(hook, "effects", ("observe",))),
        "return_contract": getattr(hook, "return_contract", None),
        "mutation_contract": getattr(hook, "mutation_contract", None),
        "mutable_fields": list(getattr(hook, "mutable_fields", ())),
        "handler": getattr(hook, "handler", "sync_or_async"),
        "notes": [*event_notes, *extra_notes],
        "extra_notes": extra_notes,
    }


def _event_payload(event_cls: type[Any]) -> dict[str, Any] | None:
    channel = getattr(event_cls, "CHANNEL", None)
    if not isinstance(channel, str):
        return None
    event_fields: list[dict[str, str]] = []
    if is_dataclass(event_cls):
        for field in fields(event_cls):
            if field.name == "dispatch_id":
                continue
            event_fields.append({"name": field.name, "type": _type_label(field.type)})
    doc = _doc_summary(event_cls)
    return {
        "channel": channel,
        "event_type": event_cls.__name__,
        "import": f"from agentm.core.abi import {event_cls.__name__}",
        "fields": event_fields,
        "doc": doc,
        "hook": _hook_payload(event_cls, doc),
        "subscribe_example": (
            f"session.bus.on({event_cls.__name__}.CHANNEL, _on_{channel.replace('-', '_')})"
        ),
    }


def _observed_payload(
    observed_events: dict[str, dict[str, Any]],
    channel: str,
) -> dict[str, Any]:
    return observed_events.get(
        channel,
        {
            "count": 0,
            "last_event_type": None,
            "last_fields": [],
        },
    )


def _doc_teaser(doc: str) -> str:
    if not doc:
        return ""
    first_paragraph = doc.split("\n\n", 1)[0]
    return first_paragraph.strip()


def _event_summary(
    item: dict[str, Any],
    *,
    observed_events: dict[str, dict[str, Any]],
    include_observed: bool,
) -> dict[str, Any]:
    raw_hook = item.get("hook")
    hook = raw_hook if isinstance(raw_hook, dict) else {}
    channel = str(item["channel"])
    summary = {
        "channel": channel,
        "event_type": item["event_type"],
        "visibility": hook.get("visibility", "advanced"),
        "effects": list(hook.get("effects", ())),
        "summary": _doc_teaser(str(item.get("doc", ""))),
    }
    if include_observed:
        summary["observed"] = _observed_payload(observed_events, channel)
    return summary


def _discover_events() -> list[dict[str, Any]]:
    by_channel: dict[str, dict[str, Any]] = {}
    for _name, obj in inspect.getmembers(abi, inspect.isclass):
        try:
            is_event = issubclass(obj, abi.Event) and obj is not abi.Event
        except TypeError:
            is_event = False
        if not is_event:
            continue
        payload = _event_payload(obj)
        if payload is not None:
            by_channel[payload["channel"]] = payload
    return [by_channel[channel] for channel in sorted(by_channel)]


def _visibility_rank(value: str) -> int:
    if value == "recommended":
        return 0
    if value == "advanced":
        return 1
    return 2


def _filter_event_catalog(
    catalog: list[dict[str, Any]],
    *,
    channel: str | None,
    visibility: Literal["recommended", "advanced", "all"],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_rank = {"recommended": 0, "advanced": 1, "all": 2}[visibility]
    for item in catalog:
        if channel is not None and item["channel"] != channel:
            continue
        raw_hook = item.get("hook")
        hook = raw_hook if isinstance(raw_hook, dict) else {}
        item_visibility = str(hook.get("visibility", "advanced"))
        if _visibility_rank(item_visibility) <= max_rank:
            out.append(item)
    return out


def _scaffold_source(
    *,
    name: str,
    tool_name: str,
    event: dict[str, Any],
    goal: str,
) -> str:
    event_type = str(event["event_type"])
    channel = str(event["channel"])
    return f"""from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    FunctionTool,
    TextContent,
    ToolResult,
    {event_type},
)
from agentm.extensions import ExtensionManifest


class _SummaryParams(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name={name!r},
    description={goal!r},
    registers=("tool:{tool_name}", "event:{channel}"),
)


def _snapshot(event: Any) -> dict[str, Any]:
    if not is_dataclass(event):
        return {{"repr": repr(event)}}
    payload: dict[str, Any] = {{}}
    for field in fields(event):
        if field.name == "dispatch_id":
            continue
        value = getattr(event, field.name)
        try:
            json.dumps(value)
            payload[field.name] = value
        except TypeError:
            payload[field.name] = repr(value)
    return payload


def install(session: Any, config: dict[str, Any] | None = None) -> None:
    state: dict[str, Any] = {{"count": 0, "recent": []}}

    def _on_event(event: {event_type}) -> None:
        state["count"] += 1
        recent = state["recent"]
        recent.append(_snapshot(event))
        del recent[:-10]

    async def _summary(args: dict[str, Any]) -> ToolResult:
        del args
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(state, indent=2, sort_keys=True),
                )
            ]
        )

    session.bus.on({event_type}.CHANNEL, _on_event)
    session.register_tool(
        FunctionTool(
            name={tool_name!r},
            description="Summarize recent {channel} events observed by {name}.",
            parameters=_SummaryParams,
            fn=_summary,
        )
    )
"""


def _find_manifest_path(session: Any) -> Path | None:
    scenario_dir_raw = session.ctx.scenario_dir
    if not scenario_dir_raw:
        return None
    scenario_dir = Path(scenario_dir_raw)
    wanted = session.ctx.scenario
    candidates = sorted(scenario_dir.glob("manifest*.yaml"))
    for path in candidates:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.debug("adapt: failed to parse manifest {}: {}", path, exc)
            loaded = None
        if isinstance(loaded, dict) and loaded.get("name") == wanted:
            return path
    fallback = scenario_dir / "manifest.yaml"
    return fallback if fallback.exists() else None


def _local_entry_index(extensions: list[Any], name: str) -> int | None:
    for index, entry in enumerate(extensions):
        if isinstance(entry, dict) and entry.get("local") == name:
            return index
    return None


def _pin_manifest(session: Any, name: str) -> dict[str, Any]:
    manifest_path = _find_manifest_path(session)
    if manifest_path is None:
        return {"ok": False, "error": "current scenario manifest not found"}
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("adapt: pin manifest read failed: {}", exc)
        return {"ok": False, "path": str(manifest_path), "error": str(exc)}
    if not isinstance(loaded, dict):
        return {
            "ok": False,
            "path": str(manifest_path),
            "error": "manifest root is not a mapping",
        }
    extensions = loaded.get("extensions")
    if not isinstance(extensions, list):
        return {
            "ok": False,
            "path": str(manifest_path),
            "error": "manifest extensions is not a list",
        }
    if _local_entry_index(extensions, name) is not None:
        return {"ok": True, "path": str(manifest_path), "changed": False}
    extensions.append({"local": name})
    manifest_path.write_text(
        yaml.safe_dump(loaded, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return {"ok": True, "path": str(manifest_path), "changed": True}


def _unpin_manifest(session: Any, name: str) -> dict[str, Any]:
    manifest_path = _find_manifest_path(session)
    if manifest_path is None:
        return {"ok": False, "error": "current scenario manifest not found"}
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("adapt: unpin manifest read failed: {}", exc)
        return {"ok": False, "path": str(manifest_path), "error": str(exc)}
    if not isinstance(loaded, dict):
        return {
            "ok": False,
            "path": str(manifest_path),
            "error": "manifest root is not a mapping",
        }
    extensions = loaded.get("extensions")
    if not isinstance(extensions, list):
        return {
            "ok": False,
            "path": str(manifest_path),
            "error": "manifest extensions is not a list",
        }
    before = len(extensions)
    loaded["extensions"] = [
        entry
        for entry in extensions
        if not (isinstance(entry, dict) and entry.get("local") == name)
    ]
    changed = len(loaded["extensions"]) != before
    if changed:
        manifest_path.write_text(
            yaml.safe_dump(loaded, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    return {"ok": True, "path": str(manifest_path), "changed": changed}


class _AdaptRuntime:
    """Per-session state and handlers for the adapt atom."""

    def __init__(self, session: Any, config: AdaptConfig) -> None:
        self.api = session
        self.session = session
        self.max_events = max(1, config.max_events)
        self.inject_events = max(0, min(config.inject_events, self.max_events))
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self.observed_events: dict[str, dict[str, Any]] = {}

    def install(self) -> None:
        self.session.bus.on(DiagnosticEvent.CHANNEL, self._on_diagnostic)
        self.session.bus.on(ExtensionInstallEvent.CHANNEL, self._on_extension_install)
        self.session.bus.on(ExtensionReloadEvent.CHANNEL, self._on_extension_reload)
        self.session.bus.on(BeforeRunEvent.CHANNEL, self._inject)
        self.api.add_observer(self._observe_event)
        self._register_tools()

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self.recent_events.append({"kind": kind, **payload})

    def _observe_event(self, channel: str, event: Any) -> None:
        record = self.observed_events.setdefault(
            channel,
            {
                "count": 0,
                "last_event_type": None,
                "last_fields": [],
            },
        )
        record["count"] += 1
        record["last_event_type"] = type(event).__name__
        if is_dataclass(event):
            record["last_fields"] = [
                field.name for field in fields(event) if field.name != "dispatch_id"
            ]

    def _on_diagnostic(self, event: DiagnosticEvent) -> None:
        if event.level == "error":
            self._record(
                "diagnostic",
                {
                    "level": event.level,
                    "source": event.source,
                    "message": event.message,
                },
            )

    def _on_extension_install(self, event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        self._record(
            "extension_install",
            {"name": event.name, "phase": event.phase},
        )

    def _on_extension_reload(self, event: ExtensionReloadEvent) -> None:
        self._record(
            "extension_reload",
            {"name": event.name},
        )

    def _inject(self, event: BeforeRunEvent) -> dict[str, str] | None:
        lines = [
            "# Adapt",
            "",
            "Adapt tools let you observe and extend your own runtime when "
            "workspace scripts are not enough. Ordinary task scripts and "
            "project checks still belong in the task workspace.",
            "",
            "Runtime observation starts with adapt_status and adapt_events. "
            "adapt_status reports loaded atoms and current diagnostics. "
            "adapt_events reports recent adaptation failures, installs, and "
            "reloads.",
            "",
            "The event system is the way to observe yourself across turns. Use "
            "adapt_list_events to discover event channels. Common observation "
            "targets include tool calls, tool results, diagnostics, and "
            "extension install/reload events. Then call adapt_get_event for a "
            "chosen channel to see payload fields, observed counts, return "
            "contract, mutation contract, import snippet, and subscription "
            "example.",
            "",
            "To make your own runtime tool, write an atom that keeps compact "
            "state, subscribes to the event you need, and registers a tool that "
            "returns the state or performs a narrow diagnostic action. "
            "adapt_event_scaffold can generate a minimal observer atom; edit "
            "that source when you need custom filtering, summaries, or tool "
            "parameters.",
            "",
            "A runtime atom source must define MANIFEST and install(session, "
            "config). In install, use session.bus.on(Event.CHANNEL, handler) to observe "
            "events and session.register_tool(FunctionTool(...)) to expose a new "
            "tool to yourself. Use Pydantic models for tool parameters.",
            "",
            "To activate an atom, write the source in the task workspace, then "
            "call adapt_install_file or adapt_reload_file. Treat the atom as "
            "unavailable until the install or reload result is ok and a small "
            "smoke check shows the new behavior or tool is visible.",
            "",
            "Workspace helpers and runtime atoms are separate. Installing a "
            "runtime atom does not copy files into the task workspace, and "
            "editing workspace files does not automatically reload a runtime "
            "atom.",
        ]
        recent = list(self.recent_events)[-self.inject_events :]
        if recent:
            lines.extend(
                [
                    "",
                    "Recent adapt diagnostics:",
                    json.dumps(recent, indent=2, sort_keys=True),
                ]
            )
        block = "\n".join(lines)
        current = event.system or ""
        return {"system": f"{current}\n\n{block}" if current else block}

    async def _status(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_EmptyParams, args)
        if isinstance(params, ToolResult):
            return params
        payload = {
            "ok": True,
            "workspace_root": self.session.ctx.cwd,
            "scenario_local_extensions": self.session.ctx.scenario_dir is not None,
            "cwd": self.session.ctx.cwd,
            "operations_session_id": self.session.services.get("agent_env.session_id"),
            "loaded_atoms": [_atom_payload(atom) for atom in self.api.list_atoms()],
            "recent_events": list(self.recent_events),
            "runtime_note": (
                "File and shell tools operate in the task workspace. Runtime "
                "extension atoms are installed in the surrounding runtime and "
                "do not copy files into the workspace."
            ),
        }
        return _json_tool(payload)

    async def _events(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_EventsParams, args)
        if isinstance(params, ToolResult):
            return params
        limit = max(1, min(params.limit, self.max_events))
        return _json_tool({"ok": True, "events": list(self.recent_events)[-limit:]})

    async def _list_events(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_ListEventsParams, args)
        if isinstance(params, ToolResult):
            return params
        catalog = _filter_event_catalog(
            _discover_events(),
            channel=None,
            visibility=params.visibility,
        )
        return _json_tool(
            {
                "ok": True,
                "source": "runtime event catalog",
                "count": len(catalog),
                "events": [
                    _event_summary(
                        item,
                        observed_events=self.observed_events,
                        include_observed=params.include_observed,
                    )
                    for item in catalog
                ],
            }
        )

    async def _get_event(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_GetEventParams, args)
        if isinstance(params, ToolResult):
            return params
        matching = _filter_event_catalog(
            _discover_events(),
            channel=params.channel,
            visibility="all",
        )
        if not matching:
            return _json_tool(
                {"ok": False, "error": f"unknown event channel {params.channel!r}"},
                is_error=True,
            )
        event = matching[0]
        if params.include_observed:
            event["observed"] = _observed_payload(
                self.observed_events,
                params.channel,
            )
        return _json_tool(
            {
                "ok": True,
                "source": "runtime event catalog",
                "event": event,
            }
        )

    async def _event_scaffold(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_EventScaffoldParams, args)
        if isinstance(params, ToolResult):
            return params
        if not params.name.isidentifier():
            return _json_tool(
                {"ok": False, "error": f"invalid atom name {params.name!r}"},
                is_error=True,
            )
        tool_name = params.tool_name or f"{params.name}_summary"
        if not tool_name.isidentifier():
            return _json_tool(
                {"ok": False, "error": f"invalid tool name {tool_name!r}"},
                is_error=True,
            )
        matching = _filter_event_catalog(
            _discover_events(),
            channel=params.channel,
            visibility="all",
        )
        if not matching:
            return _json_tool(
                {"ok": False, "error": f"unknown event channel {params.channel!r}"},
                is_error=True,
            )
        event = matching[0]
        source = _scaffold_source(
            name=params.name,
            tool_name=tool_name,
            event=event,
            goal=params.goal,
        )
        return _json_tool(
            {
                "ok": True,
                "name": params.name,
                "tool_name": tool_name,
                "channel": params.channel,
                "source": source,
                "next_steps": [
                    "Write source to a .py file in the current operations backend.",
                    "Call adapt_install_file with that path.",
                    f"Call {tool_name} after the event has occurred.",
                ],
            }
        )

    async def _read_operations_text(self, path: str) -> str:
        # v2: resource writer pending
        return ""

    def _install_source(
        self,
        *,
        name: str,
        source: str,
        rationale: str,
        atom_config: dict[str, Any] | None,
        scope: str,
        pin_manifest: bool,
        source_origin: str,
    ) -> ToolResult:
        if atom_config is not None and not isinstance(atom_config, dict):
            return _json_tool(
                {"ok": False, "error": "config must be an object"}, is_error=True
            )
        target_path: str | None = None
        if scope == "scenario":
            if not self.session.ctx.scenario_dir:
                return _json_tool(
                    {"ok": False, "error": "scope=scenario requires session.ctx.scenario_dir"},
                    is_error=True,
                )
            target_path = str(Path(self.session.ctx.scenario_dir) / f"{name}.py")
        elif scope != "user":
            return _json_tool(
                {"ok": False, "error": "scope must be 'user' or 'scenario'"},
                is_error=True,
            )

        try:
            result = self.api.install_atom(
                name=name,
                source=source,
                target_path=target_path,
                config=atom_config,
                rationale=rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("adapt: install_atom raised: {}", exc)
            return _json_tool(
                {"ok": False, "error": f"install_atom raised: {exc}"}, is_error=True
            )

        payload = _result_payload(result)
        payload["scope"] = scope
        payload["source_origin"] = source_origin
        if not bool(payload.get("ok", False)):
            return _json_tool(payload, is_error=True)

        if scope == "scenario" and pin_manifest:
            pin_result = _pin_manifest(self.api, name)
            payload["manifest"] = pin_result
            if not pin_result.get("ok"):
                payload["ok"] = False
                payload["partial"] = "atom installed live, but manifest pin failed"
                return _json_tool(payload, is_error=True)
        return _json_tool(payload)

    async def _install(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_InstallParams, args)
        if isinstance(params, ToolResult):
            return params
        return self._install_source(
            name=params.name,
            source=params.source,
            rationale=params.rationale,
            atom_config=params.config,
            scope=params.scope,
            pin_manifest=params.pin_manifest,
            source_origin="inline",
        )

    async def _install_file(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_InstallFileParams, args)
        if isinstance(params, ToolResult):
            return params
        source_path = params.source_path
        try:
            source = await self._read_operations_text(source_path)
        except UnicodeDecodeError as exc:
            return _json_tool(
                {
                    "ok": False,
                    "source_path": source_path,
                    "error": f"source is not utf-8: {exc}",
                },
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("adapt: read source failed: {}", exc)
            return _json_tool(
                {
                    "ok": False,
                    "source_path": source_path,
                    "error": f"read source failed: {exc}",
                },
                is_error=True,
            )
        return self._install_source(
            name=params.name,
            source=source,
            rationale=params.rationale,
            atom_config=params.config,
            scope=params.scope,
            pin_manifest=params.pin_manifest,
            source_origin=source_path,
        )

    async def _reload(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_ReloadParams, args)
        if isinstance(params, ToolResult):
            return params
        try:
            result = self.api.reload_atom(
                params.name,
                params.source,
                rationale=params.rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("adapt: reload_atom raised: {}", exc)
            return _json_tool(
                {"ok": False, "error": f"reload_atom raised: {exc}"}, is_error=True
            )
        payload = _result_payload(result)
        return _json_tool(payload, is_error=not bool(payload.get("ok", False)))

    async def _reload_file(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_ReloadFileParams, args)
        if isinstance(params, ToolResult):
            return params
        source_path = params.source_path
        try:
            source = await self._read_operations_text(source_path)
        except UnicodeDecodeError as exc:
            return _json_tool(
                {
                    "ok": False,
                    "source_path": source_path,
                    "error": f"source is not utf-8: {exc}",
                },
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("adapt: read source failed: {}", exc)
            return _json_tool(
                {
                    "ok": False,
                    "source_path": source_path,
                    "error": f"read source failed: {exc}",
                },
                is_error=True,
            )
        try:
            result = self.api.reload_atom(
                params.name,
                source,
                rationale=params.rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("adapt: reload_atom raised: {}", exc)
            return _json_tool(
                {"ok": False, "error": f"reload_atom raised: {exc}"}, is_error=True
            )
        payload = _result_payload(result)
        payload["source_origin"] = source_path
        return _json_tool(payload, is_error=not bool(payload.get("ok", False)))

    async def _unload(self, args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_UnloadParams, args)
        if isinstance(params, ToolResult):
            return params
        try:
            result = self.api.unload_atom(params.name, agent_initiated=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("adapt: unload_atom raised: {}", exc)
            return _json_tool(
                {"ok": False, "error": f"unload_atom raised: {exc}"}, is_error=True
            )
        payload = _result_payload(result)
        if params.unpin_manifest:
            payload["manifest"] = _unpin_manifest(self.api, params.name)
        return _json_tool(payload, is_error=not bool(payload.get("ok", False)))

    def _register_tools(self) -> None:
        self.session.register_tool(
            FunctionTool(
                name="adapt_status",
                description=(
                    "Orientation tool for runtime adaptation. Call it to see the "
                    "workspace root, operations session id when available, currently "
                    "loaded atoms, and recent diagnostics before deciding whether to "
                    "install, reload, or debug a runtime atom."
                ),
                parameters=_EmptyParams,
                fn=self._status,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_events",
                description=(
                    "Inspect recent adaptation diagnostics and atom lifecycle records. "
                    "Use after adapt_install/adapt_reload fails, or after a custom "
                    "atom behaves unexpectedly, to see validation errors, install "
                    "errors, and reload outcomes."
                ),
                parameters=_EventsParams,
                fn=self._events,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_list_events",
                description=(
                    "Step 1 for observing yourself through runtime events. Lists "
                    "event hook channels as a compact catalog with event type, "
                    "visibility, allowed effects, short summaries, and optional "
                    "observed counts. Choose a channel here, then call adapt_get_event."
                ),
                parameters=_ListEventsParams,
                fn=self._list_events,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_get_event",
                description=(
                    "Step 2 for event observation. Given one channel from "
                    "adapt_list_events, returns the exact event payload fields, "
                    "handler return/mutation contract, import snippet, subscription "
                    "example, notes, and observed counts so you can write a correct "
                    "observer atom."
                ),
                parameters=_GetEventParams,
                fn=self._get_event,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_event_scaffold",
                description=(
                    "Generate starter source for a runtime observer atom. The source "
                    "subscribes to one event channel, stores recent event snapshots, "
                    "and registers a summary tool. This only returns source; edit it "
                    "as needed, then install it with adapt_install or adapt_install_file."
                ),
                parameters=_EventScaffoldParams,
                fn=self._event_scaffold,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_install",
                description=(
                    "Install and activate a new runtime atom from inline source. Use "
                    "this when the full source is already in the tool arguments. "
                    "On ok=true, any tools registered by the atom are available in "
                    "the current session, and the source is persisted per the scope "
                    "parameter (default: <cwd>/.agentm/atoms, auto-loaded in future "
                    "sessions). Scenario-scope installs can partially succeed (atom "
                    "live but manifest pin failed -> ok=false with a `partial` "
                    "field). Prefer adapt_install_file when you wrote the source as "
                    "a workspace file."
                ),
                parameters=_InstallParams,
                fn=self._install,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_install_file",
                description=(
                    "Install and activate a new runtime atom from a workspace source "
                    "file. Use this after writing or editing atom source with file/bash "
                    "tools. On ok=true, the runtime has validated and loaded it, any "
                    "new tools it registers are available in the current session, and "
                    "the atom is persisted per the scope parameter (default: "
                    "<cwd>/.agentm/atoms, auto-loaded in future sessions). "
                    "Scenario-scope installs can partially succeed (atom live but "
                    "manifest pin failed -> ok=false with a `partial` field)."
                ),
                parameters=_InstallFileParams,
                fn=self._install_file,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_reload",
                description=(
                    "Replace an already-loaded runtime atom from inline source. Use "
                    "when you need to fix or improve a custom atom. Reload is "
                    "transactional: on validation or install failure, the previous "
                    "live atom remains active and structured error details are returned."
                ),
                parameters=_ReloadParams,
                fn=self._reload,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_reload_file",
                description=(
                    "Replace an already-loaded runtime atom from a workspace source "
                    "file. Use this after editing atom source with file/bash tools. "
                    "Reload is transactional: on failure, the previous live atom "
                    "remains active and diagnostics explain what to fix."
                ),
                parameters=_ReloadFileParams,
                fn=self._reload_file,
            )
        )
        self.session.register_tool(
            FunctionTool(
                name="adapt_unload",
                description=(
                    "Remove a loaded runtime atom from the current session when it is "
                    "misleading, noisy, or no longer useful. The source file is never "
                    "deleted by this tool — delete it separately for permanent "
                    "removal. unpin_manifest only removes the atom's `local:` entry "
                    "from the scenario manifest (no effect for user-scope atoms)."
                ),
                parameters=_UnloadParams,
                fn=self._unload,
            )
        )


def install(session: Any, config: AdaptConfig) -> None:
    _AdaptRuntime(session, config).install()
