"""Scenario-local ``adapt`` atom for online task self-modification.

The atom is intentionally small: it gives a single agent enough control
to install/reload its own helper atoms, see lifecycle diagnostics, and keep
scenario-local atoms persistent when explicitly requested. In ARL-backed
scenarios, these tools run in the host AgentM process; bash/file tools still
operate in the remote sandbox.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    BeforeAgentStartEvent,
    DiagnosticEvent,
    ExtensionAPI,
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
        "Online self-adaptation tools for task agents: install/reload/unload "
        "agent-authored atoms and inspect recent extension diagnostics."
    ),
    registers=(
        "tool:adapt_status",
        "tool:adapt_events",
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
        description="Maximum number of recent adapt events to return.",
    )


class _InstallParams(_StrictParams):
    name: str = Field(
        description="Atom name; must be a valid identifier and match MANIFEST.name."
    )
    source: str = Field(
        description=(
            "Full Python source for the atom. Must define MANIFEST and "
            "install(api, config), and pass the AgentM atom contract."
        )
    )
    rationale: str = Field(description="Why this atom helps the current task.")
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional config passed to install(api, config).",
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
        description="Atom name; must be a valid identifier and match MANIFEST.name."
    )
    source_path: str = Field(
        description=(
            "Path to the atom source file in the current operations backend. "
            "Under ARL this is read from the remote sandbox, usually /app."
        )
    )
    rationale: str = Field(description="Why this atom helps the current task.")
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional config passed to install(api, config).",
    )
    scope: Literal["user", "scenario"] = Field(
        default="user",
        description="Where to persist the installed host-side atom.",
    )
    pin_manifest: bool = Field(
        default=True,
        description="Only for scope=scenario; append `local: <name>` to the manifest.",
    )


class _ReloadParams(_StrictParams):
    name: str = Field(description="Loaded atom name to replace.")
    source: str = Field(description="Full replacement Python source for the atom.")
    rationale: str = Field(description="Why this reload is needed.")


class _ReloadFileParams(_StrictParams):
    name: str = Field(description="Loaded atom name to replace.")
    source_path: str = Field(
        description=(
            "Path to replacement source in the current operations backend. "
            "Under ARL this is read from the remote sandbox, usually /app."
        )
    )
    rationale: str = Field(description="Why this reload is needed.")


class _UnloadParams(_StrictParams):
    name: str = Field(description="Loaded atom name to unload from the running session.")
    rationale: str = Field(description="Why this atom should be unloaded.")
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


def _parse_params(model_cls: type[_Params], args: dict[str, Any]) -> _Params | ToolResult:
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


def _find_manifest_path(api: ExtensionAPI) -> Path | None:
    scenario_dir_raw = api.scenario_dir
    if not scenario_dir_raw:
        return None
    scenario_dir = Path(scenario_dir_raw)
    wanted = api.scenario
    candidates = sorted(scenario_dir.glob("manifest*.yaml"))
    for path in candidates:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
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


def _pin_manifest(api: ExtensionAPI, name: str) -> dict[str, Any]:
    manifest_path = _find_manifest_path(api)
    if manifest_path is None:
        return {"ok": False, "error": "current scenario manifest not found"}
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": str(manifest_path), "error": str(exc)}
    if not isinstance(loaded, dict):
        return {"ok": False, "path": str(manifest_path), "error": "manifest root is not a mapping"}
    extensions = loaded.get("extensions")
    if not isinstance(extensions, list):
        return {"ok": False, "path": str(manifest_path), "error": "manifest extensions is not a list"}
    if _local_entry_index(extensions, name) is not None:
        return {"ok": True, "path": str(manifest_path), "changed": False}
    extensions.append({"local": name})
    manifest_path.write_text(
        yaml.safe_dump(loaded, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return {"ok": True, "path": str(manifest_path), "changed": True}


def _unpin_manifest(api: ExtensionAPI, name: str) -> dict[str, Any]:
    manifest_path = _find_manifest_path(api)
    if manifest_path is None:
        return {"ok": False, "error": "current scenario manifest not found"}
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": str(manifest_path), "error": str(exc)}
    if not isinstance(loaded, dict):
        return {"ok": False, "path": str(manifest_path), "error": "manifest root is not a mapping"}
    extensions = loaded.get("extensions")
    if not isinstance(extensions, list):
        return {"ok": False, "path": str(manifest_path), "error": "manifest extensions is not a list"}
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


def install(api: ExtensionAPI, config: AdaptConfig) -> None:
    max_events = max(1, config.max_events)
    inject_events = max(0, min(config.inject_events, max_events))
    events: deque[dict[str, Any]] = deque(maxlen=max_events)

    def _record(kind: str, payload: dict[str, Any]) -> None:
        events.append({"kind": kind, **payload})

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        if event.level == "error":
            _record(
                "diagnostic",
                {
                    "level": event.level,
                    "source": event.source,
                    "message": event.message,
                },
            )

    def _on_extension_install(event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        _record(
            "extension_install",
            {
                "module_path": event.module_path,
                "phase": event.phase,
                "trigger": event.trigger,
                "error": event.error,
            },
        )

    def _on_extension_reload(event: ExtensionReloadEvent) -> None:
        _record(
            "extension_reload",
            {
                "name": event.name,
                "trigger": event.trigger,
                "error": event.error,
                "is_self_modify": event.is_self_modify,
            },
        )

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str]:
        lines = [
            "# Adapt",
            "",
            "You may improve your task-specific feedback loop before and during "
            "the task. Use normal file and bash tools for scripts/checks that "
            "belong in the task workspace. Use adapt tools only for AgentM "
            "native atoms that run in the host AgentM process.",
            "",
            "In ARL scenarios, bash/file tools execute in the remote sandbox, "
            "while adapt_install/adapt_reload affect host-side AgentM atoms. "
            "Installing a native atom does not copy files into /app.",
            "",
            "Before relying on a new helper atom or script, run a small smoke "
            "check and inspect failures with adapt_events/adapt_status.",
        ]
        recent = list(events)[-inject_events:]
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
        event.system = f"{current}\n\n{block}" if current else block
        return {"system": event.system}

    async def _status(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_EmptyParams, args)
        if isinstance(params, ToolResult):
            return params
        payload = {
            "ok": True,
            "scenario": api.scenario,
            "scenario_dir": api.scenario_dir,
            "cwd": api.cwd,
            "agent_env_session_id": api.get_service("agent_env.session_id"),
            "loaded_atoms": [_atom_payload(atom) for atom in api.list_atoms()],
            "recent_events": list(events),
            "arl_note": (
                "When operations backend is agent_env, bash/file tools run in "
                "the ARL sandbox; adapt native atoms run in the host AgentM "
                "process and are installed through the atom reloader."
            ),
        }
        return _json_tool(payload)

    async def _events(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_EventsParams, args)
        if isinstance(params, ToolResult):
            return params
        limit = max(1, min(params.limit, max_events))
        return _json_tool({"ok": True, "events": list(events)[-limit:]})

    async def _read_operations_text(path: str) -> str:
        data = await api.get_operations().file.read_file(path)
        return data.decode("utf-8")

    def _install_source(
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
            return _json_tool({"ok": False, "error": "config must be an object"}, is_error=True)
        target_path: str | None = None
        if scope == "scenario":
            if not api.scenario_dir:
                return _json_tool(
                    {"ok": False, "error": "scope=scenario requires api.scenario_dir"},
                    is_error=True,
                )
            target_path = str(Path(api.scenario_dir) / f"{name}.py")
        elif scope != "user":
            return _json_tool(
                {"ok": False, "error": "scope must be 'user' or 'scenario'"},
                is_error=True,
            )

        try:
            result = api.install_atom(
                name=name,
                source=source,
                target_path=target_path,
                config=atom_config,
                rationale=rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_tool({"ok": False, "error": f"install_atom raised: {exc}"}, is_error=True)

        payload = _result_payload(result)
        payload["scope"] = scope
        payload["source_origin"] = source_origin
        if not bool(payload.get("ok", False)):
            return _json_tool(payload, is_error=True)

        if scope == "scenario" and pin_manifest:
            pin_result = _pin_manifest(api, name)
            payload["manifest"] = pin_result
            if not pin_result.get("ok"):
                payload["ok"] = False
                payload["partial"] = "atom installed live, but manifest pin failed"
                return _json_tool(payload, is_error=True)
        return _json_tool(payload)

    async def _install(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_InstallParams, args)
        if isinstance(params, ToolResult):
            return params
        return _install_source(
            name=params.name,
            source=params.source,
            rationale=params.rationale,
            atom_config=params.config,
            scope=params.scope,
            pin_manifest=params.pin_manifest,
            source_origin="inline",
        )

    async def _install_file(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_InstallFileParams, args)
        if isinstance(params, ToolResult):
            return params
        source_path = params.source_path
        try:
            source = await _read_operations_text(source_path)
        except UnicodeDecodeError as exc:
            return _json_tool(
                {"ok": False, "source_path": source_path, "error": f"source is not utf-8: {exc}"},
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_tool(
                {"ok": False, "source_path": source_path, "error": f"read source failed: {exc}"},
                is_error=True,
            )
        return _install_source(
            name=params.name,
            source=source,
            rationale=params.rationale,
            atom_config=params.config,
            scope=params.scope,
            pin_manifest=params.pin_manifest,
            source_origin=source_path,
        )

    async def _reload(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_ReloadParams, args)
        if isinstance(params, ToolResult):
            return params
        try:
            result = api.reload_atom(
                params.name,
                params.source,
                rationale=params.rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_tool({"ok": False, "error": f"reload_atom raised: {exc}"}, is_error=True)
        payload = _result_payload(result)
        return _json_tool(payload, is_error=not bool(payload.get("ok", False)))

    async def _reload_file(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_ReloadFileParams, args)
        if isinstance(params, ToolResult):
            return params
        source_path = params.source_path
        try:
            source = await _read_operations_text(source_path)
        except UnicodeDecodeError as exc:
            return _json_tool(
                {"ok": False, "source_path": source_path, "error": f"source is not utf-8: {exc}"},
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_tool(
                {"ok": False, "source_path": source_path, "error": f"read source failed: {exc}"},
                is_error=True,
            )
        try:
            result = api.reload_atom(
                params.name,
                source,
                rationale=params.rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_tool({"ok": False, "error": f"reload_atom raised: {exc}"}, is_error=True)
        payload = _result_payload(result)
        payload["source_origin"] = source_path
        return _json_tool(payload, is_error=not bool(payload.get("ok", False)))

    async def _unload(args: dict[str, Any]) -> ToolResult:
        params = _parse_params(_UnloadParams, args)
        if isinstance(params, ToolResult):
            return params
        try:
            result = api.unload_atom(params.name, agent_initiated=True)
        except Exception as exc:  # noqa: BLE001
            return _json_tool({"ok": False, "error": f"unload_atom raised: {exc}"}, is_error=True)
        payload = _result_payload(result)
        if params.unpin_manifest:
            payload["manifest"] = _unpin_manifest(api, params.name)
        return _json_tool(payload, is_error=not bool(payload.get("ok", False)))

    api.on(DiagnosticEvent.CHANNEL, _on_diagnostic)
    api.on(ExtensionInstallEvent.CHANNEL, _on_extension_install)
    api.on(ExtensionReloadEvent.CHANNEL, _on_extension_reload)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject)

    api.register_tool(
        FunctionTool(
            name="adapt_status",
            description=(
                "Inspect adapt state: current scenario, ARL session id when "
                "available, loaded atoms, and recent self-modification diagnostics."
            ),
            parameters=_EmptyParams,
            fn=_status,
        )
    )
    api.register_tool(
        FunctionTool(
            name="adapt_events",
            description="Return recent adapt diagnostics and atom lifecycle events.",
            parameters=_EventsParams,
            fn=_events,
        )
    )
    api.register_tool(
        FunctionTool(
            name="adapt_install",
            description=(
                "Install a new agent-authored native AgentM atom and activate it "
                "in the current session. Default scope='user' persists under "
                "<cwd>/.agentm/atoms, which works for ARL sessions because the "
                "host atom reloader writes it. scope='scenario' writes a "
                "scenario-local atom and can pin `local: name` in the current "
                "manifest."
            ),
            parameters=_InstallParams,
            fn=_install,
        )
    )
    api.register_tool(
        FunctionTool(
            name="adapt_install_file",
            description=(
                "Install a new host-side AgentM atom from a source file in the "
                "current operations backend. In ARL, write the source under /app "
                "with file/bash tools, then call this tool to read it back and "
                "install it through the host atom reloader."
            ),
            parameters=_InstallFileParams,
            fn=_install_file,
        )
    )
    api.register_tool(
        FunctionTool(
            name="adapt_reload",
            description=(
                "Transactionally reload a loaded atom from full source. On "
                "validation or install failure, AgentM restores the previous "
                "live atom and returns structured error details."
            ),
            parameters=_ReloadParams,
            fn=_reload,
        )
    )
    api.register_tool(
        FunctionTool(
            name="adapt_reload_file",
            description=(
                "Transactionally reload a loaded host-side AgentM atom from a "
                "source file in the current operations backend. In ARL this "
                "lets the agent edit source under /app and ask the host to load "
                "that source without exposing evaluation files."
            ),
            parameters=_ReloadFileParams,
            fn=_reload_file,
        )
    )
    api.register_tool(
        FunctionTool(
            name="adapt_unload",
            description=(
                "Unload a live atom from the current session. The source file is "
                "kept unless a scenario manifest pin is explicitly removed."
            ),
            parameters=_UnloadParams,
            fn=_unload,
        )
    )
