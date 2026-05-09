from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from pathlib import Path
import sys
import types
from typing import Any, cast

import pytest

from agentm.core.abi import BeforeSendToLlmEvent, EventBus, MessageEnd, Model, StreamFn
from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions import ExtensionManifest
from agentm.extensions.builtin import cost_budget, skill_loader
from agentm.extensions.discover import discover_builtin
from agentm.extensions.loader import ScenarioLoadError, sort_extensions_by_requires
from agentm.harness.catalog import DefaultProjectLayout
from agentm.harness.events import ResourcesDiscoverEvent, SessionReadyEvent
from agentm.harness.extension import ProviderConfig
from agentm.harness.session_helpers import missing_required_fields


def _module(name: str, manifest: ExtensionManifest) -> str:
    module = types.ModuleType(name)
    setattr(module, "MANIFEST", manifest)
    setattr(module, "install", lambda api, config: None)
    sys.modules[name] = module
    return name


def test_sort_extensions_by_requires_orders_dependencies() -> None:
    dep = _module(
        "tests.unit.extensions._dep_atom",
        ExtensionManifest(name="dep_atom", description="", registers=()),
    )
    child = _module(
        "tests.unit.extensions._child_atom",
        ExtensionManifest(
            name="child_atom",
            description="",
            registers=(),
            requires=("dep_atom",),
        ),
    )
    try:
        ordered = sort_extensions_by_requires([(child, {}), (dep, {})])
    finally:
        sys.modules.pop(dep, None)
        sys.modules.pop(child, None)

    assert [module for module, _ in ordered] == [dep, child]


def test_sort_extensions_by_requires_rejects_missing_dependency() -> None:
    child = _module(
        "tests.unit.extensions._missing_child_atom",
        ExtensionManifest(
            name="missing_child_atom",
            description="",
            registers=(),
            requires=("absent_atom",),
        ),
    )
    try:
        with pytest.raises(ScenarioLoadError, match="requires 'absent_atom'"):
            sort_extensions_by_requires([(child, {})])
    finally:
        sys.modules.pop(child, None)


def test_sort_extensions_by_requires_rejects_duplicate_extension_entries() -> None:
    module = _module(
        "tests.unit.extensions._duplicate_atom",
        ExtensionManifest(name="duplicate_atom", description="", registers=()),
    )
    try:
        with pytest.raises(ScenarioLoadError, match="loaded more than once"):
            sort_extensions_by_requires([(module, {"a": 1}), (module, {"a": 2})])
    finally:
        sys.modules.pop(module, None)


@pytest.mark.asyncio
async def test_cost_budget_warns_once_for_unpriced_provider() -> None:
    bus = EventBus()
    diagnostics: list[DiagnosticEvent] = []
    bus.on(DiagnosticEvent.CHANNEL, diagnostics.append)

    class _Api:
        events = bus
        model = Model(id="m", provider="unknown", context_window=1000, max_output_tokens=10)

        def on(self, channel: str, handler: Callable[..., Any]) -> None:
            bus.on(channel, handler)

    cost_budget.install(cast(Any, _Api()), {"limit": 1.0, "pricing": {}})
    event = BeforeSendToLlmEvent(
        messages=[],
        model=Model(id="m", provider="unknown", context_window=1000, max_output_tokens=10),
        tools=[],
        system=None,
    )

    await bus.emit(BeforeSendToLlmEvent.CHANNEL, event)
    await bus.emit(BeforeSendToLlmEvent.CHANNEL, event)

    warnings = [item for item in diagnostics if item.level == "warning"]
    assert len(warnings) == 1
    assert "unknown" in warnings[0].message


@pytest.mark.asyncio
async def test_skill_loader_warns_on_unknown_resource_response_key(tmp_path: Path) -> None:
    bus = EventBus()
    diagnostics: list[DiagnosticEvent] = []
    bus.on(DiagnosticEvent.CHANNEL, diagnostics.append)

    def peer_handler(_: ResourcesDiscoverEvent) -> dict[str, Any]:
        return {"skill_path": ["typo"]}

    setattr(peer_handler, "_agentm_obs_owner", "peer_atom")
    bus.on(ResourcesDiscoverEvent.CHANNEL, peer_handler)

    class _Skills:
        def load_skills(self, **_: Any) -> tuple[list[Any], list[Any]]:
            return [], []

        def format_skills_for_prompt(self, skills: list[Any]) -> str:
            assert skills == []
            return ""

    class _Api:
        cwd = str(tmp_path)
        events = bus
        skills = _Skills()

        def on(self, channel: str, handler: Callable[..., Any]) -> None:
            bus.on(channel, handler)

        def add_observer(self, callback: Any) -> Callable[[], None]:
            return bus.add_observer(callback)

    await skill_loader.install(
        cast(Any, _Api()), {"include_defaults": False, "inherit_claude": False}
    )
    await bus.emit(
        SessionReadyEvent.CHANNEL,
        SessionReadyEvent(
            cwd=str(tmp_path),
            session_id="s",
            tool_names=(),
            command_names=(),
            extension_module_paths=(),
            model=None,
            root_session_id="s",
            task_id=None,
            persona=None,
        ),
    )

    assert any(
        item.level == "warning"
        and "skill_path" in item.message
        and "peer_atom" in item.message
        for item in diagnostics
    )


class _DummyWriter:
    def current_version_for_path(self, _: str) -> str | None:
        return None


class _DummyCatalog:
    def compute_atom_hash(self, source: str) -> str:
        return str(len(source))

    def compute_active_set_fingerprint(
        self, *, loaded: dict[str, str], scenario: str | None, core_hash: str | None
    ) -> dict[str, Any]:
        return {"loaded": loaded, "scenario": scenario, "core_hash": core_hash}


class _DummyPromptTemplates:
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    def register_prompt(self, name: str, body: str) -> None:
        self.prompts[name] = body

    def get_prompt(self, name: str) -> str | None:
        return self.prompts.get(name)

    def load_prompt_templates(self, **_: Any) -> list[Any]:
        return []

    def expand_prompt_template(self, text: str, _: list[Any]) -> str:
        return text


class _DummySkills:
    def load_skills(self, **_: Any) -> tuple[list[Any], list[Any]]:
        return [], []

    def format_skills_for_prompt(self, skills: list[Any]) -> str:
        assert skills == []
        return ""


class _DummySession:
    def get_branch(self) -> list[Any]:
        return []

    def get_messages(self) -> list[Any]:
        return []

    def append_entry(self, _: str, __: dict[str, Any]) -> str:
        return "entry"

    def get_loop_config(self) -> dict[str, Any]:
        return {}


class _DummyOperations:
    file: object = object()
    bash: object = object()


class _InstallApi:
    def __init__(self, tmp_path: Path) -> None:
        self.cwd = str(tmp_path)
        self.session_id = "session"
        self.root_session_id = "session"
        self.task_id = None
        self.persona = None
        self.events = EventBus()
        self.tools: list[Any] = []
        self.model = Model(id="m", provider="fake", context_window=1000, max_output_tokens=10)
        self.provider = None
        self.session = _DummySession()
        self.catalog = _DummyCatalog()
        self.prompt_templates = _DummyPromptTemplates()
        self.skills = _DummySkills()
        self.compaction = object()
        self._services: dict[str, Any] = {}
        self._writer = _DummyWriter()
        self._layout = DefaultProjectLayout(tmp_path)
        self.registered_providers: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self.renderers: dict[str, Any] = {}

    def on(self, channel: str, handler: Callable[..., Any], **_: Any) -> None:
        self.events.on(channel, handler)

    def add_observer(self, callback: Any) -> Callable[[], None]:
        return self.events.add_observer(callback)

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def register_provider(self, name: str, provider: Any) -> None:
        self.registered_providers[name] = provider

    def register_command(self, name: str, command: Any) -> None:
        self.commands[name] = command

    def register_message_renderer(self, name: str, renderer: Any) -> None:
        self.renderers[name] = renderer

    def get_operations(self) -> _DummyOperations:
        return _DummyOperations()

    def get_resource_writer(self) -> _DummyWriter:
        return self._writer

    def get_project_layout(self) -> DefaultProjectLayout:
        return self._layout

    def set_service(self, name: str, service: Any) -> None:
        self._services[name] = service

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    async def spawn_child_session(self, **_: Any) -> Any:
        raise AssertionError("schema roundtrip must not execute child sessions")


def _schema_placeholder(property_schema: Any) -> Any:
    if not isinstance(property_schema, dict):
        return ""
    schema_type = property_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "string")
    if schema_type in {"number", "integer"}:
        return property_schema.get("minimum", 0)
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}
    return ""


@pytest.mark.asyncio
async def test_builtin_schema_defaults_install_roundtrip(tmp_path: Path) -> None:
    for entry in discover_builtin().values():
        schema = entry.manifest.config_schema
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties", {})
        config = {
            str(key): value["default"]
            for key, value in properties.items()
            if isinstance(value, dict) and "default" in value
        }
        for key in schema.get("required", []):
            config.setdefault(str(key), _schema_placeholder(properties.get(key)))
        if entry.name == "inherit_provider":

            async def _stream(**_: Any) -> AsyncIterator[MessageEnd]:
                if False:
                    yield MessageEnd(type="message_end", stop_reason="end_turn")
                raise AssertionError("schema roundtrip must not stream")

            config["provider"] = ProviderConfig(
                stream_fn=cast(StreamFn, _stream),
                model=Model(
                    id="m",
                    provider="fake",
                    context_window=1000,
                    max_output_tokens=10,
                ),
                name="fake",
            )
        assert missing_required_fields(entry.manifest, config) == []
        result = entry.module.install(cast(Any, _InstallApi(tmp_path)), config)
        if inspect.isawaitable(result):
            await result
