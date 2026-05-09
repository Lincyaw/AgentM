from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
import types
from typing import Any, cast

import pytest

from agentm.core.abi import BeforeSendToLlmEvent, EventBus, Model
from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions import ExtensionManifest
from agentm.extensions.builtin import cost_budget, skill_loader
from agentm.extensions.discover import discover_builtin
from agentm.extensions.loader import ScenarioLoadError, sort_extensions_by_requires
from agentm.harness.events import ResourcesDiscoverEvent, SessionReadyEvent
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


def test_builtin_schema_defaults_roundtrip() -> None:
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
            config.setdefault(str(key), "")
        assert missing_required_fields(entry.manifest, config) == []
