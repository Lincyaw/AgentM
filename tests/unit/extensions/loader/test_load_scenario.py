from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from agentm.extensions.loader import ScenarioLoadError, load_scenario


_BUILTIN_SCENARIOS = [
    "general_purpose",
    "trajectory_analysis",
    "plan_mode",
]


def test_load_scenario_resolves_builtin_name() -> None:
    loaded = load_scenario("general_purpose")

    assert loaded == [
        ("agentm.extensions.builtin.tool_read", {}),
        ("agentm.extensions.builtin.tool_bash", {}),
        ("agentm.extensions.builtin.tool_edit", {}),
        ("agentm.extensions.builtin.tool_write", {}),
        (
            "agentm.extensions.builtin.system_prompt",
            {
                "prompt": (
                    "You are a helpful coding assistant. Use the available tools to read,\n"
                    "modify, and execute code. Be careful with destructive operations.\n"
                )
            },
        ),
    ]
    assert loaded[0][0] == "agentm.extensions.builtin.tool_read"


def test_load_scenario_accepts_absolute_path() -> None:
    scenario_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "agentm"
        / "extensions"
        / "scenarios"
        / "general_purpose.yaml"
    )

    assert load_scenario(str(scenario_path)) == load_scenario("general_purpose")


def test_load_scenario_rejects_non_list_extensions(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("extensions: not-a-list\n", encoding="utf-8")

    with pytest.raises(ScenarioLoadError, match="extensions"):
        load_scenario(str(path))


def test_load_scenario_rejects_missing_module_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("extensions:\n  - config: {}\n", encoding="utf-8")

    with pytest.raises(ScenarioLoadError, match="module"):
        load_scenario(str(path))


@pytest.mark.parametrize("name", _BUILTIN_SCENARIOS)
def test_builtin_scenarios_resolve_importable_installers(name: str) -> None:
    loaded = load_scenario(name)

    assert loaded
    for module_path, _config in loaded:
        module = import_module(module_path)
        install = getattr(module, "install", None)
        assert callable(install), f"{module_path} must export install()"
