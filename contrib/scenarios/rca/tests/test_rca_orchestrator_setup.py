from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    rca_root = Path(__file__).resolve().parents[1]
    module_name = "agentm._tests.rca_orchestrator_setup"
    if module_name in sys.modules:
        return sys.modules[module_name]
    file_path = rca_root / "orchestrator_setup.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_persona_metadata_parser_surfaces_input_schema_and_budget_defaults() -> None:
    module = _load_module()
    personas = module._load_personas()

    scout = personas["scout"]
    assert scout["input_schema"]["required"] == [
        "objective",
        "scope_services",
        "output_format",
    ]
    assert scout["budget_defaults"] == {"max_tool_calls": 8, "max_turns": 6}
    assert "brief_rejection" in scout["artifact_kinds"]

    block = module._format_available_agents_block(personas)
    assert "<available_agents>" in block
    assert '<input_schema advisory="true">' in block
    assert "objective, scope_services, output_format" in block
