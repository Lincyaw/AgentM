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

    critic = personas["critic"]
    assert critic["input_schema"]["required"] == [
        "objective",
        "current_conclusion",
        "supporting_evidence",
        "output_format",
    ]
    assert critic["budget_defaults"] == {"max_turns": 12}
    assert "brief_rejection" in critic["artifact_kinds"]

    block = module.available_agents_block(personas, include_input_schema=True)
    assert "<available_agents>" in block
    assert '<input_schema advisory="true">' in block
    assert "objective, current_conclusion, supporting_evidence, output_format" in block


def test_resolve_handler_accepts_typed_subagent_event() -> None:
    import asyncio

    from agentm.harness.events import ResolveSubagentEvent, SessionReadyEvent

    module = _load_module()
    handlers = {}

    class _Api:
        def on(self, channel, handler):
            handlers[channel] = handler

    asyncio.run(module.install(_Api(), {}))
    asyncio.run(
        handlers[SessionReadyEvent.CHANNEL](
            SessionReadyEvent(
                cwd=str(Path.cwd()),
                session_id="s",
                tool_names=(),
                command_names=(),
                extension_module_paths=(),
                model=None,
                root_session_id="s",
            )
        )
    )

    resolved = handlers[ResolveSubagentEvent.CHANNEL](
        ResolveSubagentEvent(name="critic")
    )

    assert resolved["body"]
    assert resolved["budget_defaults"] == {"max_turns": 12}
