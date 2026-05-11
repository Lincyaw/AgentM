from __future__ import annotations

from pathlib import Path


def test_persona_metadata_parser_surfaces_input_schema_and_budget_defaults() -> None:
    from agentm_rca import prompt_loader

    personas = prompt_loader._load_personas()

    critic = personas["critic"]
    assert critic["input_schema"]["required"] == [
        "objective",
        "current_conclusion",
        "supporting_evidence",
        "output_format",
    ]
    assert critic["budget_defaults"] == {"max_turns": 12}
    assert "brief_rejection" in critic["artifact_kinds"]

    block = prompt_loader.available_agents_block(personas, include_input_schema=True)
    assert "<available_agents>" in block
    assert '<input_schema advisory="true">' in block
    assert "objective, current_conclusion, supporting_evidence, output_format" in block


def test_resolve_handler_accepts_typed_subagent_event() -> None:
    import asyncio

    from agentm.core.abi.events import ResolveSubagentEvent, SessionReadyEvent
    from agentm_rca import prompt_loader

    handlers: dict[str, object] = {}

    class _Api:
        def on(self, channel, handler):
            handlers[channel] = handler

    asyncio.run(
        prompt_loader.install(_Api(), {"prompt": "orchestrator.md", "personas": True})
    )
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


def test_personas_disabled_skips_resolve_handler() -> None:
    import asyncio

    from agentm.core.abi.events import ResolveSubagentEvent, SessionReadyEvent
    from agentm_rca import prompt_loader

    handlers: dict[str, object] = {}

    class _Api:
        def on(self, channel, handler):
            handlers[channel] = handler

    asyncio.run(
        prompt_loader.install(_Api(), {"prompt": "investigator.md", "personas": False})
    )

    assert SessionReadyEvent.CHANNEL in handlers
    assert ResolveSubagentEvent.CHANNEL not in handlers
