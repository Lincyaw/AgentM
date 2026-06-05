from __future__ import annotations


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
