"""Gateway must deliver the creation-time ``session_ready`` frame.

Fail-stop for the single-process gateway's outbound-sink wiring: a freshly
created chat session has to push its ``session_ready`` envelope — carrying the
slash-command catalog the chat client autocompletes from — to the connected
peer. The frame fires *inside* ``create_agent_session`` (the factory emits
``SessionReadyEvent`` right after the runtime is built), so the ``wire_driver``
atom must already be installed and subscribed by then. The gateway achieves
this by seeding the wire services into ``AgentSessionConfig.initial_services``
and listing ``wire_driver`` in ``extra_extensions`` so it installs *during*
``create()`` — stamping it after ``create()`` returns drops this first frame.

This drives the real gateway ``SessionManager.get_or_create`` +
``outbound_sink`` path (not a bare in-process session): the factory mirrors the
gateway's ``_build_session_factory`` (seed + mount wire_driver), the outbound
sink the gateway hands to ``SessionManager`` is captured here, and we assert a
``session_ready`` body arrives whose ``command_names`` includes ``compact``
(registered by the ``llm_compaction`` atom, which ships in the
``general_purpose`` scenario).

Pre-fix (services + wire_driver stamped *after* ``create()``): no
``session_ready`` body is delivered. Post-fix: the creation-time frame arrives
with the command catalog.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.session import AgentSession
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.session_manager import SessionManager
from agentm.gateway.wire import InboundBody


def _install_provider_module(name: str) -> str:
    """A stub provider that ends every turn immediately — enough to build the
    session; the test never drives a real prompt."""

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="ok")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-ready",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="fake-ready",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-ready",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _inbound() -> InboundBody:
    return InboundBody(
        channel="terminal",
        chat_id="chat-1",
        content="hi",
        sender_id="user-1",
    )


@pytest.mark.asyncio
async def test_gateway_delivers_creation_time_session_ready(tmp_path: Any) -> None:
    provider_module = _install_provider_module(
        "tests.integration._fake_session_ready_provider"
    )
    outbound: list[dict[str, Any]] = []

    async def _sink(body: dict[str, Any]) -> None:
        outbound.append(body)

    async def _factory(
        cwd: str,
        session_key: str,
        scenario: str | None,
        resume: str | None,
        wire_services: dict[str, Any],
    ) -> Any:
        del resume
        config = AgentSessionConfig(
            cwd=cwd,
            provider=(provider_module, {}),
            scenario=scenario,
            resource_loader=InMemoryResourceLoader(),
            # Mirror the gateway's _build_session_factory: seed the wire
            # services and mount wire_driver so it installs DURING create()
            # and forwards the creation-time SessionReadyEvent.
            initial_services=dict(wire_services),
            extra_extensions=[("agentm.extensions.builtin.wire_driver", {})],
        )
        return await AgentSession.create(config)

    mgr = SessionManager(
        cwd=str(tmp_path),
        chat_map=ChatSessionMap(tmp_path / "session_map.json"),
        session_factory=_factory,
        outbound_sink=_sink,
    )

    sess = await mgr.get_or_create("key-1", "general_purpose", _inbound())
    try:
        ready = [
            body
            for body in outbound
            if body.get("metadata", {}).get("kind") == "session_ready"
        ]
        assert ready, (
            "gateway dropped the creation-time session_ready frame; "
            "outbound kinds seen: "
            f"{[b.get('metadata', {}).get('kind') for b in outbound]!r}"
        )
        command_names = ready[-1]["metadata"]["command_names"]
        assert "compact" in command_names, (
            "session_ready frame must carry the atom-registered slash-command "
            f"catalog; command_names={command_names!r}"
        )
    finally:
        await sess.shutdown()
