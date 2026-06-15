"""A clean turn persists a ``turn_committed`` boundary marker."""

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
    ProviderConfig,
    TextContent,
)
from agentm.core.abi.session import ENTRY_TYPE_MESSAGE, ENTRY_TYPE_TURN_COMMITTED
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.session import AgentSession


def _install_provider(name: str) -> str:
    async def _stream(
        *, messages: Any, model: Model, tools: Any, system: Any = None,
        signal: Any = None, thinking: str = "off",
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
            "fake",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(id="fake", provider="fake", context_window=10000, max_output_tokens=1000),
                name="fake",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


@pytest.mark.asyncio
async def test_clean_turn_appends_boundary_marker(tmp_path: Any) -> None:
    prov = _install_provider("tests.integration._fake_boundary_provider")
    sess = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(prov, {}),
            scenario="local",
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        await sess.prompt("hi")
        mgr = sess.session_manager
        leaf = mgr.get_leaf_entry()
        assert leaf is not None
        # The turn ended cleanly → the active leaf is the boundary marker.
        assert leaf.type == ENTRY_TYPE_TURN_COMMITTED
        types_seen = [e.type for e in mgr.get_branch()]
        assert types_seen.count(ENTRY_TYPE_TURN_COMMITTED) == 1
        # The user + assistant messages sit on the branch before the marker.
        assert types_seen.count(ENTRY_TYPE_MESSAGE) >= 2

        # A second turn appends after the marker and re-commits a fresh one.
        await sess.prompt("again")
        assert mgr.get_leaf_entry().type == ENTRY_TYPE_TURN_COMMITTED
        assert [e.type for e in mgr.get_branch()].count(ENTRY_TYPE_TURN_COMMITTED) == 2
    finally:
        await sess.shutdown()
