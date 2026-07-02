"""E2E: stub MCP server -> install snapshots tools -> tool call returns result.

This is the *only* test for the mcp_bridge atom, per CLAUDE.md "Testing
philosophy" — there is no per-transport unit test and no schema-validation
unit test. The fail-stop position is the snapshot → catalog → call path; if
that path breaks, the bridge has failed its value proposition.

The stub session implements the tiny subset of :class:`mcp.ClientSession`
the bridge actually calls (``initialize`` / ``list_tools`` / ``call_tool``)
plus the response shapes the SDK normally produces. Going through a real
transport here would test the SDK, not us.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.runtime.extension import (
    _ExtensionAPIImpl,
    build_extension_api_scope,
)
from agentm.core.runtime.session_inbox import SessionInbox

from contrib.extensions.mcp_bridge import MANIFEST, install
from contrib.extensions.mcp_bridge.client import (
    MCPSessionManager,
    set_test_session_factory,
)


# ---- Stub MCP server ------------------------------------------------------


@dataclass(slots=True)
class _StubTool:
    name: str
    description: str
    inputSchema: dict[str, Any]


@dataclass(slots=True)
class _StubToolsList:
    tools: list[_StubTool]


@dataclass(slots=True)
class _StubTextBlock:
    type: str
    text: str


@dataclass(slots=True)
class _StubCallResult:
    content: list[_StubTextBlock]
    isError: bool = False


class _StubSession:
    """In-memory ``ClientSession`` stand-in.

    Records the calls it received so the test can assert flow without
    poking at the bridge's privates.
    """

    def __init__(self, tools: list[_StubTool]) -> None:
        self._tools = tools
        self.initialized = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> _StubToolsList:
        return _StubToolsList(tools=list(self._tools))

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> _StubCallResult:
        self.calls.append((name, dict(arguments or {})))
        return _StubCallResult(
            content=[_StubTextBlock(type="text", text=f"echo:{name}:{arguments}")]
        )


# ---- ExtensionAPI scaffolding --------------------------------------------


class _SessionView:
    def get_messages(self) -> list[Any]:
        return []

    def get_branch(self) -> list[Any]:
        return []

    def get_leaf_id(self) -> str | None:
        return None

    def get_entry(self, entry_id: str) -> Any | None:
        del entry_id
        return None

    def get_loop_config(self) -> Any:
        return None

    def append_entry(
        self, type: str, payload: Any, parent_id: str | None = None
    ) -> str:
        del type, payload, parent_id
        return "entry"


def _make_api(tmp_path: Path, tools: list[Any]) -> _ExtensionAPIImpl:
    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id="session",
        session=_SessionView(),
        tools=tools,
        commands={},
        providers={},
        renderers={},
        inbox=SessionInbox(),
        model_getter=lambda: None,
        provider_getter=lambda: None,
    )
    return _ExtensionAPIImpl(scope)


# ---- The one test --------------------------------------------------------


@pytest.mark.asyncio
async def test_install_snapshot_and_call_roundtrip(tmp_path: Path) -> None:
    """install → tool registered → execute() returns a real ToolResult."""

    assert MANIFEST.name == "mcp_bridge"

    stub_session = _StubSession(
        tools=[
            _StubTool(
                name="echo",
                description="Echo back the arguments verbatim.",
                inputSchema={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            )
        ]
    )

    async def _factory(spec: Any, _stack: Any) -> _StubSession:
        # Single-server test; the manager only ever calls us for the one
        # spec we configured below.
        assert spec.name == "fs"
        return stub_session

    tools_catalog: list[Any] = []
    api = _make_api(tmp_path, tools_catalog)

    config = {
        "servers": [
            {
                "name": "fs",
                "transport": "stdio",
                "command": ["unused", "--because-of-factory"],
            }
        ],
    }

    set_test_session_factory(_factory)
    await install(api, config)

    # --- 1. server connect + snapshot landed -----------------------------
    assert stub_session.initialized is True
    assert len(tools_catalog) == 1
    bridged = tools_catalog[0]
    assert bridged.name == "mcp__fs__echo"
    assert bridged.description == "Echo back the arguments verbatim."
    assert bridged.parameters["properties"]["msg"]["type"] == "string"

    # --- 2. manager published as a service -------------------------------
    manager = api.get_service("mcp")
    assert isinstance(manager, MCPSessionManager)
    assert manager.get_session("fs") is stub_session

    # --- 3. calling the bridged tool routes through MCP ------------------
    result = await bridged.execute({"msg": "hi"})
    assert stub_session.calls == [("echo", {"msg": "hi"})]
    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "echo:echo:{'msg': 'hi'}" in result.content[0].text
    assert result.extras == {"mcp": {"server": "fs"}}

    # --- 4. session-shutdown event tears the manager down ---------------
    await api.events.emit(
        SessionShutdownEvent.CHANNEL,
        SessionShutdownEvent(cwd=str(tmp_path)),
    )
    # ``aclose`` is idempotent; calling it again must not raise.
    await manager.aclose()

    # Quiet asyncio's "task was never awaited" linter in case of partial fail.
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_test_factory_survives_malformed_spec(tmp_path: Path) -> None:
    """Regression: a parse failure must not drain the test seam.

    Previously ``consume_test_session_factory()`` ran before
    ``parse_server_spec``, so a malformed spec ate the factory and the
    next ``install()`` (with a valid spec) had nothing to fall back on
    — confusing as hell to debug from a test failure. Now the factory
    is consumed only after every spec parses cleanly.
    """

    stub = _StubSession(tools=[])

    async def _factory(_spec: Any, _stack: Any) -> _StubSession:
        return stub

    set_test_session_factory(_factory)
    api = _make_api(tmp_path, [])

    # Malformed: ``transport`` missing → parse_server_spec raises.
    bad_config = {"servers": [{"name": "fs"}]}
    with pytest.raises(Exception):
        await install(api, bad_config)

    # Factory must still be available for the next attempt.
    good_config = {
        "servers": [
            {"name": "fs", "transport": "stdio", "command": ["unused"]}
        ],
    }
    await install(api, good_config)
    # If the factory had been drained, ``connect_all`` would have tried
    # to spawn a real subprocess for ``unused`` and failed; reaching here
    # with the stub initialised proves the factory survived.
    assert stub.initialized is True
