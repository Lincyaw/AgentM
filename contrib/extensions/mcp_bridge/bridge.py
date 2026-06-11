"""``install(api, config)`` body for the MCP bridge.

See ``.claude/designs/mcp-integration.md`` §4.2 for the lifecycle and §8
for the failure-mode contract. The short version:

1. Connect every configured server.
2. ``tools/list`` once per server (install-time snapshot).
3. Register each remote tool as an :class:`MCPTool`.
4. Publish the manager via ``api.set_service("mcp", manager)``.
5. Subscribe ``SessionShutdownEvent`` to tear transports down.

Any failure during steps 1–3 raises; partial tool sets are worse than no
session.
"""

from __future__ import annotations


from agentm.core.abi import ExtensionAPI, SessionShutdownEvent

from .client import (
    MCPSessionManager,
    ServerSpec,
    consume_test_session_factory,
    parse_server_spec,
)
from .manifest import MCPBridgeConfig
from .tool import MCPTool

_DEFAULT_NAMING = "mcp__{server}__{tool}"

async def install(api: ExtensionAPI, config: MCPBridgeConfig) -> None:
    """Connect, snapshot, register. Async per design §4.2."""

    # Parse first; consume the test seam only on the path where it is
    # actually used. Previously ``consume_test_session_factory`` ran
    # before ``parse_server_spec`` — a malformed spec would silently
    # drain the factory and the next ``install()`` would get nothing.
    specs: list[ServerSpec] = [
        parse_server_spec(s.model_dump(exclude_none=True)) for s in config.servers
    ]

    naming_template = config.naming or _DEFAULT_NAMING
    timeout: float | None
    if config.timeout_seconds is None:
        timeout = 30.0
    else:
        timeout = config.timeout_seconds

    # Module-level test hook (single-shot, consumed here). Production
    # callers never set this; the public config schema rejects it.
    # Consumed *after* parse_server_spec so a parse failure leaves the
    # factory in place for the next install() attempt.
    session_factory = consume_test_session_factory()

    manager = MCPSessionManager()
    await manager.connect_all(specs, session_factory=session_factory)

    try:
        for spec in specs:
            await _snapshot_and_register(
                api=api,
                manager=manager,
                server_name=spec.name,
                naming_template=naming_template,
                timeout=timeout,
            )
    except BaseException:
        await manager.aclose()
        raise

    api.set_service("mcp", manager)

    async def _on_shutdown(_event: SessionShutdownEvent) -> None:
        await manager.aclose()

    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)

async def _snapshot_and_register(
    *,
    api: ExtensionAPI,
    manager: MCPSessionManager,
    server_name: str,
    naming_template: str,
    timeout: float | None,
) -> None:
    """Call ``tools/list`` once and register every returned tool.

    Raises on transport errors and on tools whose schema cannot be coerced
    into a JSON-Schema-shaped dict — both are install-time failures per
    design §8.
    """

    session = manager.get_session(server_name)
    raw_list = await session.list_tools()
    tools = getattr(raw_list, "tools", None)
    if tools is None:
        raise RuntimeError(
            f"mcp_bridge: server {server_name!r} tools/list response "
            "missing 'tools' field"
        )

    names: list[str] = []
    for remote in tools:
        remote_name = getattr(remote, "name", None)
        if not isinstance(remote_name, str) or not remote_name:
            raise ValueError(
                f"mcp_bridge: server {server_name!r} returned tool with "
                "no usable name"
            )
        description = str(getattr(remote, "description", "") or "")
        schema = getattr(remote, "inputSchema", None)
        if schema is None:
            schema = {"type": "object", "properties": {}, "additionalProperties": True}
        if not isinstance(schema, dict):
            raise ValueError(
                f"mcp_bridge: server {server_name!r} tool {remote_name!r} "
                f"has non-object input schema: {type(schema).__name__}"
            )

        bridged_name = naming_template.format(server=server_name, tool=remote_name)
        tool = MCPTool(
            name=bridged_name,
            description=description,
            parameters=schema,
            _server=server_name,
            _remote_name=remote_name,
            _session=session,
            _timeout=timeout,
            _extras={"mcp": {"server": server_name}},
        )
        api.register_tool(tool)
        names.append(bridged_name)

    # Lifecycle observability — emit via the public event bus the same way
    # other atoms do (e.g. ``llm_compaction``). Channels are free-form
    # strings; payload is a plain dict so observers don't need to import
    # this atom's module.
    await api.events.emit(  # type: ignore[attr-defined]
        "mcp.server.connected",
        {"server": server_name, "tool_count": len(names)},
    )
    await api.events.emit(  # type: ignore[attr-defined]
        "mcp.tools.snapshotted",
        {"server": server_name, "tools": tuple(names)},
    )

__all__ = ["install"]
