"""Manifest for the ``contrib.extensions.mcp_bridge`` atom.

Kept separate from ``bridge.py`` so importers reading just the manifest
(catalog tooling, scenario validators) don't pay the cost of importing
the MCP SDK and its async stack.

See ``.claude/designs/mcp-integration.md`` §4.1 for the config schema
rationale.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentm.extensions import ExtensionManifest


class MCPServerSpec(BaseModel):
    model_config = {"extra": "allow"}

    name: str
    transport: Literal["stdio", "http"]
    command: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None


class MCPBridgeConfig(BaseModel):
    servers: list[MCPServerSpec] = Field(min_length=1)
    naming: str | None = None
    timeout_seconds: float | None = None


MANIFEST = ExtensionManifest(
    name="mcp_bridge",
    description=(
        "Bridge to Model Context Protocol servers. Connects to each "
        "configured server at install time, snapshots its tools/list "
        "response, and registers every remote tool as an AgentM Tool. "
        "stdio + http transports only; prompts/resources/sampling "
        "deferred to follow-up atoms."
    ),
    registers=(
        # Concrete tool names are discovered at install time from the
        # remote server, so this manifest only declares the *kind* of
        # registration (tool catalog mutation). The actual names land
        # under the ``mcp__<server>__<tool>`` namespace.
        "mutates:tool_catalog",
    ),
    config_schema=MCPBridgeConfig,
    requires=(),  # leaf atom: consumes nothing from other atoms
    api_version=1,
    tier=1,
)


__all__ = ["MANIFEST", "MCPBridgeConfig"]
