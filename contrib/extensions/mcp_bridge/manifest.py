"""Manifest for the ``contrib.extensions.mcp_bridge`` atom.

Kept separate from ``bridge.py`` so importers reading just the manifest
(catalog tooling, scenario validators) don't pay the cost of importing
the MCP SDK and its async stack.

See ``.claude/designs/mcp-integration.md`` §4.1 for the config schema
rationale.
"""

from __future__ import annotations

from agentm.extensions import ExtensionManifest


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
    config_schema={
        "type": "object",
        "properties": {
            "servers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "transport": {
                            "type": "string",
                            "enum": ["stdio", "http"],
                        },
                        # stdio-only
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "env": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        # http-only
                        "url": {"type": "string"},
                        "headers": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["name", "transport"],
                    "additionalProperties": True,
                },
            },
            "naming": {
                "type": "string",
                "description": (
                    "Template used to render Tool.name. Substitutions: "
                    "{server}, {tool}. Default: 'mcp__{server}__{tool}'."
                ),
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0,
            },
        },
        "required": ["servers"],
        "additionalProperties": False,
    },
    requires=(),  # leaf atom: consumes nothing from other atoms
    api_version=1,
    tier=1,
)


__all__ = ["MANIFEST"]
