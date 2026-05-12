"""MCP bridge — package-form contrib atom.

Mount with ``agentm --extension contrib.extensions.mcp_bridge``. The atom
opens an MCP client session per configured server, takes an install-time
snapshot of each server's tool list, and registers every remote tool as an
AgentM :class:`~agentm.core.abi.tool.Tool` via ``api.register_tool``.

See ``.claude/designs/mcp-integration.md`` for the full design rationale.
"""

from __future__ import annotations

from .bridge import install
from .manifest import MANIFEST

__all__ = ["MANIFEST", "install"]
