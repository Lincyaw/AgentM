"""End-to-end smoke test for ``contrib.extensions.mcp_bridge``.

Spawns the real ``mcp-server-fetch`` MCP server via ``uvx``, drives the
full ``install`` → ``tools/list`` → ``tools/call`` → close cycle, and
prints what it saw. No LLM provider required.

Run with::

    uv run --package agentm-mcp-bridge python contrib/scenarios/mcp_demo/smoke.py

This script is intentionally NOT a pytest test — it depends on network
access (``https://example.com``) and on ``uvx`` being able to fetch the
``mcp-server-fetch`` package on first run, neither of which belongs in
the unit-test gate. The pytest test under
``contrib/extensions/mcp_bridge/tests/test_e2e.py`` covers the same path
with an in-process stub.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable so ``contrib.extensions.mcp_bridge``
# resolves when the script is launched directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from agentm.core.abi import EventBus, Tool  # noqa: E402
from agentm.core.runtime.extension import (  # noqa: E402
    _ExtensionAPIImpl,
    build_extension_api_scope,
)
from agentm.core.runtime.session_inbox import SessionInbox  # noqa: E402

from contrib.extensions.mcp_bridge import install  # noqa: E402
from contrib.extensions.mcp_bridge.client import MCPSessionManager  # noqa: E402

class _SessionView:
    """Minimum surface ``_ExtensionAPIImpl`` needs (matches test_e2e.py)."""

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
        self,
        type: str,
        payload: Any,
        parent_id: str | None = None,
    ) -> str:
        del type, payload, parent_id
        return "entry"

async def main() -> int:
    tools: list[Tool] = []
    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd="/tmp",
        session_id="mcp-smoke",
        session=_SessionView(),
        tools=tools,
        commands={},
        providers={},
        renderers={},
        inbox=SessionInbox(),
        model_getter=lambda: None,
        provider_getter=lambda: None,
    )
    api = _ExtensionAPIImpl(scope)

    config = {
        "servers": [
            {
                "name": "fetch",
                "transport": "stdio",
                "command": ["uvx", "mcp-server-fetch"],
            }
        ],
        "timeout_seconds": 60,
    }

    print(">>> install(mcp_bridge) against uvx mcp-server-fetch...")
    await install(api, config)
    print(f">>> discovered {len(tools)} tool(s):")
    for t in tools:
        desc = getattr(t, "description", "")
        params = getattr(t, "parameters", {}) or {}
        print(f"    - {t.name}: {desc[:80]}")
        print(
            "      params keys: "
            f"{sorted((params.get('properties') or {}).keys())}"
        )

    print(">>> invoking mcp__fetch__fetch on https://example.com ...")
    fetch_tool = next(t for t in tools if t.name == "mcp__fetch__fetch")
    raw_result: Any = await fetch_tool.execute(
        {"url": "https://example.com", "max_length": 200}
    )
    is_error = bool(getattr(raw_result, "is_error", False))
    blocks: list[Any] = list(getattr(raw_result, "content", []) or [])
    print(f">>> is_error={is_error}, content blocks={len(blocks)}")
    if blocks:
        first = blocks[0]
        text = getattr(first, "text", repr(first))
        print(f"    first block (truncated 300): {text[:300]!r}")

    manager = api.get_service("mcp")
    assert isinstance(manager, MCPSessionManager)
    await manager.aclose()
    print(">>> closed cleanly.")

    if is_error or not blocks:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
