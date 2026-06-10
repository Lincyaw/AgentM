"""Unit tests for the get_turn(idx) drill-down tool (REQ-028).

Each test exercises the _get_turn async closure extracted from the
``auditor_get_turn_tool`` module via the same capture pattern used in
test_two_phase_audit_integration.py for submit_tool.

Fail-stop position guarded: the get_turn tool is a boundary defence on
invalid idx values. If it raises instead of returning is_error=True, the
auditor child loop crashes, which silently terminates the whole audit
firing without a verdict entry — same class of failure as the V0 silent-
drop bug. These tests pin that boundary.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import ToolResult

# ---------------------------------------------------------------------------
# Helper: install the extension and extract the registered tool's fn
# ---------------------------------------------------------------------------


def _get_turn_fn(snapshot: list[dict[str, Any]]) -> Any:
    """Mint a ``get_turn`` tool over ``snapshot`` and return its ``fn``."""
    from llmharness.agents.auditor.tools import build_get_turn_tool

    return build_get_turn_tool(snapshot).fn


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGetTurnOutOfRange:
    """get_turn with out-of-range idx returns is_error=True, does not raise."""

    def test_negative_idx_returns_error(self) -> None:
        snapshot = [
            {"index": 0, "role": "user", "content": []},
            {"index": 1, "role": "assistant", "content": []},
        ]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": -1}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        body = "".join(b.text for b in result.content if hasattr(b, "text"))
        assert "out of range" in body.lower() or "range" in body.lower()


# ---------------------------------------------------------------------------
# Scenario G integration test (simplified compose assertion)
# ---------------------------------------------------------------------------
