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
import json
from typing import Any

from agentm.core.abi import ToolResult

# ---------------------------------------------------------------------------
# Helper: install the extension and extract the registered tool's fn
# ---------------------------------------------------------------------------


def _get_turn_fn(snapshot: list[dict[str, Any]]) -> Any:
    """Install ``get_turn_tool`` with ``snapshot`` and return the tool's ``fn``."""
    from llmharness.audit.auditor.get_turn_tool import install

    captured: list[Any] = []

    class _CapturAPI:
        def register_tool(self, tool: Any) -> None:
            captured.append(tool)

    install(_CapturAPI(), {"trajectory_snapshot": snapshot})  # type: ignore[arg-type]
    assert len(captured) == 1, "get_turn_tool must register exactly one tool"
    return captured[0].fn


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGetTurnValidIdx:
    """get_turn with a valid idx returns the correct message dict."""

    def test_valid_idx_returns_correct_message(self) -> None:
        snapshot = [
            {"index": 0, "role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"index": 1, "role": "assistant", "content": [{"type": "text", "text": "world"}]},
            {"index": 2, "role": "user", "content": [{"type": "text", "text": "again"}]},
        ]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": 1}))

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert len(result.content) == 1
        parsed = json.loads(result.content[0].text)
        assert parsed == snapshot[1]

    def test_idx_zero_returns_first_message(self) -> None:
        snapshot = [
            {"index": 0, "role": "user", "content": [{"type": "text", "text": "first"}]},
        ]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": 0}))

        assert result.is_error is False
        parsed = json.loads(result.content[0].text)
        assert parsed == snapshot[0]


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

    def test_idx_beyond_end_returns_error(self) -> None:
        snapshot = [
            {"index": 0, "role": "user", "content": []},
            {"index": 1, "role": "assistant", "content": []},
        ]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": 5}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True

    def test_empty_snapshot_returns_error(self) -> None:
        fn = _get_turn_fn([])

        result: ToolResult = asyncio.run(fn({"idx": 0}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True


class TestGetTurnNonIntIdx:
    """get_turn with non-integer idx returns is_error=True, does not raise.

    The JSON Schema declares ``"type": "integer"`` but the tool handler must
    defend its own boundary — same pattern as the V2 verdict coercer.
    """

    def test_string_idx_returns_error(self) -> None:
        snapshot = [{"index": 0, "role": "user", "content": []}]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": "0"}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True

    def test_float_idx_returns_error(self) -> None:
        snapshot = [{"index": 0, "role": "user", "content": []}]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": 1.5}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True

    def test_bool_idx_returns_error(self) -> None:
        """bool is a subclass of int in Python — must be rejected as non-int."""
        snapshot = [{"index": 0, "role": "user", "content": []}]
        fn = _get_turn_fn(snapshot)

        # True would be index 1, which is out of range — but we also guard bool
        result: ToolResult = asyncio.run(fn({"idx": True}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True

    def test_none_idx_returns_error(self) -> None:
        snapshot = [{"index": 0, "role": "user", "content": []}]
        fn = _get_turn_fn(snapshot)

        result: ToolResult = asyncio.run(fn({"idx": None}))

        assert isinstance(result, ToolResult)
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Scenario G integration test (simplified compose assertion)
# ---------------------------------------------------------------------------


class TestScenarioG:
    """Scenario G: get_turn tool appears in compose_auditor_extensions when
    trajectory_snapshot is provided.

    This is the wire-up test: proves that the snapshot makes it from the
    adapter through compose_auditor_extensions into the registered tool.
    A full multi-tool-call flow via the stub provider would require
    significant harness changes for a single ordering constraint; we
    validate the integration via the simpler route (composition + tool
    invocation) which covers the same structural requirement.
    """

    def test_compose_with_snapshot_includes_get_turn_module(self) -> None:
        """compose_auditor_extensions(trajectory_snapshot=[...]) must
        include the get_turn module in the returned extension list."""
        from llmharness.audit.auditor.extensions import compose_auditor_extensions
        from llmharness.audit.auditor.get_turn_tool import GET_TURN_TOOL_NAME

        snapshot = [{"index": 0, "role": "user", "content": []}]
        exts = compose_auditor_extensions(
            cards_tools_config=None,
            observability_config=None,
            trajectory_snapshot=snapshot,
        )
        modules = {mod for mod, _cfg in exts}
        assert "llmharness.audit.auditor.get_turn_tool" in modules, (
            f"get_turn_tool module not in auditor extensions: {modules}"
        )
        # Config must carry the snapshot.
        for mod, cfg in exts:
            if mod == "llmharness.audit.auditor.get_turn_tool":
                assert cfg.get("trajectory_snapshot") == snapshot, (
                    f"trajectory_snapshot not forwarded in config: {cfg}"
                )

        # Confirm the tool name constant is exported.
        assert GET_TURN_TOOL_NAME == "get_turn"

    def test_compose_without_snapshot_omits_get_turn_module(self) -> None:
        """compose_auditor_extensions(trajectory_snapshot=None) must NOT
        include the get_turn module — auditor prompt's 'may not be
        available' caveat is correct by default."""
        from llmharness.audit.auditor.extensions import compose_auditor_extensions

        exts = compose_auditor_extensions(
            cards_tools_config=None,
            observability_config=None,
        )
        modules = {mod for mod, _cfg in exts}
        assert "llmharness.audit.auditor.get_turn_tool" not in modules, (
            "get_turn_tool must not be registered when trajectory_snapshot=None"
        )

    def test_get_turn_tool_returns_correct_snapshot_entry_via_compose(self) -> None:
        """Full wire-up: compose → install → invoke get_turn → verify payload.

        Uses the same _CapturAPI pattern to simulate what the AgentM kernel
        does when it loads an extension list.
        """
        from llmharness.audit.auditor.extensions import compose_auditor_extensions

        snapshot = [
            {"index": 0, "role": "user", "content": [{"type": "text", "text": "turn-zero"}]},
            {"index": 1, "role": "assistant", "content": [{"type": "text", "text": "turn-one"}]},
            {"index": 2, "role": "user", "content": [{"type": "text", "text": "turn-two"}]},
        ]
        exts = compose_auditor_extensions(
            cards_tools_config=None,
            observability_config=None,
            trajectory_snapshot=snapshot,
        )

        # Find and install the get_turn_tool module.
        import importlib

        captured: list[Any] = []

        class _CapturAPI:
            def register_tool(self, tool: Any) -> None:
                captured.append(tool)

        for mod_path, cfg in exts:
            if mod_path == "llmharness.audit.auditor.get_turn_tool":
                mod = importlib.import_module(mod_path)
                mod.install(_CapturAPI(), cfg)  # type: ignore[attr-defined]
                break

        assert len(captured) == 1
        fn = captured[0].fn

        # Call get_turn(2) — must return snapshot[2].
        result: ToolResult = asyncio.run(fn({"idx": 2}))
        assert result.is_error is False
        parsed = json.loads(result.content[0].text)
        assert parsed == snapshot[2], f"Expected snapshot[2], got {parsed}"
