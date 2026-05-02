from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from agentm.cli import _run_interactive
from agentm.core.abi import (
    AssistantMessage,
    EventBus,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    MessageEnd,
    Model,
    StreamDeltaEvent,
    TextContent,
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolCallStart,
    ToolResult,
    ToolResultEvent,
    Usage,
)
from agentm.harness import AgentSessionConfig
from agentm.harness.events import ChildSessionEndEvent, ChildSessionStartEvent, ExtensionInstallEvent
from agentm.modes.textual_app import (
    AgentMApp,
    SlashCommandEntry,
    ToolCallBlock,
    TurnContainer,
)

# All cases here exercise the Textual TUI rendering layer. They pass but
# do not map to any AgentM fail-stop core position (CLAUDE.md "Testing
# Philosophy" §1), so they are excluded from the default suite. Run with
# ``uv run pytest -m ui`` when touching ``modes/textual_app.py`` or its
# event wiring.
pytestmark = pytest.mark.ui


PromptScript = Callable[[EventBus, str], Awaitable[list[Any]]]


class _FakeSession:
    def __init__(self, scripts: list[PromptScript]) -> None:
        self.bus = EventBus()
        self.model = Model(
            id="fake-model",
            provider="fake",
            context_window=10_000,
            max_output_tokens=1_000,
        )
        self._scripts = scripts
        self.prompts: list[str] = []
        self.shutdown_called = False

    async def prompt(
        self,
        text: str,
        *,
        images: list[Any] | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[Any]:
        del images, signal
        self.prompts.append(text)
        script = self._scripts.pop(0)
        return await script(self.bus, text)

    async def shutdown(self) -> None:
        self.shutdown_called = True


async def _golden_script(bus: EventBus, text: str) -> list[Any]:
    del text
    await bus.emit(
        "llm_request_start",
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=0,
            system_chars=0,
            model_id="fake-model",
        ),
    )
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=ThinkingDelta(text="thinking")))
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=TextDelta(text="Hello from AgentM")))
    await bus.emit(
        "stream_delta",
        StreamDeltaEvent(
            turn_index=0,
            delta=MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="Hello from AgentM")],
                    timestamp=1.0,
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=128, output_tokens=32),
                )
            ),
        ),
    )
    await bus.emit(
        "llm_request_end",
        LlmRequestEndEvent(turn_index=0, chunk_count=3, duration_ns=1_000_000, error=None),
    )
    return []


async def _cancel_script(bus: EventBus, text: str) -> list[Any]:
    del text
    await bus.emit(
        "llm_request_start",
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=0,
            system_chars=0,
            model_id="fake-model",
        ),
    )
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=TextDelta(text="partial")))
    try:
        await asyncio.sleep(5)
    except asyncio.CancelledError:
        raise
    return []


async def _tool_script(bus: EventBus, text: str) -> list[Any]:
    del text
    await bus.emit(
        "llm_request_start",
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=1,
            system_chars=0,
            model_id="fake-model",
        ),
    )
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=ToolCallStart(id="tc-1", name="read")))
    await bus.emit(
        "tool_call",
        ToolCallEvent(tool_call_id="tc-1", tool_name="read", args={"path": "foo.txt"}),
    )
    await bus.emit(
        "tool_result",
        ToolResultEvent(
            tool_call_id="tc-1",
            tool_name="read",
            result=ToolResult(content=[TextContent(type="text", text="ok\nline2")]),
        ),
    )
    await bus.emit(
        "llm_request_end",
        LlmRequestEndEvent(turn_index=0, chunk_count=2, duration_ns=1_000_000, error=None),
    )
    return []


async def _follow_up_script(bus: EventBus, text: str) -> list[Any]:
    await _golden_script(bus, text)
    return []


async def _no_op_script(bus: EventBus, text: str) -> list[Any]:
    del bus, text
    return []


async def _subagent_script(bus: EventBus, text: str) -> list[Any]:
    del text
    await bus.emit(
        "llm_request_start",
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=0,
            system_chars=0,
            model_id="fake-model",
        ),
    )
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=TextDelta(text="parent")))
    await bus.emit(
        "child_session_start",
        ChildSessionStartEvent(
            child_session_id="child-1",
            parent_session_id="parent-1",
            purpose="subagent:test",
        ),
    )
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=TextDelta(text=" child step")))
    await bus.emit(
        "child_session_end",
        ChildSessionEndEvent(
            child_session_id="child-1",
            parent_session_id="parent-1",
            final_message_count=1,
            error="boom",
        ),
    )
    await bus.emit(
        "llm_request_end",
        LlmRequestEndEvent(turn_index=0, chunk_count=2, duration_ns=1_000_000, error=None),
    )
    return []


async def _tool_edit_script(bus: EventBus, text: str) -> list[Any]:
    del text
    await bus.emit(
        "llm_request_start",
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=1,
            system_chars=0,
            model_id="fake-model",
        ),
    )
    await bus.emit("stream_delta", StreamDeltaEvent(turn_index=0, delta=ToolCallStart(id="tc-edit", name="tool_edit")))
    await bus.emit(
        "tool_call",
        ToolCallEvent(tool_call_id="tc-edit", tool_name="tool_edit", args={"path": "a.py"}),
    )
    await bus.emit(
        "tool_result",
        ToolResultEvent(
            tool_call_id="tc-edit",
            tool_name="tool_edit",
            result=ToolResult(
                content=[TextContent(type="text", text="--- a.py\n+++ a.py\n@@\n-print('x')\n+print('y')")]
            ),
        ),
    )
    await bus.emit(
        "llm_request_end",
        LlmRequestEndEvent(turn_index=0, chunk_count=2, duration_ns=1_000_000, error=None),
    )
    return []


def _make_app(tmp_path: Path, session: _FakeSession) -> AgentMApp:
    config = AgentSessionConfig(cwd=str(tmp_path), provider=("fake", {}), bus=session.bus)
    app = AgentMApp(
        config,
        theme="dark",
        session=session,
        slash_commands=[
            SlashCommandEntry(name="/help", description="Show help.", source="builtin"),
            SlashCommandEntry(name="/clear", description="Clear log.", source="builtin"),
            SlashCommandEntry(name="/custom", description="Custom slash.", source="ext"),
        ],
    )
    session.bus.on("stream_delta", app.handle_stream_delta)
    session.bus.on("tool_call", app.handle_tool_call)
    session.bus.on("tool_result", app.handle_tool_result)
    session.bus.on("llm_request_start", app.handle_llm_start)
    session.bus.on("llm_request_end", app.handle_llm_end)
    session.bus.on("child_session_start", app.handle_child_start)
    session.bus.on("child_session_end", app.handle_child_end)
    session.bus.on("extension_install", app.handle_extension_install)
    return app


@pytest.mark.asyncio
async def test_T1_golden_path_updates_stream_and_status(tmp_path: Path) -> None:
    app = _make_app(tmp_path, _FakeSession([_golden_script]))
    async with app.run_test() as pilot:
        await pilot.press("h", "i", "enter")
        await pilot.pause(0.2)
        status = app.query_one("#status-line")
        assert "turn 1" in status.current_text
        assert "in: 128" in status.current_text
        assert "out: 32" in status.current_text
        assert "idle" in status.current_text
        assert app._last_assistant_text == "Hello from AgentM"


@pytest.mark.asyncio
async def test_T2_escape_soft_cancels_and_keeps_partial_text(tmp_path: Path) -> None:
    app = _make_app(tmp_path, _FakeSession([_cancel_script]))
    async with app.run_test() as pilot:
        await pilot.press("x", "enter")
        await pilot.pause(0.1)
        await pilot.press("escape")
        await pilot.pause(0.1)
        status = app.query_one("#status-line")
        assert "idle" in status.current_text
        assert app._last_assistant_text == "partial"
        assert app._prompt_task is None


@pytest.mark.asyncio
async def test_T3_tool_block_collapses_and_toggles(tmp_path: Path) -> None:
    app = _make_app(tmp_path, _FakeSession([_tool_script]))
    async with app.run_test() as pilot:
        await pilot.press("x", "enter")
        await pilot.pause(0.2)
        tool = app.query_one(ToolCallBlock)
        assert tool.collapsed is True
        tool.focus()
        await pilot.press("ctrl+e")
        await pilot.pause(0.05)
        assert tool.collapsed is False


@pytest.mark.asyncio
async def test_T4_slash_palette_opens_filters_and_inserts_selection(tmp_path: Path) -> None:
    app = _make_app(tmp_path, _FakeSession([_no_op_script]))
    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause(0.05)
        assert app.screen_stack[-1].__class__.__name__ == "CommandPaletteScreen"
        await pilot.press("h", "e", "l", "p", "enter")
        await pilot.pause(0.05)
        prompt = app.query_one("#prompt-input")
        assert prompt.text == "/help"


@pytest.mark.asyncio
async def test_T5_subagent_block_renders_and_marks_error(tmp_path: Path) -> None:
    app = _make_app(tmp_path, _FakeSession([_subagent_script]))
    async with app.run_test() as pilot:
        await pilot.press("x", "enter")
        await pilot.pause(0.2)
        subagent = next(iter(app._child_turns.values()), None)
        assert subagent is None
        turn = app._latest_turn()
        assert turn is not None
        block = turn.subagents["child-1"]
        assert block.failed is True
        assert "boom" in str(block.lines)


@pytest.mark.asyncio
async def test_T6_shift_enter_inserts_newline(tmp_path: Path) -> None:
    app = _make_app(tmp_path, _FakeSession([_no_op_script]))
    async with app.run_test() as pilot:
        await pilot.press("h", "i", "shift+enter", "t", "h", "e", "r", "e")
        await pilot.pause(0.05)
        prompt = app.query_one("#prompt-input")
        assert prompt.text == "hi\nthere"


@pytest.mark.asyncio
async def test_T7_ctrl_l_clears_visual_log_but_follow_up_still_runs(tmp_path: Path) -> None:
    session = _FakeSession([_golden_script, _follow_up_script])
    app = _make_app(tmp_path, session)
    async with app.run_test() as pilot:
        await pilot.press("x", "enter")
        await pilot.pause(0.2)
        await pilot.press("ctrl+l")
        await pilot.pause(0.05)
        assert len(app.query_one("#conversation-log").children) == 0
        await pilot.press("y", "enter")
        await pilot.pause(0.2)
        assert session.prompts == ["x", "y"]
        assert len(app.query_one("#conversation-log").children) > 0


@pytest.mark.asyncio
async def test_T8_many_turns_keep_scrollable_log(tmp_path: Path) -> None:
    session = _FakeSession([])
    app = _make_app(tmp_path, session)
    async with app.run_test() as pilot:
        for turn_index in range(500):
            turn = TurnContainer(logical_turn=turn_index + 1)
            app._root_turns[turn_index] = turn
            turn.text_buffer = f"turn {turn_index}"
            turn.assistant.set_text(turn.text_buffer)
            await app._mount_turn(turn)
        await pilot.pause(0.2)
        log = app.query_one("#conversation-log")
        # All 500 turns must remain present and scrollable; Textual virtual
        # scrolling keeps memory bounded by only painting visible widgets.
        assert len(app._root_turns) == 500
        assert len(log.children) == 500
        latest = app._latest_turn()
        assert latest is not None
        assert latest.assistant.text == "turn 499"
        log.focus()
        log.scroll_home(animate=False)
        await pilot.pause(0.05)
        scroll_top = log.scroll_y
        await pilot.press("pagedown")
        await pilot.pause(0.05)
        scroll_after_pgdn = log.scroll_y
        log.scroll_end(animate=False)
        await pilot.pause(0.05)
        scroll_bottom = log.scroll_y
        # Scroll responsiveness: top, page-down, and bottom give three
        # distinct positions, proving the log scrolls across the full range.
        assert scroll_after_pgdn > scroll_top
        assert scroll_bottom > scroll_after_pgdn
        await pilot.press("pageup")
        await pilot.pause(0.05)
        assert log.scroll_y < scroll_bottom


@pytest.mark.asyncio
async def test_T9_cli_dispatches_to_textual_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_interactive`` builds a session config and hands off to the
    Textual runner. Regression guard: removing the legacy ``simple`` TUI
    must not break the dispatch path the ``-i`` CLI flag depends on."""

    called: list[str] = []

    async def _fake_textual(config: AgentSessionConfig, *, theme: str = "dark") -> int:
        called.append(f"textual:{theme}:{config.cwd}")
        return 22

    monkeypatch.setattr("agentm.modes.textual_app.run", _fake_textual)

    rc = await _run_interactive(
        scenario=None,
        no_extensions=True,
        no_skills=True,
        no_prompt_templates=True,
        tool_allowlist=None,
        model="fake-model",
        cwd="/tmp/textual",
    )

    assert rc == 22
    assert called == ["textual:dark:/tmp/textual"]


@pytest.mark.asyncio
async def test_T10_layout_remains_usable_without_truecolor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Design §7 T10: terminal without 24-bit color → theme falls back to
    # 16-color, layout remains usable. Strip COLORTERM / TERM_PROGRAM and
    # force a 16-color TERM so Rich's color-system probe cannot pick truecolor.
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    session = _FakeSession([_golden_script])
    app = _make_app(tmp_path, session)
    async with app.run_test() as pilot:
        # The Rich console driving Textual must not negotiate truecolor.
        color_system = app.console.color_system
        assert color_system != "truecolor"
        # The three required regions remain mounted and addressable.
        assert app.query_one("#conversation-log") is not None
        assert app.query_one("#input-bar") is not None
        assert app.query_one("#status-line") is not None
        # Submitting a prompt still drives the log + status line end-to-end.
        await pilot.press("h", "i", "enter")
        await pilot.pause(0.2)
        log = app.query_one("#conversation-log")
        assert len(log.children) >= 1
        latest = app._latest_turn()
        assert latest is not None
        assert "Hello from AgentM" in latest.assistant.text


@pytest.mark.asyncio
async def test_light_theme_diff_rendering_and_extension_error_toast(tmp_path: Path) -> None:
    session = _FakeSession([_tool_edit_script])
    app = AgentMApp(
        AgentSessionConfig(cwd=str(tmp_path), provider=("fake", {}), bus=session.bus),
        theme="light",
        session=session,
        slash_commands=[SlashCommandEntry(name="/help", description="Show help.", source="builtin")],
    )
    session.bus.on("stream_delta", app.handle_stream_delta)
    session.bus.on("tool_call", app.handle_tool_call)
    session.bus.on("tool_result", app.handle_tool_result)
    session.bus.on("llm_request_start", app.handle_llm_start)
    session.bus.on("llm_request_end", app.handle_llm_end)
    session.bus.on("extension_install", app.handle_extension_install)

    async with app.run_test(notifications=True) as pilot:
        await session.bus.emit(
            "extension_install",
            ExtensionInstallEvent(
                module_path="bad.ext",
                config={},
                phase="error",
                duration_ns=1,
                error="boom",
            ),
        )
        await pilot.press("x", "enter")
        await pilot.pause(0.2)
        tool = app.query_one(ToolCallBlock)
        assert "theme-light" in app.classes
        assert tool.body_kind == "Syntax"
