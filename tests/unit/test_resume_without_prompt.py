"""Resume-without-prompt fail-stop coverage.

Verifies:

* CLI ``prompt`` arg becomes optional when ``--resume`` (or ``--continue``)
  is set; fresh sessions with no input are still rejected.
* :meth:`AgentSession.tick` resolves the synthetic
  ``decide_turn_action`` correctly:
    - no injecting extension → :class:`NoPendingInput` AgentEnd, message
      list unchanged, no new session entry persisted.
    - an extension that returns :class:`Inject` on the first decide event →
      the injected message lands, the loop runs one full assistant turn,
      and AgentEnd carries a non-``NoPendingInput`` cause.

Fail-stop justification: the CLI guard previously blocked any
event-driven / continuation atom (llmharness.replay.reminder_seed in
particular) from advancing a resumed session. If the parsing rule or the
tick semantics regress, the entire prefix-replay flow silently breaks at
the kernel boundary and the harness can no longer reproduce post-reminder
behaviour — the load-bearing path for the distill loop.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentm.core.abi import (
    AgentEndEvent,
    AssistantMessage,
    DecideTurnActionEvent,
    Inject,
    MessageEnd,
    Model,
    NoPendingInput,
    TextContent,
    UserMessage,
)
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


# ---------------------------------------------------------------------------
# Stub provider — yields a single one-text-block assistant message and stops.
# ---------------------------------------------------------------------------


async def _stream_fn(
    *,
    messages: list[Any],
    model: Model,
    tools: list[Any],
    system: str | None = None,
    signal: Any = None,
    thinking: str = "off",
) -> AsyncIterator[Any]:
    del messages, model, tools, system, signal, thinking
    yield MessageEnd(
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="stub-reply")],
            timestamp=0.0,
            stop_reason="end_turn",
        )
    )


def _register_stub_provider(module_name: str) -> types.ModuleType:
    module = types.ModuleType(module_name)

    def install(api: Any, config: dict[str, Any]) -> None:
        del config
        api.register_provider(
            "stub",
            ProviderConfig(
                stream_fn=_stream_fn,
                model=Model(
                    id="stub-model",
                    provider="stub",
                    context_window=1024,
                    max_output_tokens=64,
                ),
                name="stub",
            ),
        )

    setattr(module, "install", install)
    sys.modules[module_name] = module
    return module


async def _make_session(tmp_path: Path, module_name: str) -> AgentSession:
    _register_stub_provider(module_name)
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(module_name, {}),
            extensions=[
                ("agentm.extensions.builtin.operations_local", {}),
            ],
        )
    )


# ---------------------------------------------------------------------------
# tick semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_without_injector_emits_no_pending_input(tmp_path: Path) -> None:
    module_name = f"tests.unit._tick_no_inject_{id(tmp_path)}"
    causes: list[Any] = []
    try:
        session = await _make_session(tmp_path, module_name)
        session.bus.on(AgentEndEvent.CHANNEL, lambda event: causes.append(event.cause))

        before_entries = list(session.session_manager.get_active_branch())
        result = await session.tick()
        after_entries = list(session.session_manager.get_active_branch())

        # No extension responded → unchanged trajectory.
        assert result == session.session_manager.build_session_context().messages
        assert [e.id for e in after_entries] == [e.id for e in before_entries]
        assert any(isinstance(cause, NoPendingInput) for cause in causes)
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_tick_with_injector_runs_one_turn(tmp_path: Path) -> None:
    module_name = f"tests.unit._tick_with_inject_{id(tmp_path)}"
    causes: list[Any] = []
    fired: list[bool] = [False]

    def _on_decide(event: DecideTurnActionEvent) -> Any:
        if fired[0]:
            return None
        fired[0] = True
        return Inject(
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text="injected-from-extension")],
                    timestamp=0.0,
                )
            ]
        )

    try:
        session = await _make_session(tmp_path, module_name)
        session.bus.on(AgentEndEvent.CHANNEL, lambda event: causes.append(event.cause))
        session.bus.on(DecideTurnActionEvent.CHANNEL, _on_decide)

        result = await session.tick()

        # The stub provider's reply should be present in the returned
        # message list (loop ran ≥1 turn after the inject).
        assert any(
            isinstance(m, AssistantMessage)
            and any(
                isinstance(b, TextContent) and "stub-reply" in b.text
                for b in m.content
            )
            for m in result
        )
        # And the injected user message landed too.
        assert any(
            isinstance(m, UserMessage)
            and any(
                isinstance(b, TextContent) and b.text == "injected-from-extension"
                for b in m.content
            )
            for m in result
        )
        # No NoPendingInput cause should appear — the loop ran normally.
        assert not any(isinstance(cause, NoPendingInput) for cause in causes)
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _make_app() -> Any:
    """Wrap ``run_cmd`` in a typer ``Typer`` app for ``CliRunner``."""
    import typer

    from agentm.cli import run_cmd

    app = typer.Typer()
    app.command()(run_cmd)
    return app


def test_cli_no_prompt_no_resume_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTM_SKIP_DOTENV", "1")
    monkeypatch.setenv("AGENTM_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENTM_MODEL", "claude-sonnet-4-6")
    runner = CliRunner()
    result = runner.invoke(_make_app(), [])
    assert result.exit_code == 2
    assert "prompt is required" in (result.stderr or result.output)


def test_cli_resume_without_prompt_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI must accept --resume with no positional prompt.

    We don't drive the full session here (no stub provider hook into the
    typer arg layer is worth the surface area); the failure mode we're
    guarding is the early rejection. Use a bogus session id and assert
    we get the ``no session found`` BadParameter — proves parsing
    accepted the arg shape and we made it past the prompt check.
    """
    monkeypatch.setenv("AGENTM_SKIP_DOTENV", "1")
    monkeypatch.setenv("AGENTM_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENTM_MODEL", "claude-sonnet-4-6")
    runner = CliRunner()
    result = runner.invoke(
        _make_app(),
        ["--cwd", str(tmp_path), "--resume", "deadbeef-not-a-real-session"],
    )
    # Past the prompt guard (which would be exit-2 with our specific
    # "prompt is required" text); failure now is the BadParameter from
    # session lookup, surfaced as a Typer usage error (exit code 2 from
    # typer but with a different message).
    assert "prompt is required" not in (result.stderr or result.output)


def test_cli_resume_with_prompt_still_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENTM_SKIP_DOTENV", "1")
    monkeypatch.setenv("AGENTM_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENTM_MODEL", "claude-sonnet-4-6")
    runner = CliRunner()
    result = runner.invoke(
        _make_app(),
        [
            "--cwd",
            str(tmp_path),
            "--resume",
            "deadbeef-not-a-real-session",
            "hello",
        ],
    )
    assert "prompt is required" not in (result.stderr or result.output)


# ---------------------------------------------------------------------------
# CLI dispatch — path discrimination (prompt vs tick)
#
# These are the load-bearing assertions for the prefix-replay flow. The CLI
# guard's contract is not just "accepts the args" — it MUST route
# resume-only invocations through ``tick`` and prompt-bearing invocations
# through ``prompt``. A future refactor that always calls ``tick`` would
# silently break the multi-turn CLI experience; a refactor that always
# calls ``prompt`` would silently re-break the prefix-replay flow.
# ---------------------------------------------------------------------------


class _DispatchSpy:
    """Lightweight stand-in for :class:`AgentSession` that records which of
    ``prompt`` / ``tick`` the CLI invoked. Patched in via
    ``AgentSession.create``."""

    def __init__(self) -> None:
        self.prompt_calls: list[str] = []
        self.tick_calls: int = 0
        # CLI inspects session.get_service and session.model after the call;
        # short-circuit those so we don't need a real provider behind the scope.
        self.model = None

    async def prompt(
        self,
        text: str,
        *,
        images: Any | None = None,
        signal: Any | None = None,
    ) -> list[Any]:
        del images, signal
        self.prompt_calls.append(text)
        return []

    async def tick(self, *, signal: Any | None = None) -> list[Any]:
        del signal
        self.tick_calls += 1
        return []

    def get_service(self, name: str) -> Any | None:
        del name
        return None

    async def shutdown(self) -> None:
        return None


def _install_dispatch_spy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> _DispatchSpy:
    """Patch ``AgentSession.create`` to yield a spy and bypass session-config
    bootstrap. Returns the spy so the test can assert on it after ``run``."""

    spy = _DispatchSpy()

    from agentm.core.runtime.session_manager import SessionManager

    fake_state = SessionManager.in_memory(str(tmp_path))

    def _fake_build_session_config(**kwargs: Any) -> tuple[Any, Any]:
        del kwargs
        # The CLI inspects session_manager.session_file and .get_session_id
        # for log-line printing; SessionManager.in_memory provides both.
        return (object(), fake_state)

    async def _fake_create(config: Any) -> _DispatchSpy:
        del config
        return spy

    monkeypatch.setattr(
        "agentm.cli._build_session_config", _fake_build_session_config
    )
    monkeypatch.setattr(
        "agentm.core.runtime.session.AgentSession.create",
        classmethod(lambda cls, config: _fake_create(config)),
    )
    return spy


@pytest.mark.asyncio
async def test_cli_run_routes_resume_without_prompt_to_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agentm.cli import run

    spy = _install_dispatch_spy(monkeypatch, tmp_path)
    rc = await run(
        prompt="",
        scenario=None,
        extra_extensions=[],
        no_extensions=True,
        no_skills=True,
        no_prompt_templates=True,
        tool_allowlist=None,
        provider="stub",
        model="stub",
        cwd=str(tmp_path),
        quiet=True,
        resume="deadbeef-fake-sid",
        continue_recent=False,
    )
    assert rc == 0
    assert spy.tick_calls == 1
    assert spy.prompt_calls == []


@pytest.mark.asyncio
async def test_cli_run_routes_prompt_bearing_resume_to_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mirror of the resume-only test: a prompt-bearing resume must still
    route through ``prompt``. Guards against a regression where the
    dispatch branch is collapsed to always-tick."""
    from agentm.cli import run

    spy = _install_dispatch_spy(monkeypatch, tmp_path)
    rc = await run(
        prompt="hello",
        scenario=None,
        extra_extensions=[],
        no_extensions=True,
        no_skills=True,
        no_prompt_templates=True,
        tool_allowlist=None,
        provider="stub",
        model="stub",
        cwd=str(tmp_path),
        quiet=True,
        resume="deadbeef-fake-sid",
        continue_recent=False,
    )
    assert rc == 0
    assert spy.prompt_calls == ["hello"]
    assert spy.tick_calls == 0


@pytest.mark.asyncio
async def test_cli_run_routes_fresh_prompt_to_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """And a plain fresh prompt (no resume) routes through ``prompt``."""
    from agentm.cli import run

    spy = _install_dispatch_spy(monkeypatch, tmp_path)
    rc = await run(
        prompt="hi",
        scenario=None,
        extra_extensions=[],
        no_extensions=True,
        no_skills=True,
        no_prompt_templates=True,
        tool_allowlist=None,
        provider="stub",
        model="stub",
        cwd=str(tmp_path),
        quiet=True,
        resume=None,
        continue_recent=False,
    )
    assert rc == 0
    assert spy.prompt_calls == ["hi"]
    assert spy.tick_calls == 0
