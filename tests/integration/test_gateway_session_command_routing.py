"""Gateway must forward session-registered slash commands to the session.

Fail-stop for the single-process gateway's command-routing seam. The gateway
has two command layers: its own builtins (``/status``, ``/help``, ...) routed
by ``CommandRouter``, and session-registered commands (``/compact`` and
friends, installed by atoms like ``llm_compaction``) dispatched INSIDE the
session by the ``slash_commands`` floor atom. The bug: a slash command the
gateway registry didn't own was rejected with an "unknown command" diagnostic
and NEVER forwarded to the session, so ``/compact`` could not run through the
gateway at all (it ended up handled as plain text by the model).

The fix: ``CommandRouter.try_dispatch`` returns ``None`` for a name no gateway
handler owns, and ``_run_command`` forwards such a command to the session
prompt path (``_prompt_session`` with the original ``/...`` content). The
gateway distinguishes a known session command (forward) from a name unknown to
both layers (surface the diagnostic) using a per-session known-command set it
learns from the ``session_ready`` frame.

These tests drive ``GatewayRuntime`` directly with stubs (no real LLM): they
assert the routing verdict — which path an inbound takes — rather than re-test
the in-session dispatch (covered by the compaction/slash_commands suites).
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.runtime import GatewayRuntime
from agentm.gateway.commands import CommandRouter, discover_commands
from agentm.gateway.outbox import SqliteOutbox
from agentm.gateway.wire import InboundBody


async def _unused_factory(*_a: Any) -> Any:  # _prompt_session is stubbed out
    raise AssertionError("session factory must not be called in these tests")


def _runtime(tmp_path: Any) -> tuple[GatewayRuntime, SqliteOutbox]:
    outbox = SqliteOutbox(str(tmp_path / "o.sqlite"))
    runtime = GatewayRuntime(
        cwd=".",
        scenario="local",
        outbox=outbox,
        chat_map=ChatSessionMap(tmp_path / "m.json"),
        session_factory=_unused_factory,
        command_router=CommandRouter(registry=discover_commands(".")),
        approval_policy=(frozenset(), frozenset(), 300.0),
        model_name="doubao",
        make_factory=lambda _n: _unused_factory,
    )
    return runtime, outbox


def _inbound(content: str) -> InboundBody:
    return InboundBody(
        channel="terminal",
        chat_id="c1",
        content=content,
        sender_id="u1",
    )


def _capture_prompt(runtime: GatewayRuntime, sink: list[tuple[str, str]]) -> None:
    async def _prompt(session_key: str, scenario: str | None, body: InboundBody) -> None:
        del scenario
        sink.append((session_key, body.content))

    runtime._prompt_session = _prompt  # type: ignore[method-assign]


def _capture_outbound(runtime: GatewayRuntime, sink: list[dict[str, Any]]) -> None:
    async def _emit(body: dict[str, Any]) -> None:
        sink.append(body)

    runtime._emit_outbound = _emit  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_known_session_command_is_forwarded_to_session(tmp_path: Any) -> None:
    runtime, outbox = _runtime(tmp_path)
    try:
        # The gateway learned /compact from a session_ready frame.
        runtime._session_commands["terminal:c1"] = {"compact"}
        prompts: list[tuple[str, str]] = []
        outbound: list[dict[str, Any]] = []
        _capture_prompt(runtime, prompts)
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/compact"))

        # Forwarded verbatim to the session prompt path; NO diagnostic.
        assert prompts == [("terminal:c1", "/compact")]
        assert not any(
            o.get("metadata", {}).get("kind") == "diagnostic_error"
            for o in outbound
        )
    finally:
        outbox.close()


@pytest.mark.asyncio
async def test_session_command_forwarded_when_set_unknown(tmp_path: Any) -> None:
    # Before any session_ready frame the gateway has no per-session set; it
    # forwards optimistically rather than rejecting a real command.
    runtime, outbox = _runtime(tmp_path)
    try:
        prompts: list[tuple[str, str]] = []
        outbound: list[dict[str, Any]] = []
        _capture_prompt(runtime, prompts)
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/compact"))

        assert prompts == [("terminal:c1", "/compact")]
        assert not any(
            o.get("metadata", {}).get("kind") == "diagnostic_error"
            for o in outbound
        )
    finally:
        outbox.close()


@pytest.mark.asyncio
async def test_gateway_builtin_still_dispatches_locally(tmp_path: Any) -> None:
    runtime, outbox = _runtime(tmp_path)
    try:
        prompts: list[tuple[str, str]] = []
        outbound: list[dict[str, Any]] = []
        _capture_prompt(runtime, prompts)
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/status"))

        # A genuine builtin dispatches at the gateway: it replies, it is NOT
        # forwarded to the session prompt path.
        assert prompts == []
        assert outbound, "/status should emit a reply"
    finally:
        outbox.close()


@pytest.mark.asyncio
async def test_unknown_to_both_yields_diagnostic(tmp_path: Any) -> None:
    runtime, outbox = _runtime(tmp_path)
    try:
        # Session set is known and does NOT contain asdf.
        runtime._session_commands["terminal:c1"] = {"compact"}
        prompts: list[tuple[str, str]] = []
        outbound: list[dict[str, Any]] = []
        _capture_prompt(runtime, prompts)
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/asdf"))

        # Unknown to gateway AND session -> diagnostic, never forwarded.
        assert prompts == []
        kinds = [o.get("metadata", {}).get("kind") for o in outbound]
        assert "diagnostic_error" in kinds
    finally:
        outbox.close()
