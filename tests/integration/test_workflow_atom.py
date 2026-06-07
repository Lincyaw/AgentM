"""Fail-stop integration test for the ``workflow`` atom.

Per ``CLAUDE.md`` testing philosophy: only the position where the atom's
value proposition fails when broken. For ``workflow`` that position is the
**journal-resume contract** (design §3.3): a re-run of a workflow must return
the cached ``agent()`` result *without re-spawning* a child session. If the
hash key drifts or the journal lookup misfires, the whole "resume only the
new/changed calls" property is gone and a workflow becomes non-idempotent.

The test drives a tiny JS script through ``agent()`` then ``parallel()``,
runs it twice in the same session, and asserts the second run spawns zero new
child sessions while returning the same results.

Skips cleanly when the optional QuickJS engine is not installed — the atom
gate is exercised separately by the unit-level import-gate path.
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.core.abi.events import ChildSessionStartEvent
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.messages import AssistantMessage, Usage
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


quickjs = pytest.importorskip(
    "quickjs", reason="workflow atom needs the optional QuickJS engine"
)


_PROVIDER_MODULE = "agentm._tests.workflow_echo_provider"


def _install_echo_provider() -> str:
    """Provider that echoes the last user message text back as the reply,
    with a small token usage so the budget aggregator has something to sum.

    Echoing the prompt lets the test assert *which* prompt produced a cached
    result — a journal key collision would surface as a wrong echo."""

    module = types.ModuleType(_PROVIDER_MODULE)

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, system, signal, thinking
        text = ""
        for msg in reversed(messages):
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = [
                        getattr(b, "text", "")
                        for b in content
                        if getattr(b, "text", None)
                    ]
                    text = " ".join(parts)
                break
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"echo:{text}")],
                timestamp=0.0,
                stop_reason="end_turn",
                usage=Usage(input_tokens=5, output_tokens=7),
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "workflow-echo",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="workflow-echo",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="workflow-echo",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _git_init(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", "-b", "agent-tests", str(path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )


def _tool(session: AgentSession, name: str) -> Any:
    for tool in session.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"missing tool {name}")


_SCRIPT = """
const a = agent("alpha");
const rest = parallel([{prompt: "beta"}, {prompt: "gamma"}]);
JSON.stringify([a, rest[0], rest[1]]);
"""


@pytest.mark.asyncio
async def test_workflow_journal_resume_skips_respawn(tmp_path: Path) -> None:
    _git_init(tmp_path)
    provider_module = _install_echo_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[
                ("agentm.extensions.builtin.operations_local", {}),
                ("agentm.extensions.builtin.artifact_store", {}),
                ("agentm.extensions.builtin.workflow", {}),
            ],
        )
    )

    spawned: list[str] = []
    session.bus.on(
        ChildSessionStartEvent.CHANNEL,
        lambda e: spawned.append(getattr(e, "child_session_id", "?")),
    )

    try:
        tool = _tool(session, "workflow")

        first = await tool.execute({"script": _SCRIPT})
        assert not first.is_error, first.content[0].text
        # alpha + beta + gamma = three distinct prompts -> three children.
        assert len(spawned) == 3
        first_text = first.content[0].text
        assert "echo:alpha" in first_text
        assert "echo:beta" in first_text
        assert "echo:gamma" in first_text
        assert first.extras["agents_spawned"] == 3
        # Budget aggregated across the three children (5+7 tokens each).
        assert first.extras["budget"]["spent"] == 36

        spawned.clear()
        second = await tool.execute({"script": _SCRIPT})
        assert not second.is_error, second.content[0].text
        # Journal resume: every agent() call hits the cache -> no new spawns.
        assert spawned == []
        assert second.extras["agents_spawned"] == 0
        assert second.content[0].text == first_text
    finally:
        await session.shutdown()
