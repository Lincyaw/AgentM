"""Fail-stop integration tests for the ``workflow`` atom.

Per ``CLAUDE.md`` testing philosophy: only the positions where the atom's
value proposition fails when broken.

1. **Journal-resume idempotence** (design §3.3): a re-run must return the
   cached ``agent()`` result *without re-spawning*. The disk-backed lookup is
   asserted directly (fresh ``_Journal``, empty in-memory cache) so a
   regression in ``list_artifacts``/``read`` can't hide behind the cache.
2. **Budget total/remaining**: the configured ceiling must back
   ``budget.remaining()``, not just raw ``spent``.
3. **pipeline staged flow**: each item flows through all stages (the real
   multi-stage primitive, not a parallel alias).
4. **Curated namespace**: the script cannot reach ``open`` (guardrail).

A stub echo provider stands in for the LLM so child sessions are real but
deterministic.
"""

from __future__ import annotations

import json
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
from agentm.core.abi import ChildSessionStartEvent
from agentm.core.abi import ProviderConfig
from agentm.core.abi import AssistantMessage, Usage
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession
from agentm.extensions.builtin.workflow import _Journal


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


async def _make_session(tmp_path: Path) -> AgentSession:
    _git_init(tmp_path)
    provider_module = _install_echo_provider()
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[
                ("agentm.extensions.builtin.operations", {}),
                ("agentm.extensions.builtin.artifact_store", {}),
                ("agentm.extensions.builtin.workflow", {"budget_tokens": 100}),
            ],
        )
    )


_RESUME_SCRIPT = """
a = await agent("alpha")
rest = await parallel([agent("beta"), agent("gamma")])
return [a, rest[0], rest[1]]
"""


@pytest.mark.asyncio
async def test_workflow_journal_resume_skips_respawn(tmp_path: Path) -> None:
    session = await _make_session(tmp_path)
    spawned: list[str] = []
    session.bus.on(
        ChildSessionStartEvent.CHANNEL,
        lambda e: spawned.append(getattr(e, "child_session_id", "?")),
    )

    try:
        tool = _tool(session, "workflow")

        first = await tool.execute({"script": _RESUME_SCRIPT})
        assert not first.is_error, first.content[0].text
        # alpha + beta + gamma = three distinct prompts -> three children.
        assert len(spawned) == 3
        first_text = first.content[0].text
        assert "echo:alpha" in first_text
        assert "echo:beta" in first_text
        assert "echo:gamma" in first_text
        assert first.extras["summary"]["agents_spawned"] == 3
        # Budget aggregated across the three children (5+7 tokens each).
        assert first.extras["summary"]["budget"]["spent"] == 36

        # Load-bearing resume path: a fresh _Journal (empty in-memory cache)
        # must resolve the first run's key from artifact_store on DISK
        # (list_artifacts -> read), not just from an in-process cache.
        store = session.get_service("artifact_store")
        assert store is not None
        cold = _Journal(store=store)
        cached_alpha = await cold.lookup(
            _Journal.key(
                "alpha",
                {
                    "schema": None,
                    "scenario": None,
                    "isolation": None,
                    "tool_allowlist": None,
                    "extra_extensions": None,
                    "atom_config": None,
                },
            )
        )
        assert cached_alpha is not None and "echo:alpha" in cached_alpha

        spawned.clear()
        second = await tool.execute({"script": _RESUME_SCRIPT})
        assert not second.is_error, second.content[0].text
        # Journal resume: every agent() call hits the cache -> no new spawns.
        assert spawned == []
        assert second.extras["summary"]["agents_spawned"] == 0
        assert second.content[0].text == first_text
    finally:
        await session.shutdown()


_BUDGET_SCRIPT = """
await agent("one")
return {"total": budget.total, "spent": budget.spent(), "remaining": budget.remaining()}
"""


@pytest.mark.asyncio
async def test_workflow_budget_total_and_remaining(tmp_path: Path) -> None:
    """budget.total / .remaining() are live when a ceiling is configured —
    guards the BudgetService contract beyond raw ``spent``."""

    session = await _make_session(tmp_path)
    try:
        tool = _tool(session, "workflow")
        result = await tool.execute({"script": _BUDGET_SCRIPT})
        assert not result.is_error, result.content[0].text
        snap = json.loads(result.content[0].text)
        # one child: 5 input + 7 output = 12 spent; total 100 -> remaining 88.
        assert snap["spent"] == 12
        assert snap["total"] == 100
        assert snap["remaining"] == 88
    finally:
        await session.shutdown()


_PIPELINE_SCRIPT = """
def upper(s):
    return s.upper()

async def echo_again(s):
    return await agent(s)

# two stages per item: async agent stage, then a sync transform stage.
results = await pipeline(args["items"], echo_again, lambda r: upper(r))
return results
"""


@pytest.mark.asyncio
async def test_workflow_pipeline_runs_each_item_through_stages(
    tmp_path: Path,
) -> None:
    """pipeline threads every item through every stage (real multi-stage
    primitive). Two items x (async agent stage + sync transform stage)."""

    session = await _make_session(tmp_path)
    try:
        tool = _tool(session, "workflow")
        result = await tool.execute(
            {"script": _PIPELINE_SCRIPT, "args": {"items": ["x", "y"]}}
        )
        assert not result.is_error, result.content[0].text
        out = json.loads(result.content[0].text)
        # each item: agent("x") -> "echo:x" -> upper -> "ECHO:X"
        assert out == ["ECHO:X", "ECHO:Y"]
        assert result.extras["summary"]["agents_spawned"] == 2
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_workflow_tool_absent_in_worker_session(tmp_path: Path) -> None:
    """Anti-recursion: a session spawned as a workflow worker (purpose=workflow)
    must NOT register the workflow tool, even though the atom is loaded — else a
    worker could spawn unbounded nested workflows."""

    _git_init(tmp_path)
    provider_module = _install_echo_provider()
    worker = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[
                ("agentm.extensions.builtin.operations", {}),
                ("agentm.extensions.builtin.artifact_store", {}),
                ("agentm.extensions.builtin.workflow", {}),
            ],
            purpose="workflow",
        )
    )
    try:
        assert "workflow" not in {t.name for t in worker.tools}
    finally:
        await worker.shutdown()


_GUARDRAIL_SCRIPT = """
return open("/etc/passwd").read()
"""


@pytest.mark.asyncio
async def test_workflow_curated_namespace_blocks_open(tmp_path: Path) -> None:
    """The curated namespace omits ``open`` — an honest script reaching for the
    filesystem fails as a script error rather than reading host files. (A
    guardrail, not a security wall; see the atom docstring.)"""

    session = await _make_session(tmp_path)
    try:
        tool = _tool(session, "workflow")
        result = await tool.execute({"script": _GUARDRAIL_SCRIPT})
        assert result.is_error
        error_text = result.content[0].text
        assert "open()" in error_text or "workflow script error" in error_text
    finally:
        await session.shutdown()
