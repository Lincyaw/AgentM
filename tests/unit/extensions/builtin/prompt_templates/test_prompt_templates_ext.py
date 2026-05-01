from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.core.kernel import TextContent, UserMessage
from agentm.harness.extension import CommandSpec
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig

from tests.unit.extensions.builtin import _helpers


@pytest.mark.asyncio
async def test_prompt_templates_expand_slash_input_before_agent_loop(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / ".agentm" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "refactor.md").write_text(
        "---\ndescription: Refactor helper\n---\nRefactor $1 because ${@:2}\n",
        encoding="utf-8",
    )

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.prompt_templates", {})],
            provider=(
                "tests.unit.extensions.builtin._helpers",
                {"response_texts": ["done"]},
            ),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        final = await session.prompt('/refactor foo.py "why now"')
        assert _helpers.LAST_STREAM is not None
        seen_messages = _helpers.LAST_STREAM.seen_messages[-1]
        first = seen_messages[0]
        assert isinstance(first, UserMessage)
        assert isinstance(first.content[0], TextContent)
        assert first.content[0].text == "Refactor foo.py because why now"
        assert isinstance(final[0], UserMessage)
        assert isinstance(final[0].content[0], TextContent)
        assert final[0].content[0].text == "Refactor foo.py because why now"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_code_commands_beat_prompt_templates_on_name_collision(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / ".agentm" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "refactor.md").write_text(
        "Refactor $1\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def install_command(api: Any, _config: dict[str, Any]) -> None:
        def handler(args: str, _api: Any) -> None:
            calls.append(args)

        api.register_command("refactor", CommandSpec(description="cmd", handler=handler))

    import sys
    import types

    module_name = "tests.unit.extensions.builtin._command_collision"
    module = types.ModuleType(module_name)
    module.install = install_command  # type: ignore[attr-defined]
    sys.modules[module_name] = module

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (module_name, {}),
                ("agentm.extensions.builtin.prompt_templates", {}),
            ],
            provider=(
                "tests.unit.extensions.builtin._helpers",
                {"response_texts": ["done"]},
            ),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        before_calls = _helpers.LAST_STREAM.calls if _helpers.LAST_STREAM is not None else 0
        await session.prompt("/refactor foo.py")
        after_calls = _helpers.LAST_STREAM.calls if _helpers.LAST_STREAM is not None else 0
        assert calls == ["foo.py"]
        assert after_calls == before_calls
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_unknown_slash_text_falls_through_verbatim(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.prompt_templates", {})],
            provider=(
                "tests.unit.extensions.builtin._helpers",
                {"response_texts": ["done"]},
            ),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        final = await session.prompt("/unknown thing")
        assert _helpers.LAST_STREAM is not None
        seen_messages = _helpers.LAST_STREAM.seen_messages[-1]
        first = seen_messages[0]
        assert isinstance(first, UserMessage)
        assert isinstance(first.content[0], TextContent)
        assert first.content[0].text == "/unknown thing"
        assert isinstance(final[0].content[0], TextContent)
        assert final[0].content[0].text == "/unknown thing"
    finally:
        await session.shutdown()
