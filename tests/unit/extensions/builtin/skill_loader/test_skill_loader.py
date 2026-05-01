from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agentm.core.skills import format_skills_for_prompt, load_skills
from agentm.harness.resource_loader import ContextFile, InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig

from tests.unit.extensions.builtin import _helpers


@pytest.mark.asyncio
async def test_skill_loader_injects_available_skills_block_and_grows_system_exactly(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "extra-skills"
    skill_dir = skill_root / "refactor"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\ndescription: Refactor code safely\n---\nBody\n",
        encoding="utf-8",
    )

    skills, diagnostics = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "unused-agent-dir"),
        skill_paths=[str(skill_root)],
        include_defaults=False,
    )
    assert diagnostics == []
    block = format_skills_for_prompt(skills)
    assert str(skill_file) in block

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.skill_loader",
                    {"skill_paths": [str(skill_root)], "include_defaults": False},
                )
            ],
            provider=(
                "tests.unit.extensions.builtin._helpers",
                {"response_texts": ["done"]},
            ),
            resource_loader=InMemoryResourceLoader(
                context_files=[ContextFile(body="BASE", source="mem")]
            ),
        )
    )

    try:
        await session.prompt("hello")
        assert _helpers.LAST_STREAM is not None
        assert _helpers.LAST_STREAM.seen_systems[-1] == f"BASE{block}"
        assert len(_helpers.LAST_STREAM.seen_systems[-1]) == len("BASE") + len(block)
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_skill_loader_uses_resources_discover_paths(tmp_path: Path) -> None:
    extra_root = tmp_path / "discovered-skills"
    skill_dir = extra_root / "analyze"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Analyze failures\n---\nBody\n",
        encoding="utf-8",
    )

    module_name = "tests.unit.extensions.builtin._resource_skill_paths"
    module = types.ModuleType(module_name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        def on_resources(_event: Any) -> dict[str, list[str]]:
            return {"skill_paths": [str(extra_root)]}

        api.on("resources_discover", on_resources)

    module.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = module

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (module_name, {}),
                ("agentm.extensions.builtin.skill_loader", {"include_defaults": False}),
            ],
            provider=(
                "tests.unit.extensions.builtin._helpers",
                {"response_texts": ["done"]},
            ),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        await session.prompt("hello")
        assert _helpers.LAST_STREAM is not None
        assert "<name>analyze</name>" in (_helpers.LAST_STREAM.seen_systems[-1] or "")
    finally:
        await session.shutdown()
