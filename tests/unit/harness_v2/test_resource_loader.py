"""Tests for ``agentm.harness.resource_loader``."""

from __future__ import annotations

from pathlib import Path

from agentm.harness.resource_loader import (
    ContextFile,
    DefaultResourceLoader,
    InMemoryResourceLoader,
    PromptTemplate,
    Skill,
)


def test_default_loader_discovers_skills_with_frontmatter(tmp_path: Path) -> None:
    """A SKILL.md with YAML frontmatter is parsed; name/description are taken
    from the frontmatter, body is the remaining markdown."""

    skill_dir = tmp_path / ".agentm" / "skills" / "foo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: foo\ndescription: A demo skill\n---\nBody of foo.\n",
        encoding="utf-8",
    )

    loader = DefaultResourceLoader(cwd=tmp_path, agent_dir=tmp_path / "_no_agent_dir_")
    skills = loader.get_skills()

    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "foo"
    assert skill.description == "A demo skill"
    assert "Body of foo." in skill.body


def test_default_loader_discovers_prompt_templates(tmp_path: Path) -> None:
    """Prompt files become templates whose ``name`` is the filename stem and
    ``body`` is the file content verbatim."""

    prompts = tmp_path / ".agentm" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "greet.md").write_text("Hello {name}.", encoding="utf-8")

    loader = DefaultResourceLoader(cwd=tmp_path, agent_dir=tmp_path / "_no_agent_dir_")
    templates = loader.get_prompt_templates()

    assert [t.name for t in templates] == ["greet"]
    assert templates[0].body == "Hello {name}."


def test_default_loader_walks_ancestors_for_context_files(tmp_path: Path) -> None:
    """AGENTS.md files in cwd and an ancestor are both loaded; the ancestor
    appears first (root → cwd ordering)."""

    parent = tmp_path
    child = tmp_path / "child"
    child.mkdir()
    (parent / "AGENTS.md").write_text("PARENT", encoding="utf-8")
    (child / "AGENTS.md").write_text("CHILD", encoding="utf-8")

    loader = DefaultResourceLoader(cwd=child, agent_dir=tmp_path / "_no_agent_dir_")
    files = loader.get_context_files()

    bodies = [cf.body for cf in files]
    # Parent must appear before child.
    assert "PARENT" in bodies
    assert "CHILD" in bodies
    assert bodies.index("PARENT") < bodies.index("CHILD")


def test_default_loader_reload_picks_up_new_skill(tmp_path: Path) -> None:
    """``reload()`` flushes the cache so subsequent calls observe new files."""

    loader = DefaultResourceLoader(cwd=tmp_path, agent_dir=tmp_path / "_no_agent_dir_")
    assert loader.get_skills() == []

    skill_dir = tmp_path / ".agentm" / "skills" / "bar"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: bar\ndescription: Bar skill\n---\nbody",
        encoding="utf-8",
    )

    # Without reload, cache still empty.
    assert loader.get_skills() == []
    loader.reload()
    skills = loader.get_skills()
    assert [s.name for s in skills] == ["bar"]


def test_in_memory_loader_returns_what_it_was_given() -> None:
    """The in-memory loader is a trivial pass-through suitable for tests and
    embedded SDK use."""

    skills = [Skill(name="s", description="d", body="b", source="mem")]
    prompts = [PromptTemplate(name="p", body="x", source="mem")]
    contexts = [ContextFile(body="c", source="mem")]

    loader = InMemoryResourceLoader(
        skills=skills,
        prompt_templates=prompts,
        context_files=contexts,
    )

    assert loader.get_skills() == skills
    assert loader.get_prompt_templates() == prompts
    assert loader.get_context_files() == contexts
