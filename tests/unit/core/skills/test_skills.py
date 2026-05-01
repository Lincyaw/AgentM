from __future__ import annotations

from pathlib import Path

from agentm.core.skills import format_skills_for_prompt, load_skills


def _write_skill(
    base: Path,
    name: str,
    *,
    description: str,
    disable_model_invocation: bool = False,
    frontmatter_name: str | None = None,
) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    if frontmatter_name is not None:
        lines.append(f"name: {frontmatter_name}")
    lines.append(f"description: {description}")
    if disable_model_invocation:
        lines.append("disable-model-invocation: true")
    lines.append("---")
    lines.append("Body")
    path = skill_dir / "SKILL.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_skills_discovers_project_skill_and_formats_prompt(tmp_path: Path) -> None:
    project_skills = tmp_path / ".agentm" / "skills"
    skill_path = _write_skill(project_skills, "refactor", description="Refactor code safely")

    skills, diagnostics = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "missing-agent-dir"),
    )

    assert diagnostics == []
    assert [skill.name for skill in skills] == ["refactor"]
    block = format_skills_for_prompt(skills)
    assert block.startswith("\n\nThe following skills provide specialized instructions")
    assert f"<location>{skill_path}</location>" in block
    assert "<name>refactor</name>" in block
    assert "<description>Refactor code safely</description>" in block


def test_load_skills_drops_empty_description_and_reports_collisions(tmp_path: Path) -> None:
    agent_skill_root = tmp_path / "agent-home" / "skills"
    project_skill_root = tmp_path / ".agentm" / "skills"
    _write_skill(agent_skill_root, "duplicate", description="First copy")
    _write_skill(project_skill_root, "duplicate", description="Second copy")
    _write_skill(project_skill_root, "empty", description="   ")

    skills, diagnostics = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "agent-home"),
    )

    assert [skill.name for skill in skills] == ["duplicate"]
    assert skills[0].description == "First copy"
    assert any(diag.level == "collision" for diag in diagnostics)
    assert any(diag.message == "description is required" for diag in diagnostics)


def test_disable_model_invocation_hides_skill_from_prompt(tmp_path: Path) -> None:
    skill_root = tmp_path / ".agentm" / "skills"
    _write_skill(skill_root, "explicit-only", description="Only explicit", disable_model_invocation=True)

    skills, diagnostics = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "missing-agent-dir"),
    )

    assert diagnostics == []
    assert [skill.name for skill in skills] == ["explicit-only"]
    assert format_skills_for_prompt(skills) == ""


def test_name_validation_warns_but_still_loads_skill(tmp_path: Path) -> None:
    skill_root = tmp_path / ".agentm" / "skills"
    _write_skill(
        skill_root,
        "refactor",
        description="Still loaded",
        frontmatter_name="Refactor",
    )

    skills, diagnostics = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "missing-agent-dir"),
    )

    assert [skill.name for skill in skills] == ["Refactor"]
    assert any(diag.level == "warning" for diag in diagnostics)


def test_symlink_loops_do_not_hang_loader(tmp_path: Path) -> None:
    skill_root = tmp_path / ".agentm" / "skills"
    loop_dir = skill_root / "loop"
    _write_skill(skill_root, "loop", description="Loop safe")
    (loop_dir / "nested").mkdir(exist_ok=True)
    (loop_dir / "nested" / "back").symlink_to(loop_dir, target_is_directory=True)

    skills, diagnostics = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "missing-agent-dir"),
    )

    assert diagnostics == []
    assert [skill.name for skill in skills] == ["loop"]
