from __future__ import annotations

from pathlib import Path

from agentm.core.prompt_templates import (
    PromptTemplateRecord,
    expand_prompt_template,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)


def test_parse_command_args_handles_quoted_groups() -> None:
    assert parse_command_args('"hello world" foo') == ["hello world", "foo"]


def test_substitute_args_supports_slices_and_non_recursive_positional_values() -> None:
    body = "first=$1 rest=${@:2} one=${@:2:1} all=$ARGUMENTS alias=$@"
    args = ["literal $1", "two", "three"]

    expanded = substitute_args(body, args)

    assert expanded == (
        "first=literal $1 rest=two three one=two "
        "all=literal $1 two three alias=literal $1 two three"
    )


def test_load_prompt_templates_uses_frontmatter_and_fallback_description(tmp_path: Path) -> None:
    prompts_dir = tmp_path / ".agentm" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "refactor.md").write_text(
        "---\ndescription: Refactor helper\nargument-hint: <path>\n---\nRefactor $1\n",
        encoding="utf-8",
    )
    (prompts_dir / "explain.md").write_text(
        "\n\nExplain the bug in detail and include next steps if needed.\n",
        encoding="utf-8",
    )
    (prompts_dir / "bad.md").write_text("---\ndescription: [\n---\nBody\n", encoding="utf-8")

    templates = load_prompt_templates(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "missing-agent-dir"),
    )

    assert [template.name for template in templates] == ["explain", "refactor"]
    explain, refactor = templates
    assert explain.description == "Explain the bug in detail and include next steps if needed."
    assert refactor.argument_hint == "<path>"
    assert refactor.description == "Refactor helper"


def test_expand_prompt_template_returns_none_for_unknown_name() -> None:
    templates = [
        PromptTemplateRecord(
            name="refactor",
            description="Refactor helper",
            argument_hint=None,
            body="Refactor $1",
            file_path="/tmp/refactor.md",
            source="project",
        )
    ]

    assert expand_prompt_template("/unknown foo", templates) is None
    assert expand_prompt_template("/refactor foo.py", templates) == "Refactor foo.py"
