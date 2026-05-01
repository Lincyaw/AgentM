from __future__ import annotations

from agentm.core.frontmatter import parse_frontmatter


def test_parse_frontmatter_returns_metadata_and_body() -> None:
    metadata, body = parse_frontmatter(
        "---\nname: demo\ndescription: hello\n---\nBody\n"
    )

    assert metadata == {"name": "demo", "description": "hello"}
    assert body == "Body"


def test_parse_frontmatter_falls_back_on_invalid_yaml() -> None:
    text = "---\nname: [\n---\nBody\n"

    metadata, body = parse_frontmatter(text)

    assert metadata == {}
    assert body == text
