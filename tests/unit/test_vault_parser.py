"""Focused regression tests for vault Markdown parser helpers."""

from __future__ import annotations

import pytest

from agentm.tools.vault.parser import (
    append_to_section,
    extract_title,
    extract_wikilinks,
    find_section,
    parse_note,
    replace_section,
    replace_string,
    serialize_note,
)


def test_parse_note_handles_frontmatter_and_no_frontmatter_cases() -> None:
    frontmatter, body = parse_note("---\ntitle: Hello\ntags: [a, b]\n---\n# Hello\nBody text.")
    assert frontmatter == {"title": "Hello", "tags": ["a", "b"]}
    assert body == "# Hello\nBody text."

    frontmatter, body = parse_note("# Heading\nBody")
    assert frontmatter == {}
    assert body == "# Heading\nBody"


def test_parse_note_treats_unclosed_frontmatter_as_plain_body() -> None:
    content = "---\ntitle: X\nmissing closing fence"
    frontmatter, body = parse_note(content)
    assert frontmatter == {}
    assert body == content


def test_serialize_note_round_trip_is_stable() -> None:
    original = "---\ntitle: Test\ntags:\n- a\n- b\ncustom: value\n---\n# Test\nBody text."
    fm, body = parse_note(original)

    rebuilt = serialize_note(fm, body)
    fm2, body2 = parse_note(rebuilt)

    assert fm2 == fm
    assert body2 == body


def test_extract_wikilinks_deduplicates_and_strips_md_extension() -> None:
    links = extract_wikilinks("[[note.md]] [[note]] [[folder/other]]")
    assert links == ["note", "folder/other"]


def test_extract_title_picks_first_h1_only() -> None:
    assert extract_title("# First\ntext\n# Second") == "First"
    assert extract_title("## Not h1\ntext") == ""


def test_find_section_respects_heading_level_boundaries() -> None:
    body = "# Title\n## Parent\nText\n### Child\nChild text\n## Sibling\nTail"
    span = find_section(body, "## Parent")
    assert span is not None
    start, end = span
    section_text = body[start:end]
    assert "### Child" in section_text
    assert "## Sibling" not in section_text


def test_replace_section_replaces_target_and_raises_for_missing_heading() -> None:
    body = "# Title\n## Section\nOld content\n## Other\nKeep"
    replaced = replace_section(body, "## Section", "New content\n")
    assert "New content" in replaced
    assert "Old content" not in replaced
    assert "## Other" in replaced

    with pytest.raises(ValueError):
        replace_section(body, "## Missing", "x")


def test_append_and_replace_string_cover_happy_and_error_paths() -> None:
    appended = append_to_section("## Section\nExisting\n## Next\nOther", "## Section", "Appended\n")
    assert "Existing\nAppended\n" in appended

    assert replace_string("Hello world", "world", "vault") == "Hello vault"
    with pytest.raises(ValueError, match="ambiguous"):
        replace_string("x x", "x", "y")
