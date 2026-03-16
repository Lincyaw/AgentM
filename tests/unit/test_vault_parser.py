"""Tests for vault parser — pure functions for Markdown + YAML frontmatter manipulation."""

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


# ---------------------------------------------------------------------------
# parse_note
# ---------------------------------------------------------------------------

class TestParseNote:
    def test_should_split_frontmatter_and_body(self):
        content = "---\ntitle: Hello\ntags: [a, b]\n---\n# Hello\nBody text."
        fm, body = parse_note(content)
        assert fm == {"title": "Hello", "tags": ["a", "b"]}
        assert body == "# Hello\nBody text."

    def test_should_return_empty_dict_when_no_frontmatter(self):
        content = "# Just a heading\nSome body."
        fm, body = parse_note(content)
        assert fm == {}
        assert body == "# Just a heading\nSome body."

    def test_should_return_empty_dict_for_empty_string(self):
        fm, body = parse_note("")
        assert fm == {}
        assert body == ""

    def test_should_preserve_unknown_fields(self):
        content = "---\ntitle: X\ncustom_field: 42\nanother: [1,2,3]\n---\nBody"
        fm, body = parse_note(content)
        assert fm["custom_field"] == 42
        assert fm["another"] == [1, 2, 3]

    def test_should_handle_frontmatter_only_no_body(self):
        content = "---\ntitle: X\n---\n"
        fm, body = parse_note(content)
        assert fm == {"title": "X"}
        assert body == ""

    def test_should_handle_body_with_triple_dashes_not_at_start(self):
        content = "Some text\n---\ntitle: X\n---\nMore text"
        fm, body = parse_note(content)
        assert fm == {}
        assert body == content

    def test_should_handle_unclosed_frontmatter(self):
        content = "---\ntitle: X\nno closing fence"
        fm, body = parse_note(content)
        assert fm == {}
        assert body == content

    def test_should_handle_non_dict_yaml_frontmatter(self):
        content = "---\n- item1\n- item2\n---\nBody"
        fm, body = parse_note(content)
        assert fm == {}
        assert body == content


# ---------------------------------------------------------------------------
# serialize_note
# ---------------------------------------------------------------------------

class TestSerializeNote:
    def test_should_produce_valid_frontmatter_and_body(self):
        fm = {"title": "Hello"}
        body = "# Hello\nBody."
        result = serialize_note(fm, body)
        assert result.startswith("---\n")
        assert "\n---\n" in result
        assert result.endswith("# Hello\nBody.")

    def test_should_roundtrip_with_parse(self):
        original = "---\ntitle: Test\ntags:\n- a\n- b\ncustom: value\n---\n# Test\nBody text."
        fm, body = parse_note(original)
        rebuilt = serialize_note(fm, body)
        fm2, body2 = parse_note(rebuilt)
        assert fm2 == fm
        assert body2 == body

    def test_should_handle_empty_frontmatter(self):
        result = serialize_note({}, "Just body.")
        assert result == "Just body."

    def test_should_handle_empty_body(self):
        result = serialize_note({"title": "X"}, "")
        fm, body = parse_note(result)
        assert fm == {"title": "X"}
        assert body == ""


# ---------------------------------------------------------------------------
# extract_wikilinks
# ---------------------------------------------------------------------------

class TestExtractWikilinks:
    def test_should_find_wikilinks_in_body(self):
        text = "See [[note-a]] and [[note-b]] for details."
        assert extract_wikilinks(text) == ["note-a", "note-b"]

    def test_should_deduplicate(self):
        text = "[[dup]] and [[dup]] again."
        result = extract_wikilinks(text)
        assert result == ["dup"]

    def test_should_return_empty_list_when_none(self):
        assert extract_wikilinks("no links here") == []

    def test_should_find_links_with_slashes(self):
        text = "[[folder/sub/note]]"
        assert extract_wikilinks(text) == ["folder/sub/note"]

    def test_should_strip_md_extension(self):
        text = "[[note.md]] and [[other]]"
        result = extract_wikilinks(text)
        assert "note" in result
        assert "other" in result

    def test_should_find_links_in_frontmatter_strings(self):
        content = "---\nrelated: '[[linked-note]]'\n---\nBody with [[body-link]]."
        links = extract_wikilinks(content)
        assert "linked-note" in links
        assert "body-link" in links


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_should_extract_first_h1(self):
        body = "# My Title\nSome content\n## Subtitle"
        assert extract_title(body) == "My Title"

    def test_should_return_empty_when_no_h1(self):
        body = "## Not an h1\nSome text"
        assert extract_title(body) == ""

    def test_should_return_empty_for_empty_body(self):
        assert extract_title("") == ""

    def test_should_pick_first_h1_only(self):
        body = "# First\ntext\n# Second"
        assert extract_title(body) == "First"

    def test_should_not_match_inline_hash(self):
        body = "Not a heading # here\n# Actual Title"
        assert extract_title(body) == "Actual Title"


# ---------------------------------------------------------------------------
# find_section
# ---------------------------------------------------------------------------

class TestFindSection:
    def test_should_find_section_between_headings(self):
        body = "# Title\nIntro\n## Section A\nContent A\n## Section B\nContent B"
        result = find_section(body, "## Section A")
        assert result is not None
        start, end = result
        extracted = body[start:end]
        assert "## Section A" in extracted
        assert "Content A" in extracted
        assert "## Section B" not in extracted

    def test_should_find_section_at_end_of_file(self):
        body = "# Title\n## Last Section\nFinal content"
        result = find_section(body, "## Last Section")
        assert result is not None
        start, end = result
        assert body[start:end].strip().endswith("Final content")

    def test_should_return_none_for_missing_heading(self):
        body = "# Title\nContent"
        assert find_section(body, "## Nonexistent") is None

    def test_should_handle_empty_section(self):
        body = "## Empty\n## Next"
        result = find_section(body, "## Empty")
        assert result is not None
        start, end = result
        section_text = body[start:end]
        assert "## Empty" in section_text
        assert "## Next" not in section_text

    def test_should_include_nested_subheadings(self):
        body = "# Title\n## Parent\nText\n### Child\nChild text\n## Sibling"
        result = find_section(body, "## Parent")
        assert result is not None
        start, end = result
        section_text = body[start:end]
        assert "### Child" in section_text
        assert "Child text" in section_text
        assert "## Sibling" not in section_text

    def test_should_return_none_for_non_heading_string(self):
        body = "## Real\nContent"
        assert find_section(body, "not a heading") is None

    def test_should_match_exact_heading_level(self):
        body = "## Heading\nContent\n# Higher Level"
        result = find_section(body, "## Heading")
        assert result is not None
        start, end = result
        assert "# Higher Level" not in body[start:end]


# ---------------------------------------------------------------------------
# replace_section
# ---------------------------------------------------------------------------

class TestReplaceSection:
    def test_should_replace_section_content(self):
        body = "# Title\n## Section\nOld content\n## Other\nKeep"
        result = replace_section(body, "## Section", "New content\n")
        assert "New content" in result
        assert "Old content" not in result
        assert "## Section" in result
        assert "## Other" in result
        assert "Keep" in result

    def test_should_replace_last_section(self):
        body = "# Title\n## Last\nOld stuff"
        result = replace_section(body, "## Last", "New stuff")
        assert "New stuff" in result
        assert "Old stuff" not in result

    def test_should_raise_when_heading_not_found(self):
        body = "# Title\nContent"
        with pytest.raises(ValueError):
            replace_section(body, "## Missing", "text")


# ---------------------------------------------------------------------------
# append_to_section
# ---------------------------------------------------------------------------

class TestAppendToSection:
    def test_should_append_before_next_heading(self):
        body = "## Section\nExisting\n## Next\nOther"
        result = append_to_section(body, "## Section", "Appended line\n")
        assert "Existing\nAppended line\n" in result
        assert "## Next" in result

    def test_should_append_at_end_of_file(self):
        body = "## Section\nExisting"
        result = append_to_section(body, "## Section", "\nAppended")
        assert result.endswith("Appended")

    def test_should_raise_when_heading_not_found(self):
        body = "# Title"
        with pytest.raises(ValueError):
            append_to_section(body, "## Missing", "text")


# ---------------------------------------------------------------------------
# replace_string
# ---------------------------------------------------------------------------

class TestReplaceString:
    def test_should_replace_exact_match(self):
        body = "Hello world, goodbye world."
        result = replace_string(body, "Hello world", "Hi earth")
        assert result == "Hi earth, goodbye world."

    def test_should_raise_when_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            replace_string("abc", "xyz", "new")

    def test_should_raise_when_ambiguous(self):
        with pytest.raises(ValueError, match="ambiguous"):
            replace_string("foo bar foo", "foo", "baz")

    def test_should_handle_multiline_old_string(self):
        body = "line1\nline2\nline3"
        result = replace_string(body, "line1\nline2", "replaced")
        assert result == "replaced\nline3"
