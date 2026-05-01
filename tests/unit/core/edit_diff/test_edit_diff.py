from __future__ import annotations

import pytest

from agentm.core.edit_diff import (
    AppliedEditsResult,
    Edit,
    EditAmbiguous,
    EditNotFound,
    FuzzyMatchResult,
    apply_edits,
    detect_line_ending,
    fuzzy_find_text,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)


def test_detect_line_ending_round_trip_for_consistent_inputs() -> None:
    samples = [
        "alpha\nbeta\n",
        "alpha\r\nbeta\r\n",
        "single line",
    ]

    for sample in samples:
        assert (
            restore_line_endings(normalize_to_lf(sample), detect_line_ending(sample))
            == sample
        )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain\ntext", "\n"),
        ("plain\r\ntext", "\r\n"),
        ("no newline", "\n"),
        ("mixed\nthen\r\nlater", "\n"),
    ],
)
def test_detect_line_ending(content: str, expected: str) -> None:
    assert detect_line_ending(content) == expected


def test_normalize_for_fuzzy_match_converts_problem_unicode_to_ascii() -> None:
    text = "A\u00a0quote:\u2018hi\u2019\nDash\u2014space\u2009end  \n"

    assert normalize_for_fuzzy_match(text) == "A quote:'hi'\nDash-space end\n"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("abc", ("", "abc")),
        ("\ufeffabc", ("\ufeff", "abc")),
    ],
)
def test_strip_bom(content: str, expected: tuple[str, str]) -> None:
    assert strip_bom(content) == expected


def test_fuzzy_find_text_reports_exact_match_without_normalizing_content() -> None:
    content = "alpha -- beta\n"

    result = fuzzy_find_text(content, "--")

    assert result == FuzzyMatchResult(
        found=True,
        index=6,
        match_length=2,
        used_fuzzy_match=False,
        content_for_replacement=content,
    )


def test_fuzzy_find_text_uses_normalized_space_when_exact_match_fails() -> None:
    content = "alpha - beta  \n"

    result = fuzzy_find_text(content, "alpha — beta")

    assert result.found is True
    assert result.index == 0
    assert result.match_length == len("alpha - beta")
    assert result.used_fuzzy_match is True
    assert result.content_for_replacement == "alpha - beta\n"


def test_apply_edits_preserves_crlf_when_old_text_uses_lf() -> None:
    base = "first\r\nsecond\r\n"

    result = apply_edits(base, [Edit(old_text="first\nsecond\n", new_text="updated\nblock\n")])

    assert result == AppliedEditsResult(
        base_content="first\nsecond\n",
        new_content="updated\r\nblock\r\n",
    )


def test_apply_edits_preserves_utf8_bom() -> None:
    base = "\ufeffhello\nworld\n"

    result = apply_edits(base, [Edit(old_text="world", new_text="friend")])

    assert result.new_content == "\ufeffhello\nfriend\n"


def test_apply_edits_switches_to_fuzzy_normalized_content_after_first_fuzzy_hit() -> None:
    base = "A — B\nC\u00a0D  \n"
    edits = [
        Edit(old_text="A - B", new_text="X - Y"),
        Edit(old_text="C D", new_text="tail"),
    ]

    result = apply_edits(base, edits)

    assert result.base_content == "A - B\nC D\n"
    assert result.new_content == "X - Y\ntail\n"


def test_apply_edits_raises_not_found_with_helpful_message() -> None:
    with pytest.raises(EditNotFound, match="Could not find the exact text in content"):
        apply_edits("alpha\n", [Edit(old_text="missing", new_text="beta")])


def test_apply_edits_raises_ambiguous_for_multiple_exact_matches() -> None:
    with pytest.raises(EditAmbiguous, match="Found 2 occurrences of the exact text"):
        apply_edits("repeat\nrepeat\n", [Edit(old_text="repeat", new_text="done")])


def test_apply_edits_raises_ambiguous_for_multiple_fuzzy_matches() -> None:
    base = "alpha - beta\nalpha - beta\n"

    with pytest.raises(EditAmbiguous, match="Found 2 occurrences of the exact text"):
        apply_edits(base, [Edit(old_text="alpha — beta", new_text="done")])


def test_apply_edits_is_pure_for_identical_inputs() -> None:
    base = "left\nright\n"
    edits = [Edit(old_text="right", new_text="center")]

    first = apply_edits(base, edits)
    second = apply_edits(base, edits)

    assert first == second
    assert base == "left\nright\n"
    assert edits == [Edit(old_text="right", new_text="center")]
