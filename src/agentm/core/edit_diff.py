"""Pure text-edit helpers shared by future file mutation tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
import unicodedata

LineEnding = Literal["\n", "\r\n"]
_BOM = "\ufeff"
_SINGLE_QUOTES_RE = re.compile(r"[\u2018\u2019\u201A\u201B]")
_DOUBLE_QUOTES_RE = re.compile(r"[\u201C\u201D\u201E\u201F]")
_DASHES_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_SPACES_RE = re.compile(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]")


class EditNotFound(ValueError):
    """Raised when an edit target cannot be found."""


class EditAmbiguous(ValueError):
    """Raised when an edit target matches multiple locations."""


@dataclass(frozen=True, slots=True)
class Edit:
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


@dataclass(frozen=True, slots=True)
class AppliedEditsResult:
    base_content: str
    new_content: str


def detect_line_ending(content: str) -> LineEnding:
    """Detect the dominant explicit line ending used by *content*."""

    crlf_index = content.index("\r\n") if "\r\n" in content else -1
    lf_index = content.index("\n") if "\n" in content else -1
    if lf_index == -1:
        return "\n"
    if crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"

def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def restore_line_endings(text: str, ending: LineEnding) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text

def normalize_for_fuzzy_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _SINGLE_QUOTES_RE.sub("'", normalized)
    normalized = _DOUBLE_QUOTES_RE.sub('"', normalized)
    normalized = _DASHES_RE.sub("-", normalized)
    return _SPACES_RE.sub(" ", normalized)

def strip_bom(content: str) -> tuple[str, str]:
    if content.startswith(_BOM):
        return _BOM, content[1:]
    return "", content

def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )

def _count_occurrences(content: str, needle: str) -> int:
    if not needle:
        return 0

    count = 0
    start = 0
    while True:
        index = content.find(needle, start)
        if index == -1:
            return count
        count += 1
        start = index + len(needle)

def _edit_prefix(index: int, total: int) -> str:
    if total == 1:
        return "the exact text"
    return f"edits[{index}]"

def _not_found_message(index: int, total: int) -> str:
    target = _edit_prefix(index, total)
    return (
        f"Could not find {target} in content. "
        "The old text must match exactly including all whitespace and newlines, "
        "even after fuzzy normalization."
    )

def _ambiguous_message(index: int, total: int, occurrences: int) -> str:
    target = _edit_prefix(index, total)
    return (
        f"Found {occurrences} occurrences of {target} in content. "
        "Each old_text must be unique. Please provide more surrounding context."
    )

def apply_edits(base: str, edits: list[Edit]) -> AppliedEditsResult:
    """Apply edits against *base* while preserving BOM and line endings."""

    bom, content = strip_bom(base)
    line_ending = detect_line_ending(content)
    current = normalize_to_lf(content)
    base_content = current

    for edit_index, edit in enumerate(edits):
        old_text = normalize_to_lf(edit.old_text)
        new_text = normalize_to_lf(edit.new_text)
        if old_text == "":
            raise ValueError("old_text must not be empty.")

        match = fuzzy_find_text(current, old_text)
        if not match.found:
            raise EditNotFound(_not_found_message(edit_index, len(edits)))

        search_space = match.content_for_replacement
        if match.used_fuzzy_match:
            base_content = search_space
        needle = normalize_for_fuzzy_match(old_text) if match.used_fuzzy_match else old_text
        occurrences = _count_occurrences(search_space, needle)
        if occurrences > 1:
            raise EditAmbiguous(
                _ambiguous_message(edit_index, len(edits), occurrences)
            )

        current = (
            search_space[: match.index]
            + new_text
            + search_space[match.index + match.match_length :]
        )

    return AppliedEditsResult(
        base_content=base_content,
        new_content=bom + restore_line_endings(current, line_ending),
    )
