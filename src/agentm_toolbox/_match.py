"""Fuzzy string matching for the edit tool.

Aligned with Claude Code's ``findActualString`` — three-level fallback:
1. Exact match
2. Quote-normalized match (curly → straight)
3. Whitespace-trimmed per-line match

Always returns the ACTUAL string from the file, not the search string.
Zero external dependencies.
"""

from __future__ import annotations

_QUOTE_MAP: dict[str, str] = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


def _normalize_quotes(s: str) -> str:
    for curly, straight in _QUOTE_MAP.items():
        s = s.replace(curly, straight)
    return s


def _strip_line_whitespace(s: str) -> str:
    return "\n".join(line.strip() for line in s.split("\n"))


def find_actual_string(file_content: str, search: str) -> str | None:
    """Find *search* in *file_content* with progressive fallbacks.

    Returns the ACTUAL matched substring from *file_content*, or ``None``.
    """
    if search in file_content:
        return search

    norm_search = _normalize_quotes(search)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx : idx + len(norm_search)]

    stripped_search = _strip_line_whitespace(search)
    stripped_file = _strip_line_whitespace(file_content)
    idx = stripped_file.find(stripped_search)
    if idx != -1:
        orig_lines = file_content.split("\n")
        prefix = stripped_file[:idx]
        start_line = prefix.count("\n")
        search_line_count = stripped_search.count("\n") + 1
        matched_lines = orig_lines[start_line : start_line + search_line_count]
        return "\n".join(matched_lines)

    return None
