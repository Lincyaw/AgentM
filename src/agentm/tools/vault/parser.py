"""Pure functions for parsing and manipulating Markdown files with YAML frontmatter."""

from __future__ import annotations

import re

import yaml


def parse_note(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body.

    Returns (frontmatter_dict, body_string).
    Missing frontmatter yields an empty dict.
    """
    if not content.startswith("---\n"):
        return {}, content

    end_idx = content.find("\n---\n", 4)
    if end_idx == -1:
        return {}, content

    raw_yaml = content[4:end_idx]
    fm = yaml.safe_load(raw_yaml)
    if not isinstance(fm, dict):
        return {}, content

    body = content[end_idx + 5:]  # skip "\n---\n"
    return fm, body


def serialize_note(frontmatter: dict, body: str) -> str:
    """Render frontmatter dict + body into a Markdown string with YAML fences."""
    if not frontmatter:
        return body

    raw = yaml.safe_dump(frontmatter, default_flow_style=False, allow_unicode=True)
    return f"---\n{raw}---\n{body}"


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def extract_wikilinks(text: str) -> list[str]:
    """Extract deduplicated wikilink targets from *text*.

    Strips a trailing ``.md`` extension if present.
    """
    seen: dict[str, None] = {}
    for match in _WIKILINK_RE.finditer(text):
        target = match.group(1)
        if target.endswith(".md"):
            target = target[:-3]
        seen.setdefault(target, None)
    return list(seen)


_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)


def extract_title(body: str) -> str:
    """Return the text of the first ``# `` heading, or empty string."""
    m = _TITLE_RE.search(body)
    return m.group(1) if m else ""


def _heading_level(line: str) -> int:
    """Return heading level (1-6) or 0 if not a heading."""
    stripped = line
    count = 0
    for ch in stripped:
        if ch == "#":
            count += 1
        else:
            break
    if count > 0 and len(stripped) > count and stripped[count] == " ":
        return count
    return 0


def find_section(body: str, heading: str) -> tuple[int, int] | None:
    """Find char offsets ``(start, end)`` of *heading*'s section.

    The section spans from the heading line through all content up to (but not
    including) the next heading at the same or higher level, or end-of-file.
    """
    target_level = _heading_level(heading)
    if target_level == 0:
        return None

    # Find the heading line start offset
    lines = body.split("\n")
    offset = 0
    section_start: int | None = None

    for line in lines:
        if section_start is None:
            if line == heading:
                section_start = offset
        else:
            lvl = _heading_level(line)
            if lvl > 0 and lvl <= target_level:
                return section_start, offset
        offset += len(line) + 1  # +1 for newline

    if section_start is not None:
        return section_start, len(body)
    return None


def replace_section(body: str, heading: str, new_content: str) -> str:
    """Replace the content under *heading*, preserving the heading line itself."""
    span = find_section(body, heading)
    if span is None:
        msg = f"Heading not found: {heading!r}"
        raise ValueError(msg)

    start, end = span
    heading_line = heading + "\n"
    return body[:start] + heading_line + new_content + body[end:]


def append_to_section(body: str, heading: str, content: str) -> str:
    """Append *content* at the end of *heading*'s section."""
    span = find_section(body, heading)
    if span is None:
        msg = f"Heading not found: {heading!r}"
        raise ValueError(msg)

    _, end = span
    return body[:end] + content + body[end:]


def replace_string(body: str, old: str, new: str) -> str:
    """Replace *old* with *new* in *body* (exact, single occurrence).

    Raises ``ValueError`` if *old* is not found or appears more than once.
    """
    count = body.count(old)
    if count == 0:
        msg = f"Target string not found: {old!r}"
        raise ValueError(msg)
    if count > 1:
        msg = f"Target string is ambiguous ({count} occurrences): {old!r}"
        raise ValueError(msg)
    return body.replace(old, new, 1)
