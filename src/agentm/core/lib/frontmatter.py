"""Parse YAML frontmatter from Markdown text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


import frontmatter as _fm  # type: ignore[import-untyped]


def parse_frontmatter(
    text: str, *, strict: bool = False
) -> tuple[dict[str, object], str]:
    """Return ``(metadata, body)`` from text with optional YAML frontmatter.

    When ``strict=False`` (default), parse failures return ``({}, text)``.
    When ``strict=True``, parse failures raise ``ValueError``.
    """
    try:
        post = _fm.loads(text)
    except Exception as exc:
        if strict:
            raise ValueError("invalid frontmatter") from exc
        return {}, text
    metadata = post.metadata
    if not isinstance(metadata, Mapping):
        if strict:
            raise ValueError("frontmatter metadata must be a mapping")
        return {}, cast(str, post.content)
    return dict(metadata), cast(str, post.content)
