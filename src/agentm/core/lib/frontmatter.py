"""Thin wrapper around ``python-frontmatter``.

Keeps the rest of AgentM on a tiny internal API surface so third-party parser
choices stay localized to one module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import frontmatter  # type: ignore[import-untyped]
from loguru import logger


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse leading frontmatter and return ``(metadata, body)``.

    On parse failure, fall back to the original text unchanged so callers can
    decide whether malformed metadata is fatal for their own use case.
    """

    try:
        post = frontmatter.loads(text)
    except Exception as exc:
        logger.debug("frontmatter: parse failed, returning raw text: {}", exc)
        return {}, text

    metadata = post.metadata
    if not isinstance(metadata, Mapping):
        return {}, cast(str, post.content)
    return dict(metadata), cast(str, post.content)
