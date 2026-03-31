"""YAML frontmatter parsing and serialization for Markdown files.

Shared utility used by both agent_loader (config layer) and vault parser
(tools layer) to avoid a config → tools dependency.
"""

from __future__ import annotations

from typing import Any

import yaml


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from body.

    Returns ``(frontmatter_dict, body_string)``.
    Missing or invalid frontmatter yields an empty dict and the full content.
    """
    content = content.replace("\r\n", "\n")
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


def serialize_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """Render frontmatter dict + body into a Markdown string with YAML fences."""
    if not frontmatter:
        return body

    raw = yaml.safe_dump(frontmatter, default_flow_style=False, allow_unicode=True)
    return f"---\n{raw}---\n{body}"
