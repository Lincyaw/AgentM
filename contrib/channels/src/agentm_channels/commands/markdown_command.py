"""Markdown prompt commands — Claude-Code-style ``.md`` files under
``<cwd>/.agentm/commands/``.

Frontmatter (optional)::

    ---
    name: standup
    summary: Generate today's standup
    ---
    Read the last 24h of commits and write a 3-bullet standup for
    $ARGUMENTS.

Without frontmatter, ``name`` defaults to the file stem and ``summary``
to the first non-empty line of the body.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .protocol import (
    CommandContext,
    CommandHandler,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


logger = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$")


@dataclass(slots=True)
class MarkdownPromptCommand:
    """One markdown file → one prompt command.

    Loading is up-front (registry build time) so a syntactically
    broken file is reported at startup, not when the user invokes it.
    """

    name: str
    summary: str
    body_template: str
    source_path: str
    namespace: str | None = None
    kind: CommandKind = "prompt"

    @classmethod
    def from_path(cls, path: Path) -> "MarkdownPromptCommand | None":
        raw = path.read_text(encoding="utf-8")
        meta, body = _split_frontmatter(raw)
        name = (meta.get("name") or path.stem).lower().strip()
        if not name:
            logger.warning("markdown command at %s has empty name; skipping", path)
            return None
        summary = (
            meta.get("summary")
            or _first_line(body)
            or "Markdown prompt command"
        ).strip()
        return cls(
            name=name,
            summary=summary,
            body_template=body.strip(),
            source_path=str(path),
        )

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del ctx  # markdown commands need no gateway state
        expanded = self.body_template.replace("$ARGUMENTS", inv.args)
        return CommandResult(expanded_prompt=expanded)


# --- frontmatter parser (no PyYAML dep) -------------------------------


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Parse the YAML-ish frontmatter most Claude-Code commands use.

    We support only ``key: value`` lines (no lists, no nested objects).
    That covers Claude Code's docs commands and our own use; richer
    schemas would be a leaky abstraction inside a command system.
    """
    match = _FRONTMATTER_RE.match(raw)
    if match is None:
        return {}, raw
    meta_block, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in meta_block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        meta[m.group(1).strip()] = m.group(2).strip().strip('"').strip("'")
    return meta, body


def _first_line(body: str) -> str:
    for line in body.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return ""


# Mypy structural check — confirms the dataclass satisfies the Protocol.
_assert_handler: CommandHandler = MarkdownPromptCommand(  # type: ignore[assignment]
    name="_", summary="_", body_template="", source_path=""
)
