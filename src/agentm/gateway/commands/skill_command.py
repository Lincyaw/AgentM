"""Skill-as-command — explicit user-driven skill activation.

The :class:`agentm.extensions.builtin.skill_loader` atom already lists
every SKILL.md in the system prompt so the LLM can self-select a skill.
This command is the **explicit override**: when the LLM has not
selected the right skill (or any skill at all), the user types
``/skill:<name> <args>`` to force the body into the current turn.

We do not reach into the ``skill_loader`` atom's loader directly — its
discovery pipeline is session-internal. The router walks the same
canonical skill directories (``<cwd>/.claude/skills/``,
``~/.claude/skills/``) the skill_loader atom walks. Discovery is
shallow: each ``<dir>/SKILL.md`` becomes one skill named after the
parent directory.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .protocol import (
    CommandContext,
    CommandHandler,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


_TURN_INJECTION = (
    "The user invoked the `{name}` skill explicitly. The skill content "
    "below applies to this turn only — read it, then act on the user's "
    "request that follows.\n\n"
    "--- SKILL: {name} ---\n"
    "{body}\n"
    "--- end skill ---\n\n"
    "User request: {args}"
)


@dataclass(slots=True)
class SkillCommand:
    """One skill directory → one ``/skill:<name>`` command.

    The body is read at handle-time, not at registry-build-time, so
    edits to a SKILL.md take effect on the next invocation without
    restarting the gateway. Trade-off: a broken SKILL.md (unreadable
    file) surfaces only when the user tries to use it. Acceptable —
    skill bodies are user-authored prose, and a startup-time failure
    that blocks the whole gateway over a typo is the worse failure
    mode.
    """

    name: str
    namespace: str | None = "skill"
    summary: str = "User-activated skill"
    kind: CommandKind = "prompt"
    source_dir: str = ""

    @classmethod
    def from_dir(cls, skill_dir: Path, name: str) -> "SkillCommand":
        summary = _peek_description(skill_dir / "SKILL.md") or "User-activated skill"
        return cls(
            name=name,
            summary=summary,
            source_dir=str(skill_dir),
        )

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del ctx
        skill_md = Path(self.source_dir) / "SKILL.md"
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"skill {self.name}: cannot read {skill_md}: {exc}")
            return CommandResult(
                expanded_prompt=(
                    f"The user invoked /skill:{self.name} but the skill body "
                    f"could not be read. Apologise briefly and proceed with "
                    f"the user's request: {inv.args}"
                )
            )
        body = _strip_frontmatter(raw).strip()
        args = inv.args.strip() or "(no additional instruction)"
        expanded = _TURN_INJECTION.format(name=self.name, body=body, args=args)
        return CommandResult(expanded_prompt=expanded)


# --- discovery helpers ------------------------------------------------


def walk_skill_dirs(
    cwd: Path,
    *,
    extra: Iterable[str | Path] = (),
    include_defaults: bool = True,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(skill_dir, name)`` for each ``<dir>/SKILL.md`` found.

    Order: configured extras first, then ``<cwd>/.claude/skills``,
    then ``~/.claude/skills``. The registry dedups by name so the
    extras win on collision.
    """
    seen_dirs: set[Path] = set()
    sources: list[Path] = [Path(p) for p in extra]
    if include_defaults:
        sources.append(cwd / ".claude" / "skills")
        sources.append(Path.home() / ".claude" / "skills")
    for root in sources:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen_dirs or not resolved.is_dir():
            continue
        seen_dirs.add(resolved)
        for entry in sorted(resolved.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            yield entry, entry.name.lower()


def _peek_description(skill_md: Path) -> str | None:
    """Best-effort: read the ``description`` line from frontmatter so
    ``/help`` shows something useful next to each skill."""
    try:
        with skill_md.open(encoding="utf-8") as fh:
            head = fh.read(2048)
    except OSError:
        return None
    if not head.startswith("---"):
        return None
    for line in head.splitlines()[1:]:
        if line.strip() == "---":
            break
        if line.lower().startswith("description:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def _strip_frontmatter(raw: str) -> str:
    if not raw.startswith("---"):
        return raw
    parts = raw.split("\n---\n", 1)
    if len(parts) == 2:
        return parts[1]
    return raw


# Structural Protocol assertion.
_assert_handler: CommandHandler = SkillCommand(name="_")  # type: ignore[assignment]
