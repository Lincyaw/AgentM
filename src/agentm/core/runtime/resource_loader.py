"""Resource discovery: skills, prompts, and project context files.

Implements §6 (ResourceLoader) of ``.claude/designs/extension-as-scenario.md``
and §3.4 (Resource Discovery) of ``.claude/designs/pluggable-architecture.md``.

The default loader walks ``cwd``, its ancestors, and ``$AGENTM_HOME`` to find:
- ``SKILL.md`` files inside ``skills/<name>/`` directories — YAML frontmatter
  declares ``name`` and ``description``; the rest is the skill body.
- Prompt templates under ``prompts/*.md`` — filename (sans ``.md``) is the
  template name; whole file content is the body.
- Project-level context files: ``AGENTS.md`` and ``CLAUDE.md``, walked from
  the topmost ancestor down to ``cwd`` so the closest file appears last
  (closest wins by convention when callers concatenate).

Embedded SDK callers who lack a filesystem can use ``InMemoryResourceLoader``.

Hard rule: this module imports only stdlib + shared core-lib helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.core.lib.user_config import agentm_home_dir


# --- Records ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill: name + description for prompt-time advertisement, plus
    the body to inject when the skill is invoked."""

    name: str
    description: str
    body: str
    source: str


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A reusable prompt template referenced by name."""

    name: str
    body: str
    source: str


@dataclass(frozen=True, slots=True)
class ContextFile:
    """An ``AGENTS.md`` / ``CLAUDE.md`` (or similar) file's body + path."""

    body: str
    source: str


# --- Protocol ---------------------------------------------------------------


@runtime_checkable
class ResourceLoader(Protocol):
    """The §3.4 boundary. Replace to swap discovery (DB, HTTP, in-memory)."""

    def get_skills(self) -> list[Skill]: ...
    def get_prompt_templates(self) -> list[PromptTemplate]: ...
    def get_context_files(self) -> list[ContextFile]: ...
    def reload(self) -> None: ...


# --- Frontmatter parser ---------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split a markdown file into ``(frontmatter, body)`` via shared parsing."""

    metadata, body = parse_frontmatter(text)
    return dict(metadata), body


# --- Default impl ----------------------------------------------------------


class DefaultResourceLoader:
    """Filesystem-backed loader. Caches results until ``reload()`` is called.

    Discovery rules (see module docstring):

    - Skills:
        - ``<agent_dir>/skills/<name>/SKILL.md``
        - ``<cwd>/.agentm/skills/<name>/SKILL.md``
    - Prompts:
        - ``<agent_dir>/prompts/*.md``
        - ``<cwd>/.agentm/prompts/*.md``
    - Context files:
        - ``<agent_dir>/AGENTS.md`` (if present, prepended)
        - ``AGENTS.md`` and ``CLAUDE.md`` walked from filesystem root down to
          ``cwd`` (ancestor-most first; ``cwd`` last).
    """

    def __init__(
        self,
        cwd: Path,
        agent_dir: Path | None = None,
        *,
        no_skills: bool = False,
        no_prompt_templates: bool = False,
    ) -> None:
        self._cwd = Path(cwd)
        self._agent_dir = (
            Path(agent_dir) if agent_dir is not None else agentm_home_dir()
        )
        self._no_skills = no_skills
        self._no_prompt_templates = no_prompt_templates
        self._skills: list[Skill] | None = None
        self._prompts: list[PromptTemplate] | None = None
        self._context: list[ContextFile] | None = None

    # --- Public API -------------------------------------------------------

    def get_skills(self) -> list[Skill]:
        if self._no_skills:
            return []
        if self._skills is None:
            self._skills = self._discover_skills()
        return list(self._skills)

    def get_prompt_templates(self) -> list[PromptTemplate]:
        if self._no_prompt_templates:
            return []
        if self._prompts is None:
            self._prompts = self._discover_prompts()
        return list(self._prompts)

    def get_context_files(self) -> list[ContextFile]:
        if self._context is None:
            self._context = self._discover_context_files()
        return list(self._context)

    def reload(self) -> None:
        self._skills = None
        self._prompts = None
        self._context = None

    # --- Discovery --------------------------------------------------------

    def _discover_skills(self) -> list[Skill]:
        roots = [
            self._agent_dir / "skills",
            self._cwd / ".agentm" / "skills",
        ]
        out: list[Skill] = []
        for root in roots:
            if not root.is_dir():
                continue
            for skill_dir in sorted(root.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue
                text = skill_md.read_text(encoding="utf-8")
                fm, body = _split_frontmatter(text)
                name = str(fm.get("name") or skill_dir.name)
                description = str(fm.get("description") or "")
                out.append(
                    Skill(
                        name=name,
                        description=description,
                        body=body,
                        source=str(skill_md),
                    )
                )
        return out

    def _discover_prompts(self) -> list[PromptTemplate]:
        roots = [
            self._agent_dir / "prompts",
            self._cwd / ".agentm" / "prompts",
        ]
        out: list[PromptTemplate] = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                if not path.is_file():
                    continue
                body = path.read_text(encoding="utf-8")
                out.append(
                    PromptTemplate(
                        name=path.stem,
                        body=body,
                        source=str(path),
                    )
                )
        return out

    def _discover_context_files(self) -> list[ContextFile]:
        out: list[ContextFile] = []

        # Agent-level (lowest precedence) first.
        agent_global = self._agent_dir / "AGENTS.md"
        if agent_global.is_file():
            out.append(
                ContextFile(
                    body=agent_global.read_text(encoding="utf-8"),
                    source=str(agent_global),
                )
            )

        # Build the chain from filesystem root down to cwd, so the most
        # specific file appears last.
        chain: list[Path] = []
        current = self._cwd.resolve()
        chain.append(current)
        for parent in current.parents:
            chain.append(parent)
        chain.reverse()  # root → cwd

        for directory in chain:
            for filename in ("AGENTS.md", "CLAUDE.md"):
                candidate = directory / filename
                if candidate.is_file():
                    out.append(
                        ContextFile(
                            body=candidate.read_text(encoding="utf-8"),
                            source=str(candidate),
                        )
                    )
        return out


# --- In-memory impl --------------------------------------------------------


class InMemoryResourceLoader:
    """Loader that returns the lists it was constructed with.

    Designed for tests and embedded-SDK use cases where the host has no
    filesystem (web app, notebook, sandbox).
    """

    def __init__(
        self,
        *,
        skills: list[Skill] | None = None,
        prompt_templates: list[PromptTemplate] | None = None,
        context_files: list[ContextFile] | None = None,
    ) -> None:
        self._skills = list(skills or [])
        self._prompts = list(prompt_templates or [])
        self._context = list(context_files or [])

    def get_skills(self) -> list[Skill]:
        return list(self._skills)

    def get_prompt_templates(self) -> list[PromptTemplate]:
        return list(self._prompts)

    def get_context_files(self) -> list[ContextFile]:
        return list(self._context)

    def reload(self) -> None:
        # No-op: the caller would have to swap in a new instance to "reload".
        return None


__all__ = [
    "ContextFile",
    "DefaultResourceLoader",
    "InMemoryResourceLoader",
    "PromptTemplate",
    "ResourceLoader",
    "Skill",
]
