"""Command discovery: builtin scan + markdown scan + skill scan + plugins.

The registry is built once at gateway start; commands added later
(new markdown file, new skill) are picked up on the next start. A
live-rescan path is plausible but not worth its complexity yet —
restart is cheap.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .atom_command import build_atom_commands
from .markdown_command import MarkdownPromptCommand
from .protocol import CommandHandler
from .skill_command import SkillCommand, walk_skill_dirs


@dataclass(frozen=True, slots=True)
class _Key:
    """Lookup key — None namespace and lowercase name."""

    namespace: str | None
    name: str


def _key(handler: CommandHandler) -> _Key:
    return _Key(namespace=handler.namespace, name=handler.name)


@dataclass(slots=True)
class CommandRegistry:
    """Snapshot of discovered handlers. Lookups are O(1); listings
    preserve discovery order so ``/help`` can group by source."""

    _by_key: dict[_Key, CommandHandler] = field(default_factory=dict)
    _ordered: list[CommandHandler] = field(default_factory=list)

    def register(self, handler: CommandHandler) -> None:
        """Add a handler. Re-registering the same key replaces the
        previous entry — the priority caller (`discover_commands`)
        feeds higher-priority sources first."""
        key = _key(handler)
        if key in self._by_key:
            return  # First-wins; higher priority sources go first.
        self._by_key[key] = handler
        self._ordered.append(handler)

    def lookup(
        self, *, namespace: str | None, name: str
    ) -> CommandHandler | None:
        return self._by_key.get(_Key(namespace=namespace, name=name))

    def all(self) -> list[CommandHandler]:
        return list(self._ordered)


# --- discovery --------------------------------------------------------


def discover_commands(
    cwd: str | Path,
    *,
    skill_paths: Iterable[str | Path] = (),
    include_default_skill_paths: bool = True,
    atom_commands_enabled: bool = False,
    atom_allow: Iterable[str] = (),
) -> CommandRegistry:
    """Build the registry. Priority (first-wins on collision):

    1. Builtin control commands (``commands/builtins/*.py``).
    2. Markdown prompt commands (``<cwd>/.agentm/commands/*.md``).
    3. Skill commands (skill directories).
    4. Atom commands (``/atom:install`` / ``/atom:uninstall`` /
       ``/atom:list``) — gated by ``atom_commands_enabled`` *and* a
       non-empty ``atom_allow`` list. Both required: the atom author's
       ``MANIFEST.mountable_via_command`` lives one layer deeper
       (filtered inside :func:`discover_mountable_atoms`), so this
       layer enforces deployment opt-in.
    """
    registry = CommandRegistry()
    _load_builtins(registry)
    _load_markdown(registry, Path(cwd))
    _load_skills(
        registry,
        Path(cwd),
        extra=skill_paths,
        include_defaults=include_default_skill_paths,
    )
    if atom_commands_enabled:
        _load_atom_commands(registry, allow=atom_allow)
    _load_entry_points(registry)
    return registry


def _load_atom_commands(
    registry: CommandRegistry, *, allow: Iterable[str]
) -> None:
    allow_set = frozenset(str(x) for x in allow)
    if not allow_set:
        logger.warning(
            "commands.atoms.enabled is true but allow list is empty; "
            "no /atom:* commands will be registered. Set "
            "commands.atoms.allow to a list of atom names (or [\"*\"])."
        )
        return
    for handler in build_atom_commands(allow=allow_set):
        registry.register(handler)


def _load_builtins(registry: CommandRegistry) -> None:
    from . import builtins as builtins_pkg

    for _finder, modname, ispkg in pkgutil.iter_modules(builtins_pkg.__path__):
        if ispkg:
            continue
        try:
            mod = importlib.import_module(
                f"{builtins_pkg.__name__}.{modname}"
            )
        except Exception:
            logger.exception(f"failed to import builtin command {modname!r}")
            continue
        handler = getattr(mod, "HANDLER", None)
        bundle = getattr(mod, "BUILTINS", None)
        if handler is not None:
            registry.register(handler)
        elif isinstance(bundle, list):
            for h in bundle:
                registry.register(h)
        else:
            logger.warning(f"builtin command module {modname} exports no HANDLER / BUILTINS")


def _load_markdown(registry: CommandRegistry, cwd: Path) -> None:
    cmd_dir = cwd / ".agentm" / "commands"
    if not cmd_dir.is_dir():
        return
    for path in sorted(cmd_dir.glob("*.md")):
        try:
            handler = MarkdownPromptCommand.from_path(path)
        except Exception:
            logger.exception(f"failed to load markdown command {path}")
            continue
        if handler is not None:
            registry.register(handler)


def _load_skills(
    registry: CommandRegistry,
    cwd: Path,
    *,
    extra: Iterable[str | Path],
    include_defaults: bool,
) -> None:
    seen: set[str] = set()
    for skill_dir, name in walk_skill_dirs(
        cwd, extra=extra, include_defaults=include_defaults
    ):
        if name in seen:
            continue
        seen.add(name)
        registry.register(SkillCommand.from_dir(skill_dir, name))


def _load_entry_points(registry: CommandRegistry) -> None:
    """External plugins register via the ``agentm.gateway.commands``
    entry-point group::

        [project.entry-points."agentm.gateway.commands"]
        my_command = "my_pkg.commands:MyCommand"

    The loaded value can be a handler instance or a callable that
    returns one.
    """
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group="agentm.gateway.commands")
    except TypeError:
        return
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:
            logger.exception(f"failed to load command plugin {ep.name!r}")
            continue
        candidate = obj() if callable(obj) and not hasattr(obj, "handle") else obj
        if not hasattr(candidate, "handle"):
            logger.warning(f"command plugin {ep.name!r} did not yield a CommandHandler")
            continue
        registry.register(candidate)
