"""Atom-as-command: ``/atom:install``, ``/atom:uninstall``, ``/atom:list``.

Mounts existing AgentM atoms into the live ``AgentSession`` for this
chat. Two gates required, neither sufficient on its own:

1. Atom author: ``MANIFEST.mountable_via_command = True`` in the atom
   source. The author has signed off on "this atom is safe to mount
   mid-session from chat" — no network setup, no constitution writes,
   no irreversible state.
2. Gateway operator: name listed in YAML ``commands.atoms.allow``.
   Deployment policy. Default empty.

Persistence is automatic: :meth:`ExtensionAPI.install_atom` writes the
atom source to ``<cwd>/.agentm/atoms/<name>.py`` (via
``ResourceWriter`` → git commit), and the user-atom auto-discovery on
next session creation re-mounts it. So source scenario files under
``contrib/scenarios/`` and the SDK ``builtin/`` tree stay untouched —
the "runtime overlay" the design doc describes is exactly this SDK
convention.

Note on session lookup: the gateway must have an active route for the
chat (i.e. the user has sent at least one regular message) before
``/atom:install`` will work. That's because mounting requires a live
``ExtensionAPI``, which only exists after ``AgentSession.create``.
``/atom:list`` returns the "would-be-allowed" set when no session
exists yet, so users can preview what is available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..bus import OutboundMessage
from .protocol import (
    CommandContext,
    CommandHandler,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


logger = logging.getLogger(__name__)


# --- discovery -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MountableAtom:
    name: str
    summary: str
    module_path: str
    source_path: str


def discover_mountable_atoms(
    *, allow: frozenset[str]
) -> list[_MountableAtom]:
    """Return atoms eligible to be surfaced as ``/atom:*`` commands.

    An atom is eligible iff (a) it lives in the AgentM catalog with
    ``MANIFEST.mountable_via_command = True`` and (b) its name appears
    in ``allow``. ``"*"`` in ``allow`` means "every mountable-opted
    atom" — useful for development but explicit-list is safer for
    production.

    Discovery walks :func:`agentm.extensions.discover.discover_builtin`
    plus :func:`discover_contrib_atoms`. User atoms under
    ``<cwd>/.agentm/atoms/`` are intentionally excluded — those are
    already installed; surfacing them as ``/atom:install`` would
    confuse users about state.
    """
    try:
        from agentm.extensions.discover import (
            discover_builtin,
            discover_contrib_atoms,
        )
    except ImportError:  # pragma: no cover — SDK missing in tests
        logger.warning("agentm.extensions.discover unavailable; no atoms surfaced")
        return []

    catalog: dict[str, _MountableAtom] = {}
    for entry in (*discover_builtin().values(), *discover_contrib_atoms().values()):
        manifest = entry.manifest
        if not getattr(manifest, "mountable_via_command", False):
            continue
        if "*" not in allow and entry.name not in allow:
            continue
        module_file = getattr(entry.module, "__file__", None)
        if module_file is None:
            continue
        catalog[entry.name] = _MountableAtom(
            name=entry.name,
            summary=manifest.description,
            module_path=entry.module_path,
            source_path=str(module_file),
        )
    return sorted(catalog.values(), key=lambda a: a.name)


# --- shared base -----------------------------------------------------


@dataclass(slots=True)
class _AtomVerbBase:
    """Common fields for the three /atom:* commands."""

    name: str
    summary: str
    allow: frozenset[str] = field(default_factory=frozenset)
    namespace: str | None = "atom"
    kind: CommandKind = "control"

    def _reply(self, ctx: CommandContext, text: str) -> CommandResult:
        return CommandResult(
            outbound=[
                OutboundMessage(
                    channel=ctx.channel,
                    chat_id=ctx.chat_id,
                    content=text,
                )
            ]
        )

    def _resolve_api(self, ctx: CommandContext) -> Any | None:
        get_api = getattr(ctx, "get_extension_api", None)
        if get_api is None:
            return None
        try:
            return get_api()
        except Exception:
            logger.exception("get_extension_api raised")
            return None


# --- /atom:install ---------------------------------------------------


@dataclass(slots=True)
class AtomInstallCommand(_AtomVerbBase):
    name: str = "install"
    summary: str = "Mount an atom into this chat's session"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        parts = inv.args.split(maxsplit=1)
        if not parts:
            return self._reply(
                ctx,
                "Usage: `/atom:install <name> [config-json]`. "
                "See `/atom:list` for what's available.",
            )
        atom_name = parts[0].strip()
        config_blob = parts[1].strip() if len(parts) > 1 else ""

        atoms = {a.name: a for a in discover_mountable_atoms(allow=self.allow)}
        if atom_name not in atoms:
            return self._reply(
                ctx,
                f"`{atom_name}` is not in the mountable-atom allow list. "
                f"Run `/atom:list` to see what's available.",
            )

        atom_config: dict[str, Any] | None = None
        if config_blob:
            try:
                parsed = json.loads(config_blob)
            except json.JSONDecodeError as exc:
                return self._reply(
                    ctx, f"config json invalid: {exc}"
                )
            if not isinstance(parsed, dict):
                return self._reply(
                    ctx,
                    "config must be a JSON object (got "
                    f"{type(parsed).__name__}).",
                )
            atom_config = parsed

        api = self._resolve_api(ctx)
        if api is None:
            return self._reply(
                ctx,
                "No live session for this chat yet. Send a regular "
                "message first, then run `/atom:install`.",
            )

        try:
            source = Path(atoms[atom_name].source_path).read_text(encoding="utf-8")
        except OSError as exc:
            return self._reply(ctx, f"could not read atom source: {exc}")

        rationale = (
            f"User {ctx.sender_id} invoked /atom:install {atom_name} "
            f"from {ctx.channel}:{ctx.chat_id}."
        )
        try:
            result = api.install_atom(
                name=atom_name,
                source=source,
                target_path=None,                  # → <cwd>/.agentm/atoms/<name>.py
                config=atom_config,
                rationale=rationale,
                agent_initiated=False,             # user-initiated, not the LLM
            )
        except Exception as exc:
            logger.exception("install_atom raised")
            return self._reply(ctx, f"install_atom raised: {exc}")
        if not result.ok:
            return self._reply(
                ctx,
                f"Install rejected: {result.error or 'unknown error'}",
            )
        target = result.target_path or "(SDK default)"
        return self._reply(
            ctx,
            f"✅ Installed `{atom_name}` at `{target}`. "
            "Active now and persists across restarts.",
        )


# --- /atom:uninstall -------------------------------------------------


@dataclass(slots=True)
class AtomUninstallCommand(_AtomVerbBase):
    name: str = "uninstall"
    summary: str = "Unload an atom from this chat's session"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        atom_name = inv.args.strip()
        if not atom_name:
            return self._reply(ctx, "Usage: `/atom:uninstall <name>`.")
        api = self._resolve_api(ctx)
        if api is None:
            return self._reply(
                ctx,
                "No live session — nothing to unload. Send a message "
                "first.",
            )
        try:
            result = api.unload_atom(
                name=atom_name,
                agent_initiated=False,
                rationale=(
                    f"User {ctx.sender_id} invoked /atom:uninstall "
                    f"{atom_name} from {ctx.channel}:{ctx.chat_id}."
                ),
            )
        except Exception as exc:
            logger.exception("unload_atom raised")
            return self._reply(ctx, f"unload_atom raised: {exc}")
        if not result.ok:
            return self._reply(
                ctx, f"Unload rejected: {result.error or 'unknown error'}"
            )
        return self._reply(
            ctx,
            f"🗑 Unloaded `{atom_name}` from the live session. "
            "The on-disk source under `.agentm/atoms/` is untouched; "
            "re-`/atom:install` to mount it again.",
        )


# --- /atom:list ------------------------------------------------------


@dataclass(slots=True)
class AtomListCommand(_AtomVerbBase):
    name: str = "list"
    summary: str = "Show mountable + currently-installed atoms"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        mountable = discover_mountable_atoms(allow=self.allow)
        api = self._resolve_api(ctx)
        live: list[str] = []
        if api is not None:
            try:
                live = sorted(a.name for a in api.list_atoms())
            except Exception:
                logger.exception("list_atoms raised")
        lines: list[str] = ["**Atoms**"]
        if live:
            lines.append("")
            lines.append("_currently loaded_")
            lines.extend(f"  - {n}" for n in live)
        else:
            lines.append("_(no live session for this chat yet)_")
        lines.append("")
        if mountable:
            lines.append("_mountable via `/atom:install <name>`_")
            for atom in mountable:
                marker = " *(loaded)*" if atom.name in live else ""
                lines.append(f"  - `{atom.name}`{marker} — {atom.summary}")
        elif not self.allow:
            lines.append(
                "_no mountable atoms — set `commands.atoms.allow` in the "
                "gateway config to a list of atom names (or `[\"*\"]`)._"
            )
        else:
            lines.append(
                "_no atom in the catalog has opted into "
                "`MANIFEST.mountable_via_command=True` yet. The allow "
                "list is configured, but every atom must also opt in "
                "at the source._"
            )
        return self._reply(ctx, "\n".join(lines))


# --- factory ---------------------------------------------------------


def build_atom_commands(
    *, allow: frozenset[str]
) -> list[CommandHandler]:
    """Build the three verb commands sharing one ``allow`` set."""
    return [
        AtomInstallCommand(allow=allow),
        AtomUninstallCommand(allow=allow),
        AtomListCommand(allow=allow),
    ]


__all__ = [
    "AtomInstallCommand",
    "AtomListCommand",
    "AtomUninstallCommand",
    "build_atom_commands",
    "discover_mountable_atoms",
]
