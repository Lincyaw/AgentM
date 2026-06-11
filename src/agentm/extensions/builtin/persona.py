"""Builtin ``persona`` atom.

File-driven identity layer for conversational scenarios. Reads a small,
ordered set of user-authored markdown files from the workspace (by
default ``SOUL.md`` -- voice/personality, ``IDENTITY.md`` -- who the bot
is, ``USER.md`` -- who it is talking to) and prepends their contents to
the system prompt. The agent's character is then edited by changing
files on disk, not by rewriting scenario YAML.

Sibling to the ``memory`` atom one axis over: ``memory`` carries *facts*
that evolve at runtime; ``persona`` carries the *stable identity* the
operator curates. Both land in the system prompt (the cache-stable
prefix), so per-turn KV/prefix caching is preserved -- nothing here
touches the volatile message tail. (Note the contribution mechanism
differs: see the ``before_agent_start`` handler below for why this atom
*returns* its replacement rather than only mutating ``event.system``.)

Missing files are skipped silently: a workspace that ships none of the
listed files simply gets no persona block, so the same scenario works
for a bare sandbox and a fully dressed assistant home.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI

_DEFAULT_FILES: Final = ("SOUL.md", "IDENTITY.md", "USER.md")
_DEFAULT_MAX_CHARS: Final = 12000

class PersonaConfig(BaseModel):
    dir: str | None = None
    files: list[str] | None = None
    max_chars: int = _DEFAULT_MAX_CHARS
    defaults: dict[str, str] | None = None

MANIFEST = ExtensionManifest(
    name="persona",
    description=(
        "Injects file-driven identity (SOUL.md / IDENTITY.md / USER.md by "
        "default) from the workspace into the system prompt. Edit the files "
        "to change the agent's character — no scenario code change needed."
    ),
    registers=("event:before_agent_start",),
    config_schema=PersonaConfig,
    # Leaf atom: reads via api.get_operations().file, seeds via
    # api.get_resource_writer(); no atom-to-atom dependency.
    requires=(),
)

def _resolve_dir(cwd: str, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (Path(cwd) / path).resolve()

def _heading(filename: str) -> str:
    """`SOUL.md` -> `## Soul`; keeps the label human and source-traceable."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return f"## {stem.title()}" if stem else f"## {filename}"

async def _read_file(file_ops: Any, path: Path, max_chars: int) -> str | None:
    """Return trimmed file text, or ``None`` when absent / empty / unreadable."""
    try:
        if not await file_ops.access(str(path)):
            return None
        raw = await file_ops.read_file(str(path))
    except Exception:
        return None
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    if len(text) > max_chars:
        dropped = len(text) - max_chars
        text = text[:max_chars] + f"\n... ({dropped} more chars truncated)"
    return text

async def _build_block(
    file_ops: Any, base: Path, files: tuple[str, ...], max_chars: int
) -> str:
    sections: list[str] = []
    for name in files:
        body = await _read_file(file_ops, base / name, max_chars)
        if body is None:
            continue
        sections.append(f"{_heading(name)}\n\n{body}")
    if not sections:
        return ""
    return "# Persona\n\n" + "\n\n".join(sections)

def _cwd_relative(path: Path, cwd: str) -> str:
    try:
        return str(path.resolve().relative_to(Path(cwd).resolve()))
    except ValueError:
        return str(path)

async def _seed_defaults(
    file_ops: Any,
    writer: Any,
    base: Path,
    cwd: str,
    files: tuple[str, ...],
    defaults: dict[str, str],
) -> None:
    """Write preset content for any listed file that is absent on disk.

    Idempotent and gated on absence: the operator's (or the agent's own)
    edits are never clobbered, and a deliberately removed file is only
    re-seeded if it ships a default. Best-effort — a write failure leaves
    that file unseeded rather than aborting startup.
    """
    for name in files:
        text = defaults.get(name)
        if not text:
            continue
        path = base / name
        try:
            if await file_ops.access(str(path)):
                continue
            await writer.write(
                _cwd_relative(path, cwd),
                text.encode("utf-8"),
                rationale="persona: seed preset identity file",
            )
        except Exception:
            continue

def install(api: ExtensionAPI, config: PersonaConfig) -> None:
    base = _resolve_dir(api.cwd, config.dir or ".")
    files = tuple(config.files) if config.files is not None else _DEFAULT_FILES
    max_chars = config.max_chars
    defaults = dict(config.defaults) if config.defaults is not None else {}
    file_ops = api.get_operations().file
    writer = api.get_resource_writer() if defaults else None
    seeded = {"done": False}

    async def _before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        # Seed preset files once, before composing — so a fresh workspace
        # has an identity from the very first reply. The agent may then
        # edit them with its own tools; the next turn re-reads from disk,
        # which is how persona changes "reload" without a restart.
        if writer is not None and not seeded["done"]:
            seeded["done"] = True
            await _seed_defaults(file_ops, writer, base, api.cwd, files, defaults)

        block = await _build_block(file_ops, base, files, max_chars)
        if not block:
            return None
        current = str(event.system or "")
        updated = f"{block}\n\n{current}" if current else block
        # Do BOTH, on purpose. The kernel's ``collect_system_replacement``
        # reads handler *returns* (not ``event.system``), so returning is
        # what guarantees this contribution survives when persona is the
        # last returning handler. The in-place mutation is still needed so
        # that any handler firing *after* persona sees it in their
        # ``current`` and folds it into their own return — without it, a
        # later returning handler would silently drop persona. Together
        # they make the result order-independent.
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, _before_agent_start)
