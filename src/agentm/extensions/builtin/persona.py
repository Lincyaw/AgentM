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

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib import truncate_text_tokens
from agentm.extensions import ExtensionManifest
from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI

_DEFAULT_FILES: Final = ("SOUL.md", "IDENTITY.md", "USER.md")

class PersonaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str | None = None
    files: list[str] | None = None
    max_tokens: int = Field(gt=0)
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
    # Leaf atom: reads and seeds via api.get_resource_writer().
    requires=(),
)

def _resolve_dir(cwd: str, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (Path(cwd) / path).resolve()

def _heading(filename: str) -> str:
    """`SOUL.md` -> `## Soul`; keeps the label human and source-traceable."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return f"## {stem.title()}" if stem else f"## {filename}"

async def _read_file(
    writer: Any,
    path: Path,
    max_tokens: int,
    model_name: str | None,
) -> str | None:
    """Return trimmed file text, or ``None`` when absent / empty / unreadable."""
    try:
        if not await writer.exists(str(path)):
            return None
        raw = await writer.read(str(path))
    except Exception as exc:
        logger.debug("persona: failed to read {}: {}", path, exc)
        return None
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    truncated = truncate_text_tokens(text, max_tokens, model=model_name)
    if not truncated.was_truncated:
        return text
    return (
        truncated.text
        + f"\n... ({truncated.truncated_tokens} more tokens truncated)"
    )

async def _build_block(
    writer: Any,
    base: Path,
    files: tuple[str, ...],
    max_tokens: int,
    model_name: str | None,
) -> str:
    sections: list[str] = []
    for name in files:
        body = await _read_file(writer, base / name, max_tokens, model_name)
        if body is None:
            continue
        sections.append(f"{_heading(name)}\n\n{body}")
    if not sections:
        return ""
    return (
        "# Persona\n\n"
        "The following defines your identity for this conversation. Stay "
        "consistent with this persona in tone, knowledge, and behavior "
        "throughout the session. Do not contradict or step outside it.\n\n"
        + "\n\n".join(sections)
    )

def _cwd_relative(path: Path, cwd: str) -> str:
    try:
        return str(path.resolve().relative_to(Path(cwd).resolve()))
    except ValueError:
        return str(path)

async def _seed_defaults(
    writer: Any,
    base: Path,
    cwd: str,
    files: tuple[str, ...],
    defaults: dict[str, str],
) -> None:
    for name in files:
        text = defaults.get(name)
        if not text:
            continue
        path = base / name
        try:
            if await writer.exists(str(path)):
                continue
            await writer.write(
                _cwd_relative(path, cwd),
                text.encode("utf-8"),
                rationale="persona: seed preset identity file",
            )
        except Exception as exc:  # noqa: BLE001
            # A preset identity file failed to seed — the agent will run
            # without it, so surface the failure rather than hiding it.
            logger.warning("persona: failed to seed identity file {}: {}", path, exc)
            continue

def install(api: ExtensionAPI, config: PersonaConfig) -> None:
    base = _resolve_dir(api.cwd, config.dir or ".")
    files = tuple(config.files) if config.files is not None else _DEFAULT_FILES
    max_tokens = config.max_tokens
    defaults = dict(config.defaults) if config.defaults is not None else {}
    writer = api.get_resource_writer()
    seeded = {"done": False}

    async def _before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if defaults and not seeded["done"]:
            seeded["done"] = True
            await _seed_defaults(writer, base, api.cwd, files, defaults)

        model_name = api.model.id if api.model is not None else None
        block = await _build_block(writer, base, files, max_tokens, model_name)
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
