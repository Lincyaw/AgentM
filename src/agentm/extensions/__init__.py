"""Public surface of the AgentM extension catalog.

Implements §11 (Single-File, Self-Contained Extension Contract) of
``.claude/designs/extension-as-scenario.md``. The intent is to make
extensions a *mechanically checkable* unit so that future agents can edit
their own behavior without breaking the harness.

Hard rules (full text in design §11):

1. One built-in extension = one file under ``extensions/builtin/<name>.py``.
2. Module-level ``MANIFEST: ExtensionManifest`` declares what the extension
   is and what it accepts.
3. ``install(api, config)`` is the only callable surface.
4. Imports are restricted to a fixed allow-list (kernel, ops, harness
   contract modules, ``agentm.extensions``).

Authors should depend only on what this package re-exports — anything
under ``agentm.extensions.discover`` / ``agentm.extensions.validate`` is
tooling and not part of the extension-author surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    """Module-level declaration every built-in extension exports.

    Read by ``discover.discover_builtin`` to populate the catalog and by
    ``validate.validate_builtin`` to enforce the §11 contract.

    Tag format for ``registers``: ``<kind>:<id>`` where ``<kind>`` ∈
    ``{tool, event, command, provider, renderer}``. Examples:
    ``"tool:read"``, ``"event:tool_call"``, ``"provider:anthropic"``.
    """

    name: str
    """Unique identifier; must equal the source-file stem (``permission`` ↔
    ``permission.py``). Lowercase ASCII + underscores."""

    description: str
    """One-line summary suitable for catalog listings."""

    registers: tuple[str, ...]
    """Capability tags this extension exposes. See class docstring for the
    tag grammar. Empty tuple is allowed for pure observers."""

    config_schema: dict[str, Any] | None = None
    """JSON-Schema describing the ``config`` dict ``install`` accepts.
    ``None`` means the extension takes no config. The validator rejects
    extensions whose schema is syntactically malformed."""

    requires: tuple[str, ...] = field(default_factory=tuple)
    """Names of other built-in extensions that MUST appear earlier in the
    load order. The runner enforces this and fails fast otherwise."""

    conflicts: tuple[str, ...] = field(default_factory=tuple)
    """Names of other built-in extensions that MUST NOT be loaded in the
    same session. The runner enforces this."""


# --- Tag parsing helper (used by validator + future tooling) ----------------

VALID_REGISTER_KINDS = frozenset(
    {"tool", "event", "command", "provider", "renderer"}
)


def parse_register_tag(tag: str) -> tuple[str, str]:
    """Split a ``MANIFEST.registers`` tag into ``(kind, id)``.

    Raises ``ValueError`` on malformed input — exposed publicly so tooling
    (validator, scenario authoring tools) emits one consistent error message.
    """

    kind, sep, ident = tag.partition(":")
    if not sep or not kind or not ident:
        raise ValueError(
            f"register tag {tag!r} must be of the form '<kind>:<id>'"
        )
    if kind not in VALID_REGISTER_KINDS:
        raise ValueError(
            f"register tag kind {kind!r} not in {sorted(VALID_REGISTER_KINDS)}"
        )
    return kind, ident


__all__ = [
    "ExtensionManifest",
    "VALID_REGISTER_KINDS",
    "parse_register_tag",
]
