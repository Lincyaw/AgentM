"""Public surface of the AgentM extension catalog.

Implements (Single-File, Self-Contained Extension Contract) of
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

# ``ExtensionManifest`` now lives in the ABI surface
# (:mod:`agentm.core.abi.manifest`) so the recovery floor — ``core/abi/`` +
# ``core/lib/`` + ``core/runtime/`` + stdlib — can construct and pattern-match
# against it without importing this autonomy-layer package. See
# ``.claude/designs/self-modifiable-architecture.md`` §1. It is re-exported
# here for backward compatibility: the ~100 atom files that do
# ``from agentm.extensions import ExtensionManifest`` keep working unchanged,
# and ``isinstance(MANIFEST, ExtensionManifest)`` stays a single class object.
from agentm.core.abi import ChannelEffects as ChannelEffects
from agentm.core.abi import ExtensionManifest as ExtensionManifest

# --- Tag parsing helper (used by validator + future tooling) ----------------

VALID_REGISTER_KINDS = frozenset(
    {"tool", "event", "command", "provider", "renderer", "mutates"}
)


def parse_register_tag(tag: str) -> tuple[str, str]:
    """Split a ``MANIFEST.registers`` tag into ``(kind, id)``.

    Raises ``ValueError`` on malformed input — exposed publicly so tooling
    (validator, scenario authoring tools) emits one consistent error message.
    """

    kind, sep, ident = tag.partition(":")
    if not sep or not kind or not ident:
        raise ValueError(f"register tag {tag!r} must be of the form '<kind>:<id>'")
    if kind not in VALID_REGISTER_KINDS:
        raise ValueError(
            f"register tag kind {kind!r} not in {sorted(VALID_REGISTER_KINDS)}"
        )
    return kind, ident


__all__ = [
    "ChannelEffects",
    "ExtensionManifest",
    "VALID_REGISTER_KINDS",
    "parse_register_tag",
]
