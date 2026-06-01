"""Extension manifest data type — the atom-facing declaration record.

``ExtensionManifest`` is the module-level ``MANIFEST`` every built-in
extension exports. It is a pure frozen dataclass (stdlib-only deps) and
lives in the ABI surface so the recovery floor (``core/abi/`` + ``core/lib/``
+ ``core/runtime/`` + stdlib) can construct and pattern-match against it
without importing the autonomy-layer ``agentm.extensions`` package. See
``.claude/designs/self-modifiable-architecture.md`` §1 for the recovery-floor
invariant this keeps intact.

The autonomy-layer validator helpers that parse ``registers`` tags
(``VALID_REGISTER_KINDS`` / ``parse_register_tag``) stay in
``agentm.extensions`` — they are tooling, not ABI.

Pluggability hard rule: this module imports only stdlib + ABI siblings.
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
    ``{tool, event, command, provider, renderer, mutates}``. Examples:
    ``"tool:read"``, ``"event:tool_call"``, ``"provider:anthropic"``,
    ``"mutates:tool_catalog"``.
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

    # --- Self-modifiable architecture fields --------
    # See ``.claude/designs/self-modifiable-architecture.md`` §4 (versioned
    # API) and §7 (tier rationale).

    api_version: int = 1
    """Integer ABI the atom expects. Validated against
    ``core-manifest.yaml::extension_api.current`` plus its grace window
    (§11.4.9). An atom written for a future revision is rejected at load
    time rather than crashing on first event delivery."""

    affects: tuple[str, ...] = field(default_factory=tuple)
    """Metric surface this atom influences. Consumed by the catalog
    indexer; atoms may leave this empty when not yet measured. Tuple (not
    list) for the same hashable-value-type reasons as ``registers``."""

    tier: int = 1
    """Reload-cost classification. ``1`` = freely-editable (validator runs,
    reload happens). ``2`` = reviewable (reload is deferred to a
    ``propose_change`` decision). The canonical tier-2 list lives in
    ``core-manifest.yaml::reload.tier_2_atoms``; the validator's §11.4.10
    check warns on disagreement between this field and that list."""

    mountable_via_command: bool = False
    """Whether deployments may surface this atom as a ``/atom:install``
    slash command in chat gateways. Two gates total: the atom author opts
    in here, and the gateway operator must additionally list the atom in
    ``commands.atoms.allow``. Both required so neither party can ship a
    user-mountable atom by accident. Default ``False`` is fully backwards
    compatible.

    See ``.claude/designs/command-routing.md`` §8 for the runtime-overlay
    semantics — installs route through ``ExtensionAPI.install_atom``,
    which writes to ``<cwd>/.agentm/atoms/<name>.py``, so source scenario
    files are never touched."""

    provides_role: tuple[str, ...] = field(default_factory=tuple)
    """Singleton roles this atom fulfils in the harness wiring.

    Unlike ``registers`` (capability *tags*, many atoms may register the
    same kind), each role names a slot the harness expects exactly one
    atom to fill — e.g. ``"command_parser"``, ``"system_prompt_provider"``,
    ``"compaction_prompts"``, ``"sub_agent_runtime"``,
    ``"provider_inheritor"``. The discovery layer indexes atoms by role
    so the session factory can resolve "which atom is today's command
    parser?" without hard-coding atom names.

    Use the constants exported from :mod:`agentm.core.abi.roles` rather
    than raw strings. Empty tuple (default) means the atom claims no
    singleton role."""


__all__ = [
    "ExtensionManifest",
]
