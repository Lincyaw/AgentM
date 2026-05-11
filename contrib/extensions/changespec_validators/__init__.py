"""ChangeSpec per-kind validators (B-3) — mountable extension package.

Each sibling module validates a single ``ChangeSpec.kind`` and exposes
``validate(change_spec, cwd, target_scenario) -> dict``. Return shape:
``{"ok": bool, "error": str | None, "resolved_path": str | None}``.

This package is mountable via ``--extension contrib.extensions.changespec_validators``.
At install time it registers the per-kind callables on the ``changespec_validators``
service so that ``tool_propose_change`` can look them up through
``api.get_service(...)`` instead of reaching across the source tree by
filesystem path. New validator kinds plug in by mounting another extension
that augments the same service entry.

Validators live under ``contrib/`` (not ``src/agentm/``) because they
encode scenario-shape policy, not SDK mechanism.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from . import atom_source, manifest_extensions, manifest_field, system_prompt

SERVICE_NAME = "changespec_validators"


MANIFEST = ExtensionManifest(
    name="changespec_validators",
    description=(
        "Registers per-kind ChangeSpec validators on the "
        f"'{SERVICE_NAME}' service consumed by tool_propose_change."
    ),
    registers=(f"service:{SERVICE_NAME}",),
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    registry = api.get_service(SERVICE_NAME)
    if not isinstance(registry, dict):
        registry = {}
    registry.update(
        {
            "atom_source": atom_source.validate,
            "manifest_extensions": manifest_extensions.validate,
            "manifest_field": manifest_field.validate,
            "system_prompt": system_prompt.validate,
        }
    )
    api.set_service(SERVICE_NAME, registry)
