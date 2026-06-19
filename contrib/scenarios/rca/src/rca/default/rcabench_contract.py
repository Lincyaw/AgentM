"""Compatibility shim for manifests that still mount rcabench_contract."""

from __future__ import annotations

from typing import Final

from agentm.extensions import ExtensionManifest
from rca.default.fpg_contract import MANIFEST as _FPG_MANIFEST
from rca.default.fpg_contract import install


MANIFEST = ExtensionManifest(
    name="rcabench_contract",
    description=(
        "Compatibility shim: inject the fpg ModelRCAOutput contract for "
        "legacy manifests that still mount rcabench_contract."
    ),
    registers=_FPG_MANIFEST.registers,
    config_schema=_FPG_MANIFEST.config_schema,
    tier=_FPG_MANIFEST.tier,
)

__all__: Final = ["MANIFEST", "install"]
