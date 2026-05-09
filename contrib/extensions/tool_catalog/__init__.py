"""Tool catalog extension package.

Mount ``contrib.extensions.tool_catalog.browse`` for read-only introspection,
``contrib.extensions.tool_catalog.mutate`` for self-modification, or this
package module as a compatibility shim that installs both.
"""

from __future__ import annotations

from typing import Any

from agentm.harness.extension import ExtensionAPI

from .browse import MANIFEST as BROWSE_MANIFEST
from .browse import install as install_browse
from .mutate import MANIFEST as MUTATE_MANIFEST
from .mutate import install as install_mutate


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    install_browse(api, config)
    install_mutate(api, config)


__all__ = [
    "BROWSE_MANIFEST",
    "MUTATE_MANIFEST",
    "install",
    "install_browse",
    "install_mutate",
]
