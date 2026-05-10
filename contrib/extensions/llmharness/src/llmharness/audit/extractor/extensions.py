"""§11 single-file extension: register the v3 extractor tool trio.

The adapter constructs a per-firing :class:`ExtractionState`, publishes
it via ``api.set_service("llmharness.extractor_state", state)``, and
mounts this module in the child session's extensions list. ``install``
reads the state out of the service registry and registers the three
closed-over tools (``register_event``, ``add_edge``,
``submit_extraction``) on the child kernel.

This file also exposes :func:`compose_extractor_extensions`, which
returns the ordered ``[(module, config), ...]`` list the adapter
mounts. Order: observability -> cards_tools -> THIS atom -> system_prompt.

§11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import. The state
service handoff is the only cross-firing channel.
"""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from .._compose import UNSET, compose_audit_extensions
from .prompt import EXTRACTOR_SYSTEM_PROMPT
from .state import ExtractionState
from .tools import build_extractor_tools

EXTRACTOR_STATE_SERVICE_KEY = "llmharness.extractor_state"
"""Service key the adapter uses to publish the per-firing ExtractionState."""

_EXTRACTOR_TOOLS_MODULE = "llmharness.audit.extractor.extensions"


MANIFEST = ExtensionManifest(
    name="extractor_tools",
    description=(
        "Register the v3 extractor tools (register_event, add_edge, "
        "submit_extraction) bound to the per-firing ExtractionState "
        "published by the adapter under "
        f"{EXTRACTOR_STATE_SERVICE_KEY!r}."
    ),
    registers=(
        "tool:register_event",
        "tool:add_edge",
        "tool:submit_extraction",
    ),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config  # no per-mount knobs in v3
    state = api.get_service(EXTRACTOR_STATE_SERVICE_KEY)
    if not isinstance(state, ExtractionState):
        raise RuntimeError(
            "extractor_tools.install: expected an ExtractionState at "
            f"service key {EXTRACTOR_STATE_SERVICE_KEY!r}; the adapter "
            "must publish one before mounting this extension."
        )
    for tool in build_extractor_tools(state):
        api.register_tool(tool)


def compose_extractor_extensions(
    *,
    prompt_override: str | None = None,
    cards_tools_config: dict[str, Any] | None = UNSET,
    observability_config: dict[str, Any] | None = UNSET,
) -> list[tuple[str, dict[str, Any]]]:
    """Default order: observability -> cards_tools -> extractor_tools -> system_prompt.

    Pass ``None`` for ``cards_tools_config`` / ``observability_config``
    to drop that extension; ``extractor_tools`` and ``system_prompt``
    always survive.
    """

    return compose_audit_extensions(
        submit_tool_module=_EXTRACTOR_TOOLS_MODULE,
        default_prompt=EXTRACTOR_SYSTEM_PROMPT,
        prompt_override=prompt_override,
        cards_tools_config=cards_tools_config,
        observability_config=observability_config,
    )


__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "MANIFEST",
    "compose_extractor_extensions",
    "install",
]
