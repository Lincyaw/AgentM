"""§11 single-file extension: register the v3.1 extractor's single tool.

The adapter constructs a per-firing :class:`ExtractionState`, hands it
to this module via ``config['state']`` at child-session-spawn time, and
``install`` registers the closed-over ``submit_events`` tool on the
child kernel. The fallback path uses
``api.get_service("llmharness.extractor_state")`` for tests that
pre-publish the state on the same session.

This file also exposes :func:`compose_extractor_extensions`, which
returns the ordered ``[(module, config), ...]`` list the adapter
mounts. Order: observability -> cards_tools -> THIS atom -> system_prompt.

§11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import. The state
config-payload handoff is the only cross-firing channel.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from .._atom_constants import (
    EXTRACTOR_STATE_SERVICE_KEY,
)
from .._atom_constants import (
    EXTRACTOR_TOOLS_MODULE as _EXTRACTOR_TOOLS_MODULE,
)
from .._compose import UNSET, compose_audit_extensions
from .prompt import EXTRACTOR_SYSTEM_PROMPT
from .state import ExtractionState
from .tools import build_extractor_tools

MANIFEST = ExtensionManifest(
    name="extractor_tools",
    description=(
        "Register the v3.1 extractor's single ``submit_events`` tool "
        "bound to the per-firing ExtractionState published by the "
        f"adapter under {EXTRACTOR_STATE_SERVICE_KEY!r}."
    ),
    registers=("tool:submit_events",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Two valid handoff channels for the per-firing ExtractionState:
    #   1. ``config["state"]`` — the adapter wires the freshly-built state
    #      into the child session's extensions list at spawn time. This is
    #      the primary path under v3: child sessions get a fresh
    #      service-registry, so a parent-side ``api.set_service`` does not
    #      cross the spawn boundary.
    #   2. ``api.get_service(EXTRACTOR_STATE_SERVICE_KEY)`` — fallback for
    #      callers that pre-publish the state on the same session this
    #      extension is being mounted onto (e.g. unit tests).
    state = config.get("state") if isinstance(config, dict) else None
    if not isinstance(state, ExtractionState):
        state = api.get_service(EXTRACTOR_STATE_SERVICE_KEY)
    if not isinstance(state, ExtractionState):
        raise RuntimeError(
            "extractor_tools.install: expected an ExtractionState via "
            f"config['state'] or service key {EXTRACTOR_STATE_SERVICE_KEY!r}; "
            "the adapter must supply one before mounting this extension."
        )
    for tool in build_extractor_tools(state):
        api.register_tool(tool)


def compose_extractor_extensions(
    *,
    base_prompt: str | None = None,
    cards_tools_config: dict[str, Any] | None = UNSET,
    observability_config: dict[str, Any] | None = UNSET,
) -> list[tuple[str, dict[str, Any]]]:
    """Default order: observability -> cards_tools -> extractor_tools -> system_prompt.

    ``base_prompt`` defaults to :data:`EXTRACTOR_SYSTEM_PROMPT` (the
    ``default`` variant). Pass an alternate framing — either resolved
    via :func:`audit.extractor.prompt.load_extractor_prompt` or any
    custom text — to A/B prompts.

    Pass ``None`` for ``cards_tools_config`` / ``observability_config``
    to drop that extension; ``extractor_tools`` and ``system_prompt``
    always survive.
    """
    framing = base_prompt if base_prompt is not None else EXTRACTOR_SYSTEM_PROMPT
    return compose_audit_extensions(
        submit_tool_module=_EXTRACTOR_TOOLS_MODULE,
        default_prompt=framing,
        prompt_override=None,
        cards_tools_config=cards_tools_config,
        observability_config=observability_config,
    )


__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "MANIFEST",
    "compose_extractor_extensions",
    "install",
]
