"""Extractor child-session extension-list composer.

Builds the ordered ``[(module, config), ...]`` list the adapter mounts
on each extractor firing. After the v19 atom split, the §11 atom lives
at :mod:`llmharness.audit.extractor.atom` (the merged tool atom that
mirrors :mod:`llmharness.audit.auditor.atom`); this module is just a
typed composer that points at it.

The atom alone — usable via ``agentm --extension
llmharness.audit.extractor.atom`` — registers the seven tools listed in
:data:`atom.EXTRACTOR_TOOL_NAMES`. The composer wires in the
observability / operations / system_prompt scaffolding that surrounds it
for an audit child.
"""

from __future__ import annotations

from typing import Any

from .._atom_constants import (
    EXTRACTOR_STATE_SERVICE_KEY,
)
from .._atom_constants import (
    EXTRACTOR_TOOLS_MODULE as _EXTRACTOR_TOOLS_MODULE,
)
from .._compose import UNSET, compose_audit_extensions
from .prompt import DEFAULT_PROMPT_NAME, load_extractor_prompt


def compose_extractor_extensions(
    *,
    base_prompt: str | None = None,
    observability_config: dict[str, Any] | None = UNSET,
) -> list[tuple[str, dict[str, Any]]]:
    """Default order: observability -> extractor_tools -> system_prompt.

    ``base_prompt`` defaults to the ``default`` variant loaded via
    :func:`load_extractor_prompt`. Pass an alternate framing — a name,
    a path resolved via that loader, or raw text — to A/B prompts.

    Pass ``None`` for ``observability_config`` to drop that extension;
    ``extractor_tools`` and ``system_prompt`` always survive.
    """
    framing = (
        base_prompt
        if base_prompt is not None
        else load_extractor_prompt(DEFAULT_PROMPT_NAME)
    )
    return compose_audit_extensions(
        submit_tool_module=_EXTRACTOR_TOOLS_MODULE,
        default_prompt=framing,
        prompt_override=None,
        observability_config=observability_config,
        submit_tool_config=None,
    )


__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "compose_extractor_extensions",
]
