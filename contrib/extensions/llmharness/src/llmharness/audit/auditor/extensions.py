"""Phase 2 auditor child-session ``extensions`` list."""

from __future__ import annotations

from typing import Any

from .._compose import UNSET, compose_audit_extensions
from .prompt import AUDITOR_SYSTEM_PROMPT

_SUBMIT_TOOL_MODULE = "llmharness.audit.auditor.submit_tool"
_GET_TURN_TOOL_MODULE = "llmharness.audit.auditor.get_turn_tool"


def compose_auditor_extensions(
    *,
    prompt_override: str | None = None,
    cards_tools_config: dict[str, Any] | None = UNSET,
    observability_config: dict[str, Any] | None = UNSET,
    trajectory_snapshot: list[dict[str, Any]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Default order: observability → cards_tools → submit_tool → system_prompt.

    Pass ``None`` for ``cards_tools_config`` / ``observability_config`` to drop
    that extension; ``submit_tool`` and ``system_prompt`` always survive.

    When ``trajectory_snapshot`` is provided (non-None), the ``get_turn`` tool
    is appended to the list so the auditor child can drill back into individual
    raw turns on demand.  When ``trajectory_snapshot=None`` (the default), the
    tool is not registered — the auditor prompt already carries the "may not be
    available" caveat, so both states are safe.
    """
    extensions = compose_audit_extensions(
        submit_tool_module=_SUBMIT_TOOL_MODULE,
        default_prompt=AUDITOR_SYSTEM_PROMPT,
        prompt_override=prompt_override,
        cards_tools_config=cards_tools_config,
        observability_config=observability_config,
    )
    if trajectory_snapshot is not None:
        # Appended after system_prompt so tool registration order does not
        # interfere with the prompt-module sentinel; order only affects which
        # extension installs first, not the child session's tool list shape.
        extensions.append((_GET_TURN_TOOL_MODULE, {"trajectory_snapshot": trajectory_snapshot}))
    return extensions


__all__ = ["compose_auditor_extensions"]
