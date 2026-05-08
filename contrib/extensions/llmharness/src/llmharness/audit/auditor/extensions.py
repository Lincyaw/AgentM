"""Phase 2 auditor child-session ``extensions`` list."""

from __future__ import annotations

from typing import Any

from .._compose import UNSET, compose_audit_extensions
from .prompt import AUDITOR_SYSTEM_PROMPT

_SUBMIT_TOOL_MODULE = "llmharness.audit.auditor.submit_tool"


def compose_auditor_extensions(
    *,
    prompt_override: str | None = None,
    cards_tools_config: dict[str, Any] | None = UNSET,
    observability_config: dict[str, Any] | None = UNSET,
) -> list[tuple[str, dict[str, Any]]]:
    """Default order: observability → cards_tools → submit_tool → system_prompt.

    Pass ``None`` for ``cards_tools_config`` / ``observability_config`` to drop
    that extension; ``submit_tool`` and ``system_prompt`` always survive.
    """
    return compose_audit_extensions(
        submit_tool_module=_SUBMIT_TOOL_MODULE,
        default_prompt=AUDITOR_SYSTEM_PROMPT,
        prompt_override=prompt_override,
        cards_tools_config=cards_tools_config,
        observability_config=observability_config,
    )


__all__ = ["compose_auditor_extensions"]
