"""Compose the diagnostic-agent's child-session ``extensions`` list.

Replaces the old ``scenarios/harness_monitor/manifest.yaml`` entirely.
``AgentSessionConfig.extensions`` accepts ``list[tuple[module_path, config_dict]]``
directly, so the audit adapter (``adapters/agentm.py``) builds the list
here and hands it to ``api.spawn_child_session`` without any scenario-name
indirection.

Default composition (mirrors the deleted manifest):

1. ``agentm.extensions.builtin.observability`` — captures the diagnostic
   child's full reasoning trace into ``.agentm/observability/<child_id>.jsonl``.
   Design §7 relies on this as the V1 training-data source.
2. ``llmharness.atoms.cards_tools`` — exposes the 42 AFC failure cards
   as ``cards_list`` / ``cards_get`` tools (design §4.4 — cards as tools,
   not as static prompt content).
3. ``agentm.extensions.builtin.system_prompt`` — installs the audit body
   from :data:`llmharness.audit.AUDIT_SYSTEM_PROMPT` (or the caller's
   override) so the diagnostic agent knows what to do.

Customization knobs:

- ``prompt_override`` — replace the entire system prompt body. Useful
  for swapping in an alternate audit voice without forking the package.
- ``cards_tools_config`` — config dict passed to the ``cards_tools``
  extension. ``None`` skips the cards extension entirely (the diagnostic
  agent will run without AFC retrieval).
- ``observability_config`` — config dict passed to the observability
  extension. ``None`` skips it (loses §7 training-data capture).

For full custom audit logic, callers can bypass this factory and pass
their own ``extensions=[...]`` list to the adapter via the
``audit_extensions`` config knob.
"""

from __future__ import annotations

from typing import Any

from .prompt import AUDIT_SYSTEM_PROMPT

_OBSERVABILITY_MODULE = "agentm.extensions.builtin.observability"
_SYSTEM_PROMPT_MODULE = "agentm.extensions.builtin.system_prompt"
_CARDS_TOOLS_MODULE = "llmharness.atoms.cards_tools"


def compose_extensions(
    *,
    prompt_override: str | None = None,
    cards_tools_config: dict[str, Any] | None = None,
    observability_config: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build the audit child-session ``extensions`` list.

    All knobs default to "include the standard piece with empty config".
    Pass ``None`` for ``cards_tools_config`` or ``observability_config``
    to drop that extension entirely.
    """

    out: list[tuple[str, dict[str, Any]]] = []

    if observability_config is not None:
        out.append((_OBSERVABILITY_MODULE, dict(observability_config)))

    if cards_tools_config is not None:
        out.append((_CARDS_TOOLS_MODULE, dict(cards_tools_config)))

    out.append(
        (
            _SYSTEM_PROMPT_MODULE,
            {"prompt": prompt_override if prompt_override is not None else AUDIT_SYSTEM_PROMPT},
        )
    )

    return out


__all__ = ["compose_extensions"]
