"""Shared extension-list composer for both audit phases.

Both `compose_extractor_extensions` and `compose_auditor_extensions` build
the same four-module list (observability → cards_tools → submit_tool →
system_prompt) with the same `_UNSET`-sentinel knob shape. The only
phase-specific bits are which submit_tool module to wire and which prompt
constant to pin — captured as parameters here.
"""

from __future__ import annotations

from typing import Any

_OBSERVABILITY_MODULE = "agentm.extensions.builtin.observability"
_OPERATIONS_MODULE = "agentm.extensions.builtin.operations_local"
_SYSTEM_PROMPT_MODULE = "agentm.extensions.builtin.system_prompt"
_CARDS_TOOLS_MODULE = "llmharness.atoms.cards_tools"

# Distinguishes "default — include with empty config" from "explicit None
# — drop entirely". A bare `None` default would conflate the two.
UNSET: Any = object()


def compose_audit_extensions(
    *,
    submit_tool_module: str,
    default_prompt: str,
    prompt_override: str | None,
    cards_tools_config: dict[str, Any] | None,
    observability_config: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    obs_cfg = {} if observability_config is UNSET else observability_config
    if obs_cfg is not None:
        out.append((_OBSERVABILITY_MODULE, dict(obs_cfg)))

    # Post harness-collapse (AgentM commit e062913) the session factory
    # fail-stops at freeze unless some atom registered an Operations
    # bundle. Audit children never touch FS/bash, but the substrate
    # check is unconditional — without this entry every spawn raises
    # ExtensionLoadError("<operations>") and the audit pipeline becomes
    # a silent no-op.
    out.append((_OPERATIONS_MODULE, {}))

    cards_cfg = {} if cards_tools_config is UNSET else cards_tools_config
    if cards_cfg is not None:
        out.append((_CARDS_TOOLS_MODULE, dict(cards_cfg)))

    out.append((submit_tool_module, {}))
    out.append(
        (
            _SYSTEM_PROMPT_MODULE,
            {"prompt": prompt_override if prompt_override is not None else default_prompt},
        )
    )
    return out


__all__ = ["UNSET", "compose_audit_extensions"]
