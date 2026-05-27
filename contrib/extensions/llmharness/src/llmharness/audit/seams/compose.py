"""Shared extension-list composer for both audit phases.

Both `compose_extractor_extensions` and `compose_auditor_extensions` build
the same module list (observability → operations → submit_tool →
system_prompt) with the same `_UNSET`-sentinel knob shape. The only
phase-specific bits are which submit_tool module to wire and which prompt
constant to pin — captured as parameters here.
"""

from __future__ import annotations

from typing import Any

from ..toolkit.atom_constants import (
    OBSERVABILITY_MODULE,
    OPERATIONS_MODULE,
    SYSTEM_PROMPT_MODULE,
)

# Distinguishes "default — include with empty config" from "explicit None
# — drop entirely". A bare `None` default would conflate the two.
UNSET: Any = object()


def compose_audit_extensions(
    *,
    submit_tool_module: str,
    default_prompt: str,
    observability_config: dict[str, Any] | None,
    submit_tool_config: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    obs_cfg = {} if observability_config is UNSET else observability_config
    if obs_cfg is not None:
        out.append((OBSERVABILITY_MODULE, dict(obs_cfg)))

    # OTLP span emission is now driven by the declarative ``Event.to_otel``
    # path inside the observability atom — no separate tracing atom needed.
    # Audit children inherit the same emission contract as their parent.

    # Post harness-collapse (AgentM commit e062913) the session factory
    # fail-stops at freeze unless some atom registered an Operations
    # bundle. Audit children never touch FS/bash, but the substrate
    # check is unconditional — without this entry every spawn raises
    # ExtensionLoadError("<operations>") and the audit pipeline becomes
    # a silent no-op.
    out.append((OPERATIONS_MODULE, {}))

    out.append((submit_tool_module, dict(submit_tool_config) if submit_tool_config else {}))
    out.append((SYSTEM_PROMPT_MODULE, {"prompt": default_prompt}))
    return out


__all__ = ["UNSET", "compose_audit_extensions"]
