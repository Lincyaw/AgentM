"""Cognitive-audit feature, packaged as composable Python modules.

The cognitive audit (design: ``.claude/designs/llmharness-cognitive-audit.md``)
is the V0 diagnostic layer that runs in a child AgentM session triggered after
every main-agent turn. The diagnostic agent's structured output rides on a
forced ``submit_audit`` tool call, not on trailing-text JSON — see
:mod:`llmharness.audit.submit_tool`.

Public API surface (everything re-exported here is part of the package's
extension contract; everything else is private):

- :func:`compose_extensions` — build the child-session ``extensions`` list
- :data:`AUDIT_SYSTEM_PROMPT` — default audit system prompt body
- :class:`RawAuditOutput` / :class:`RawAuditEvent` — typed coercion of the
  ``submit_audit`` tool-call arguments into the typed Verdict + Event
  payloads the adapter persists to the session entry tree
"""

from __future__ import annotations

from .extensions import compose_extensions
from .output import RawAuditEvent, RawAuditOutput
from .prompt import AUDIT_SYSTEM_PROMPT

__all__ = [
    "AUDIT_SYSTEM_PROMPT",
    "RawAuditEvent",
    "RawAuditOutput",
    "compose_extensions",
]
