"""Cognitive-audit feature, packaged as composable Python modules.

The cognitive audit (design: ``.claude/designs/llmharness-cognitive-audit.md``)
is the V0 diagnostic layer that runs in a child AgentM session triggered after
every main-agent turn. Until 2026-05-08 it lived as a separate
``scenarios/harness_monitor/`` directory whose ``manifest.yaml`` listed three
extensions and inlined a 250-line system prompt; the V0 adapter then resolved
the scenario by name through ``AgentSessionConfig(scenario="harness_monitor")``.

That indirection bought nothing. The diagnostic agent is never invoked
standalone — only spawned by ``adapters/agentm.py`` — and ``AgentSessionConfig``
already accepts ``extensions=list[tuple[str, dict]]`` directly. So the scenario
directory is gone; the same composition is now built in :func:`compose_extensions`
and the prompt is a Python string in :mod:`llmharness.audit.prompt`.

Public API surface (everything re-exported here is part of the package's
extension contract; everything else is private):

- :func:`compose_extensions` — build the child-session ``extensions`` list
- :data:`AUDIT_SYSTEM_PROMPT` — default audit system prompt body
- :class:`RawAuditOutput` / :class:`RawAuditEvent` — typed parser for the
  diagnostic agent's emitted JSON; shared between the in-process audit
  (``adapters/agentm.py``) and the dormant subprocess bridge
  (``agentm_bridge.py``)
- :func:`extract_json` — trailing-JSON extractor used by both consumers
"""

from __future__ import annotations

from .extensions import compose_extensions
from .json_extract import extract_json
from .output import RawAuditEvent, RawAuditOutput
from .prompt import AUDIT_SYSTEM_PROMPT

__all__ = [
    "AUDIT_SYSTEM_PROMPT",
    "RawAuditEvent",
    "RawAuditOutput",
    "compose_extensions",
    "extract_json",
]
