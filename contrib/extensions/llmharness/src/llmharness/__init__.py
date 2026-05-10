"""LLM-as-harness: cognitive-audit AgentM extension.

Public surface is the typed payloads (``Event``, ``Verdict``, ``Reminder``,
``EventKind``) and the AFC card loader. The runtime entry point is the AgentM
extension at ``llmharness.adapters.agentm``, loaded via
``AgentSessionConfig(extensions=[("llmharness.adapters.agentm", {})])``.

V2 breaking change (issue #134, 2026-05-10): ``DriftType`` is removed.
"""

from .schema import Event, EventKind, Reminder, Verdict

__all__ = [
    "Event",
    "EventKind",
    "Reminder",
    "Verdict",
]
