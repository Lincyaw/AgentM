"""LLM-as-harness: cognitive-audit AgentM extension.

Public surface is the typed payloads (``Event``, ``Verdict``, ``Reminder``,
``EventKind``, ``DriftType``) and the AFC card loader. The runtime entry
point is the AgentM extension at ``llmharness.adapters.agentm``, loaded
via ``AgentSessionConfig(extensions=[("llmharness.adapters.agentm", {})])``.
"""

from .schema import DriftType, Event, EventKind, Reminder, Verdict

__all__ = [
    "DriftType",
    "Event",
    "EventKind",
    "Reminder",
    "Verdict",
]
