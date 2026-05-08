"""Adapter layer: per-main-agent integrations.

Each adapter implements the ``TrajectoryAdapter`` / ``ReminderInjector`` /
``AuditTrigger`` shape from
``.claude/designs/llmharness-cognitive-audit.md`` §4.2 for a specific
main-agent kind. The core layer (``schema.py``, ``store.py``, ``cards.py``)
is agent-agnostic; everything that knows about Claude Code hooks or AgentM
events lives here.
"""
