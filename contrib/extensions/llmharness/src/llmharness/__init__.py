"""LLM-as-harness: cognitive-audit AgentM extension.

Public surface — kept deliberately small. Re-export rule: a symbol
appears here only if at least one in-tree consumer (rca eval, the
strict-A/B test suite, smoke tests) OR an out-of-tree trainer
(rca-autorl) imports it through the top-level package. Everything else
stays reachable via its submodule path and gets promoted on demand.

Phase contract surface — exposed for external trainers that RL-train
either child agent. Both extractor and auditor phase faces are now
mounted here. The extractor's tool surface lives at
:mod:`llmharness.audit.extractor.atom`; the auditor's lives at
:mod:`llmharness.audit.auditor.atom`.

Currently exported:

* Wire-type dataclasses from :mod:`llmharness.schema` —
  ``Event`` / ``EventKind`` / ``Edge`` / ``EdgeKind`` / ``Finding`` /
  ``Phase`` / ``Reminder`` / ``Verdict``. These define the
  replay-record / audit-graph data model and are the typed view every
  consumer needs.
* :class:`ReplayRecord` + :func:`iter_records` / :func:`write_record`
  — replay sidecar record format and I/O. Used directly by both the
  strict-A/B helpers and downstream consumers that read sidecars.
* Strict-A/B fork orchestration — :class:`ReminderCandidate`,
  :class:`OfflineAuditRun`, :func:`run_offline_auditor_over_control`,
  :func:`write_strict_ab_replay`, :func:`strict_ab_replay_path`. The
  primary entry points the rca eval driver calls.

Other helpers (``AuditorOutputError``, ``RawVerdictOutput``,
``merge_to_phases``, ``flatten_assistant_blocks``,
``serialize_full_trajectory``, ``now_ns``, ``replay_auditor_record``)
remain available via their submodules. Promote them here when an
in-tree caller actually needs them through the top-level surface.

The runtime entry point is the AgentM extension at
``llmharness.adapters.agentm``, loaded via
``AgentSessionConfig(extensions=[("llmharness.adapters.agentm", {})])``.

V2 breaking change (issue #134, 2026-05-10): ``DriftType`` is removed.
"""

from .audit.auditor import (
    AUDITOR_TERMINATION_REASON,
    AUDITOR_TOOL_NAMES,
    AUDITOR_TOOLS,
    load_auditor_prompt,
)
from .audit.extractor import (
    EXTRACTOR_TERMINATION_REASON,
    EXTRACTOR_TOOL_NAMES,
    load_extractor_prompt,
)
from .replay.record import ReplayRecord, iter_records, write_record
from .replay.strict_ab import (
    OfflineAuditRun,
    ReminderCandidate,
    run_offline_auditor_over_control,
    strict_ab_replay_path,
    write_strict_ab_replay,
)
from .schema import (
    Edge,
    EdgeKind,
    Event,
    EventKind,
    Finding,
    Phase,
    Reminder,
    Verdict,
)
from .train_signals import (
    ToolEvent,
    auditor_process_reward,
    extractor_process_reward,
)

__all__ = [
    "AUDITOR_TERMINATION_REASON",
    "AUDITOR_TOOLS",
    "AUDITOR_TOOL_NAMES",
    "EXTRACTOR_TERMINATION_REASON",
    "EXTRACTOR_TOOL_NAMES",
    "Edge",
    "EdgeKind",
    "Event",
    "EventKind",
    "Finding",
    "OfflineAuditRun",
    "Phase",
    "Reminder",
    "ReminderCandidate",
    "ReplayRecord",
    "ToolEvent",
    "Verdict",
    "auditor_process_reward",
    "extractor_process_reward",
    "iter_records",
    "load_auditor_prompt",
    "load_extractor_prompt",
    "run_offline_auditor_over_control",
    "strict_ab_replay_path",
    "write_record",
    "write_strict_ab_replay",
]
