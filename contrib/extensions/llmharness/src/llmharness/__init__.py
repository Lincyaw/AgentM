"""LLM-as-harness: cognitive-audit AgentM extension.

Public surface is the typed payloads from :mod:`llmharness.schema`
(``Event``, ``Edge``, ``Finding``, ``Verdict``, ``Reminder``,
``EventKind``, ``EdgeKind``, ``Phase``), the AFC card loader, and the
top-level entry points consumers need to drive an offline auditor pass
or a fork-and-replay over recorded trajectories:

* :func:`flatten_assistant_blocks` / :func:`serialize_full_trajectory` —
  trajectory helpers re-exported from :mod:`llmharness.adapters.agentm`
  so downstream eval drivers don't reach into the adapter module.
* :class:`AuditorOutputError` / :class:`RawVerdictOutput` — typed view
  over the auditor's terminal-tool payload.
* :func:`merge_to_phases` — fold ``Event`` records into ``Phase``
  windows (re-exported from :mod:`llmharness.audit.phase`).
* :class:`ReplayRecord`, :func:`iter_records`, :func:`write_record`,
  :func:`now_ns` — replay-sidecar record format and IO helpers.
* :func:`replay_auditor_record` — re-run a recorded auditor firing.
* :class:`ReminderCandidate` / :class:`OfflineAuditRun` /
  :func:`run_offline_auditor_over_control` /
  :func:`write_strict_ab_replay` / :func:`strict_ab_replay_path` —
  strict-A/B fork orchestration (replay auditor against a fixed
  control trajectory, then stitch a unified replay sidecar).

The runtime entry point is the AgentM extension at
``llmharness.adapters.agentm``, loaded via
``AgentSessionConfig(extensions=[("llmharness.adapters.agentm", {})])``.

V2 breaking change (issue #134, 2026-05-10): ``DriftType`` is removed.
"""

from .adapters.agentm import flatten_assistant_blocks, serialize_full_trajectory
from .audit.auditor.output import AuditorOutputError, RawVerdictOutput
from .audit.phase import merge_to_phases
from .replay.record import ReplayRecord, iter_records, now_ns, write_record
from .replay.runner import replay_auditor_record
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

__all__ = [
    "AuditorOutputError",
    "Edge",
    "EdgeKind",
    "Event",
    "EventKind",
    "Finding",
    "OfflineAuditRun",
    "Phase",
    "RawVerdictOutput",
    "Reminder",
    "ReminderCandidate",
    "ReplayRecord",
    "Verdict",
    "flatten_assistant_blocks",
    "iter_records",
    "merge_to_phases",
    "now_ns",
    "replay_auditor_record",
    "run_offline_auditor_over_control",
    "serialize_full_trajectory",
    "strict_ab_replay_path",
    "write_record",
    "write_strict_ab_replay",
]
