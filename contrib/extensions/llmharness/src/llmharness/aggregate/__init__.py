"""Per-case aggregation of llmharness run artefacts.

A "case" is one main-agent session run on one input. The aggregator
walks the replay sidecar (and optional meta sidecar) for a session,
groups the data by source phase, and writes a directory layout suited
to human review and downstream training-data export.

Entry point: ``llmharness-aggregate`` CLI (see :mod:`.cli`).
"""

from __future__ import annotations

from .case import CaseData, CaseMeta, FiringRecord, IndexSnapshot
from .collector import collect_case
from .fork_collector import export_forks
from .session_collector import collect_session_case
from .writer import write_case

__all__ = [
    "CaseData",
    "CaseMeta",
    "FiringRecord",
    "IndexSnapshot",
    "collect_case",
    "collect_session_case",
    "export_forks",
    "write_case",
]
