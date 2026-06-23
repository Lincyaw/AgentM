"""Replay machinery for llmharness eval workflows.

``record.py`` defines the replay artifact format. The orchestration library —
``engine`` (single-phase replay), ``runner`` (rebuild extensions from a
record), ``offline`` / ``offline_driver`` (offline pipeline replay), and
``fork_tree`` (intervention fork-tree experiment) lives under ``eval`` because
replay is only used for offline evaluation and artifact aggregation.
"""

from .record import ReplayRecord, iter_records, read_records, write_record

__all__ = [
    "ReplayRecord",
    "iter_records",
    "read_records",
    "write_record",
]
