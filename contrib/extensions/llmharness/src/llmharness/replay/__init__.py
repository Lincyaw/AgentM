"""Replay machinery for the cognitive-audit pipeline.

``record.py`` defines the replay artifact format. The orchestration library —
``engine`` (single-phase replay), ``runner`` (rebuild extensions from a
record), ``offline`` / ``offline_driver`` (offline pipeline replay), and
``fork_tree`` (intervention fork-tree experiment) — lives here in the
package so package code (``distill``, ``eval``) and wheel-installed
scenarios (``rca_eval``) can import it.
"""

from .record import ReplayRecord, iter_records, read_records, write_record

__all__ = [
    "ReplayRecord",
    "iter_records",
    "read_records",
    "write_record",
]
