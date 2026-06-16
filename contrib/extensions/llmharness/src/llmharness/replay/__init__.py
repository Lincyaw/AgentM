"""Replay machinery for the cognitive-audit pipeline.

``record.py`` defines the sidecar format. The orchestration library —
``engine`` (single-phase replay), ``runner`` (rebuild extensions from a
record), ``offline`` / ``offline_driver`` (offline pipeline replay), and
``fork_tree`` (intervention fork-tree experiment) — lives here in the
package so package code (``distill``, ``eval``) and wheel-installed
scenarios (``rca_eval``) can import it. The thin CLI/script entry points
(``cli``, ``chain``, ``prefix_replay``, ``reminder_seed``) stay under
``tools/replay/`` and import this library via ``llmharness.replay.*``.
"""

from .record import ReplayRecord, iter_records, read_records, write_record

__all__ = [
    "ReplayRecord",
    "iter_records",
    "read_records",
    "write_record",
]
