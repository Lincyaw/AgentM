"""Replay-record I/O for the cognitive-audit pipeline.

``record.py`` defines the sidecar format shared between core and tools.
Orchestration tools (chain replay, fork tree, offline driver) live under
``tools/replay/``.
"""

from .record import ReplayRecord, iter_records, read_records, write_record

__all__ = [
    "ReplayRecord",
    "iter_records",
    "read_records",
    "write_record",
]
