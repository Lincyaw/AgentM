"""Offline replay for the two-phase cognitive audit pipeline.

Lets you run the extractor or auditor in isolation against recorded inputs,
or chain them end-to-end on a recorded main-agent trajectory.

Submodule layout
----------------
* :mod:`llmharness.replay.record` — sidecar I/O. No agentm dependency.
* :mod:`.runner` — rebuild extensions from a record + run.
* :mod:`.cli`    — ``llmharness-replay`` entry point.
* :mod:`.engine` — phase driver (host-side, spawns top-level sessions).
"""

from llmharness.replay.record import ReplayRecord, iter_records, read_records, write_record

__all__ = [
    "ReplayRecord",
    "iter_records",
    "read_records",
    "write_record",
]
