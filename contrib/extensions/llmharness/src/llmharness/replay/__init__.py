"""Offline replay for the two-phase cognitive audit pipeline.

Lets you run the extractor or auditor in isolation against recorded inputs,
or chain them end-to-end on a recorded main-agent trajectory. Use cases:

* **Isolate auditor quality** — collect a baseline with
  ``enable_auditor: false``, then replay the auditor on saved graphs to
  see what it would have surfaced without polluting the main run.
* **Swap models / prompts** — feed a recorded payload through a different
  provider or with ``--prompt-override`` to A/B prompt revisions.
* **Bug bisection** — when a regression appears, replay the same inputs
  through old and new auditor prompts on the same graph; diff outputs.

Submodule layout
----------------
* :mod:`llmharness.replay.record` — sidecar I/O. No agentm dependency.
* :mod:`llmharness.replay.runner` — rebuild extensions from a record + run.
* :mod:`llmharness.replay.cli`    — ``llmharness-replay`` entry point.

The host-side phase driver and prefix-replay helpers live in
:mod:`llmharness.tools` (``tools.engine`` / ``tools.prefix_replay``)
because they reach into ``agentm.core.runtime.*`` to spawn standalone
sessions — that placement signals "not an atom, not API surface".

``runner`` / ``cli`` import :mod:`agentm`, so they are *not* re-exported
from this ``__init__`` — downstream consumers that only need the record
types (e.g. ``rca-autorl`` reading sidecar files) can
``from llmharness.replay.record import …`` without pulling AgentM in.
"""

from .record import ReplayRecord, iter_records, read_records, write_record

__all__ = [
    "ReplayRecord",
    "iter_records",
    "read_records",
    "write_record",
]
