"""TELBench span-level error localization evaluation.

Data adapter (span → typed-message conversion) and pure scoring for
TELBench, consumed by the ``auditor_eval`` benchmark. The standalone
``telbench-eval`` CLI and the eval→reflect→evolve iteration tooling were
removed; ``agentm-eval auditor run --telbench-data`` is the entry point.
"""

from __future__ import annotations
