"""Reference audit check functions for the cognitive-audit pipeline.

Each submodule exports a pure function ``(events, edges) → list[Finding]``.
These are no longer atoms — the trigger/registry machinery was removed
during the architecture simplification. They can be called directly by
any consumer that has a graph view.

Available checks:

- :mod:`.check_repeated_actions` — flags identical-summary ``act`` events.
- :mod:`.check_open_branches` — flags ``dec`` / ``hyp`` events with no
  outgoing data edge.
- :mod:`.check_premature_conclusion` — flags ``concl`` events with fewer
  than two incoming edges.
"""

from __future__ import annotations

__all__: list[str] = []
