"""Reference scenario check atoms for the v3 cognitive-audit pipeline.

Each submodule is a §11 single-file extension. They are mounted on
the parent session AFTER ``llmharness.adapter`` has published
the audit registry service. Each atom calls
``api.get_service("llmharness.audit_registry").register_check(...)``
from inside ``install(api, config)`` and fails fast if the service is
not present.

Available reference atoms:

- :mod:`.check_repeated_actions` — flags identical-summary ``act`` events.
- :mod:`.check_open_branches` — flags ``dec`` / ``hyp`` events with no
  outgoing data edge.
- :mod:`.check_premature_conclusion` — flags ``concl`` events with fewer
  than two incoming edges.

These ship as worked examples; downstream scenarios (e.g. rca-autorl)
typically register their own one-file atoms following the same shape.
"""

from __future__ import annotations

__all__: list[str] = []
