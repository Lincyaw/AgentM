"""Per-firing ``ExtractionState`` for the v3 extractor (split out of the
former 1241-line ``state.py``).

The full validation pipeline still lives as methods on
:class:`ExtractionState` in :mod:`.core` — the standalone module-level
validators (``_validate_event_shape``, ``_validate_ref_shape``,
``_validate_external_ref_shape``, ``_compute_degree_warning``,
``_coerce_int``) are factored out into :mod:`.validate`.

The further commit / shape-witness split called out in the reorg spec
(``commit.py``, ``witness.py``) was deferred: the dataclass methods
that implement the commit pipeline (``commit``, ``commit_batch``,
``finalize``, ``_validate_and_append``, ``apply_*``) hold non-trivial
shared state on ``self`` and can't be cleanly extracted into separate
modules without restructuring the class into mixins or rewriting the
methods as free functions taking the state instance as first arg —
either of which is a behavior-adjacent refactor beyond the
pure-rename scope.
"""

from __future__ import annotations

from .core import ExtractionState
from .validate import _compute_degree_warning

__all__ = ["ExtractionState", "_compute_degree_warning"]
