"""Per-firing ``ExtractionState`` for the v4 extractor.

The ``apply_*`` op-log surface and :meth:`ExtractionState.finalize` live
as methods on :class:`ExtractionState` in :mod:`.core`; the standalone
module-level validators (``_validate_event_shape``,
``_compute_degree_warning``, ``_coerce_int``) are factored out into
:mod:`.validate`.
"""

from __future__ import annotations

from .core import ExtractionState
from .validate import _compute_degree_warning

__all__ = ["ExtractionState", "_compute_degree_warning"]
