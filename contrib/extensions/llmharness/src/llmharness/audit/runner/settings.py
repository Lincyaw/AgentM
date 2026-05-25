"""Per-install settings dataclasses for the harness runner.

Taxonomy-only forwarder: ``ExtractorSettings`` and ``AuditorSettings``
remain defined in :mod:`.runner` next to :class:`HarnessRunner`. The
split was deferred — out of scope for the pure-rename refactor; the
dependency graph is a clean DAG (settings have no back-edge into
``HarnessRunner``) and the split can be done in a follow-up.
Importers that want the categorical view
(``from llmharness.audit.runner.settings import ExtractorSettings``)
get it via these re-exports without behavior change.
"""

from __future__ import annotations

from .runner import AuditorSettings, ExtractorSettings

__all__ = ["AuditorSettings", "ExtractorSettings"]
