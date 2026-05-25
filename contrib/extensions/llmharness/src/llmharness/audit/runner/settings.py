"""Per-install settings dataclasses for the harness runner.

Taxonomy-only forwarder: ``ExtractorSettings`` and ``AuditorSettings``
are defined in :mod:`.runner` next to :class:`HarnessRunner` because
the runner instantiates their factory classmethods directly and the
two settings classes pull from ``compose_*_extensions`` helpers that
also live in the runner's import graph. Splitting them into a
self-contained file would have introduced a cycle. Importers that
want the categorical view (``from llmharness.audit.runner.settings
import ExtractorSettings``) get it via these re-exports without
behavior change.
"""

from __future__ import annotations

from .runner import AuditorSettings, ExtractorSettings

__all__ = ["AuditorSettings", "ExtractorSettings"]
