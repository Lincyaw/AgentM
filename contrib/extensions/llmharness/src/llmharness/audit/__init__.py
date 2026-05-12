"""Cognitive-audit subpackages.

Phase-specific entry points live under :mod:`.extractor` / :mod:`.auditor`.
Shared schema types live in :mod:`llmharness.schema`; registry types in
:mod:`.registry`. Import directly from those modules — this package
init exposes no convenience re-exports.
"""

from __future__ import annotations
