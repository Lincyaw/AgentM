"""Sidecar writer + child/op-sink Protocol boundaries.

Taxonomy-only forwarder. The split was deferred — out of scope for the
pure-rename refactor; the dependency graph is a clean DAG and can be
done in a follow-up.
"""

from __future__ import annotations

from .runner import ChildRunner, OpSink, SidecarWriter

__all__ = ["ChildRunner", "OpSink", "SidecarWriter"]
