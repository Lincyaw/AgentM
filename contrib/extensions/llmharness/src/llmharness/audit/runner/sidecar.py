"""Sidecar writer + child/op-sink Protocol boundaries.

Taxonomy-only forwarder. See :mod:`.settings` for rationale on why the
implementation stays consolidated in :mod:`.runner`.
"""

from __future__ import annotations

from .runner import ChildRunner, OpSink, SidecarWriter

__all__ = ["ChildRunner", "OpSink", "SidecarWriter"]
