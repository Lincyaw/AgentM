"""Wire-protocol exceptions.

Pure module — no I/O, no logging. Callers above the wire layer decide
whether a :class:`WireError` is fatal to the connection or recoverable.
"""

from __future__ import annotations


class WireError(Exception):
    """Base class for wire-protocol failures."""


class IncompleteFrame(WireError):
    """Buffer does not yet contain a full length-prefixed frame.

    Not a protocol error — the caller should read more bytes and retry.
    """


class InvalidEnvelope(WireError):
    """Envelope structurally invalid or oversized.

    Raised for malformed JSON, missing required fields, illegal field
    values, and frames whose declared length exceeds the maximum allowed.
    """


__all__ = ["IncompleteFrame", "InvalidEnvelope", "WireError"]
