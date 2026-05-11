"""Outbox error types."""

from __future__ import annotations


class OutboxError(Exception):
    """Base error for the outbox package."""


class OutboxClosed(OutboxError):
    """Raised when an operation is attempted on a closed outbox/inbox."""


__all__ = ["OutboxClosed", "OutboxError"]
