"""Durable outbox + idempotent inbox for the channels gateway.

See ``.claude/designs/client-server-architecture.md`` §4.5. The
package exposes two ``typing.Protocol``s as the only extension
surface; default implementations are SQLite-backed.
"""

from __future__ import annotations

from .errors import OutboxClosed, OutboxError
from .protocol import InboxLog, OutboxRecord, OutboxStore
from .sqlite import LEASE_TTL_SECONDS, SqliteInbox, SqliteOutbox

__all__ = [
    "LEASE_TTL_SECONDS",
    "InboxLog",
    "OutboxClosed",
    "OutboxError",
    "OutboxRecord",
    "OutboxStore",
    "SqliteInbox",
    "SqliteOutbox",
]
