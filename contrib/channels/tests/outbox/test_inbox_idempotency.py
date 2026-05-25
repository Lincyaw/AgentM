"""InboxLog.record_seen returns False on duplicate."""

from __future__ import annotations

from pathlib import Path

from agentm_channels.outbox import SqliteInbox


def test_record_seen_dedupes_by_peer_and_envelope_id(tmp_path: Path) -> None:
    inbox = SqliteInbox(str(tmp_path / "ib.sqlite"))
    try:
        assert inbox.record_seen("peer-A", "m1", ts=100.0) is True
        assert inbox.record_seen("peer-A", "m1", ts=101.0) is False
        # Same envelope id but different peer is independent.
        assert inbox.record_seen("peer-B", "m1", ts=102.0) is True
    finally:
        inbox.close()


