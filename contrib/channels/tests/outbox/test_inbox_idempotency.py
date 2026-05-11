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


def test_prune_drops_old_entries(tmp_path: Path) -> None:
    inbox = SqliteInbox(str(tmp_path / "ib.sqlite"))
    try:
        inbox.record_seen("peer-A", "m1", ts=100.0)
        inbox.record_seen("peer-A", "m2", ts=200.0)
        removed = inbox.prune(older_than=150.0)
        assert removed == 1
        # m1 was pruned, so it is "new" again.
        assert inbox.record_seen("peer-A", "m1", ts=300.0) is True
        # m2 remains.
        assert inbox.record_seen("peer-A", "m2", ts=300.0) is False
    finally:
        inbox.close()
