"""Tests for the SessionBindingStore — the persistent session → host map.

Fail-stop positions:
- Round-trip after process restart (the whole point — survive gateway restart).
- ``upsert`` with ``resume_id=None`` preserves an existing resume_id.
- ``set_resume_id`` updates only that field.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentm_channels.session_bindings import SessionBindingStore


def test_get_returns_none_for_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = SessionBindingStore(Path(d) / "b.sqlite")
        try:
            assert store.get("terminal:t1") is None
        finally:
            store.close()


def test_upsert_then_get_round_trip() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = SessionBindingStore(Path(d) / "b.sqlite")
        try:
            store.upsert("terminal:t1", host_id="worker-A", resume_id="sess-001")
            b = store.get("terminal:t1")
            assert b is not None
            assert b.host_id == "worker-A"
            assert b.resume_id == "sess-001"
            assert b.last_seen_at > 0
        finally:
            store.close()


def test_upsert_survives_close_and_reopen() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "b.sqlite"
        s1 = SessionBindingStore(path)
        s1.upsert("terminal:t1", host_id="worker-A", resume_id="sess-001")
        s1.close()
        # New process, new connection — what gateway restart looks like.
        s2 = SessionBindingStore(path)
        try:
            b = s2.get("terminal:t1")
            assert b is not None
            assert b.host_id == "worker-A"
            assert b.resume_id == "sess-001"
        finally:
            s2.close()


def test_upsert_none_resume_preserves_existing() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = SessionBindingStore(Path(d) / "b.sqlite")
        try:
            store.upsert("k", host_id="A", resume_id="sess-1")
            # Re-bind to a different host without supplying resume_id —
            # the previously recorded one must NOT be erased.
            store.upsert("k", host_id="B", resume_id=None)
            b = store.get("k")
            assert b is not None
            assert b.host_id == "B"
            assert b.resume_id == "sess-1"
        finally:
            store.close()


def test_set_resume_id_only_updates_that_field() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = SessionBindingStore(Path(d) / "b.sqlite")
        try:
            store.upsert("k", host_id="A", resume_id=None)
            store.set_resume_id("k", "sess-42")
            b = store.get("k")
            assert b is not None
            assert b.host_id == "A"
            assert b.resume_id == "sess-42"
        finally:
            store.close()


def test_all_for_host_returns_only_matching() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = SessionBindingStore(Path(d) / "b.sqlite")
        try:
            store.upsert("k1", host_id="A", resume_id="s1")
            store.upsert("k2", host_id="A", resume_id="s2")
            store.upsert("k3", host_id="B", resume_id="s3")
            assert {b.session_key for b in store.all_for_host("A")} == {"k1", "k2"}
            assert {b.session_key for b in store.all_for_host("B")} == {"k3"}
            assert store.all_for_host("nope") == []
        finally:
            store.close()


def test_delete_removes_binding() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = SessionBindingStore(Path(d) / "b.sqlite")
        try:
            store.upsert("k", host_id="A", resume_id="s")
            store.delete("k")
            assert store.get("k") is None
        finally:
            store.close()
