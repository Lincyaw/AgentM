"""Lazy-init contract for ``GitBackedResourceWriter``.

Locks down "cost-tied-to-usage": constructing a writer must not touch disk.
Disk side effects (shadow ``.agentm/repo`` bare repo, initial snapshot,
advisory-mode warning) only happen on the first mutating call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agentm.core.abi import EventBus
from agentm.core.runtime.resource_writer import GitBackedResourceWriter


@pytest.mark.asyncio
async def test_no_writes_no_disk_footprint(tmp_path: Path) -> None:
    """Constructing a writer against a cwd with no .git must not create any
    files. Read-only / classify-only scenarios pay zero filesystem cost.
    """
    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="lazy-session",
        bus=EventBus(),
        auto_commit=True,
    )

    # Inspection-only methods must not trigger setup.
    writer.classify("skills/foo/SKILL.md")
    # current_version_for_path is on the observability hot path — it fires
    # on every session entry and must stay read-only.
    assert writer.current_version_for_path("skills/foo/SKILL.md") is None
    assert writer.current_version_for_path("anything.txt") is None

    assert not (tmp_path / ".agentm").exists()
    assert not (tmp_path / ".agentm" / "repo").exists()
    # Sanity: no leftover entries in cwd at all besides what pytest seeded.
    assert list(tmp_path.iterdir()) == []
    # Internal state should remain pristine — no probing happened yet.
    assert writer._setup_done is False  # type: ignore[attr-defined]
    assert writer._advisory_mode is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_first_write_creates_shadow_repo(tmp_path: Path) -> None:
    """First write triggers lazy setup: shadow repo appears, content lands,
    commit is recorded.
    """
    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="first-write",
        bus=EventBus(),
        auto_commit=True,
    )

    assert not (tmp_path / ".agentm" / "repo").exists()

    result = await writer.write(
        "notes.txt",
        b"hello\n",
        rationale="first write",
    )

    shadow = tmp_path / ".agentm" / "repo"
    assert shadow.is_dir(), "shadow bare repo should be created on first write"
    assert (tmp_path / "notes.txt").read_bytes() == b"hello\n"
    # notes.txt is unmanaged (no manifest globs match), so committed=False
    # is the expected shape; what we care about is the shadow repo existing
    # and the bytes being on disk.
    assert result.error is None


@pytest.mark.asyncio
async def test_lazy_setup_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_lazy_setup`` must be safe to call repeatedly: only the first call
    runs the ``git init --bare`` subprocess.
    """
    real_run = subprocess.run
    init_call_count = {"n": 0}

    def counting_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "init":
            init_call_count["n"] += 1
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(
        "agentm.core.runtime.resource_writer.subprocess.run",
        counting_run,
    )

    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="idem-session",
        bus=EventBus(),
        auto_commit=True,
    )

    # Two writes in a row.
    await writer.write("a.txt", b"a\n", rationale="one")
    await writer.write("b.txt", b"b\n", rationale="two")

    assert writer._setup_done is True  # type: ignore[attr-defined]
    assert init_call_count["n"] == 1, (
        f"expected exactly one `git init --bare`, saw {init_call_count['n']}"
    )

    # Direct repeat invocation must also no-op.
    writer._lazy_setup()  # type: ignore[attr-defined]
    writer._lazy_setup()  # type: ignore[attr-defined]
    assert init_call_count["n"] == 1


@pytest.mark.asyncio
async def test_advisory_warning_deferred_until_first_write(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With ``auto_commit=False``, the advisory-mode warning must fire on
    first write — not at construction time — and must fire exactly once
    across multiple writes.
    """
    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="advisory-deferred",
        bus=EventBus(),
        auto_commit=False,
    )

    # No warning yet — construction must be silent under lazy init.
    init_warnings = [
        rec for rec in caplog.records
        if "resource writer advisory mode enabled" in rec.message
    ]
    assert init_warnings == []
    assert writer._advisory_mode is False  # type: ignore[attr-defined]

    target = tmp_path / "x.txt"
    await writer.write("x.txt", b"one\n", rationale="one")
    await writer.write("x.txt", b"two\n", rationale="two")

    warnings = [
        rec for rec in caplog.records
        if "resource writer advisory mode enabled" in rec.message
    ]
    assert len(warnings) == 1, f"expected one advisory warning, saw {len(warnings)}"
    assert writer._advisory_mode is True  # type: ignore[attr-defined]
    assert target.read_bytes() == b"two\n"


@pytest.mark.asyncio
async def test_observability_hot_path_does_not_trigger_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the observability fingerprint hook: it calls
    ``current_version_for_path`` many times per session entry. None of
    those calls may spawn a subprocess or create the shadow repo.

    Regression guard — the unit canary alone is insufficient because real
    sessions call this hook before any tool fires a write.
    """
    real_run = subprocess.run
    subprocess_calls: list[list[str]] = []

    def tracking_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list):
            subprocess_calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(
        "agentm.core.runtime.resource_writer.subprocess.run",
        tracking_run,
    )

    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="observability-loop",
        bus=EventBus(),
        auto_commit=True,
    )

    # Observability iterates over every loaded builtin source file.
    fake_atom_paths = [f"src/agentm/extensions/builtin/atom_{i}.py" for i in range(50)]
    for path in fake_atom_paths:
        result = writer.current_version_for_path(path)
        assert result is None

    assert subprocess_calls == [], (
        f"current_version_for_path must not spawn subprocesses pre-write; "
        f"saw {len(subprocess_calls)} call(s): {subprocess_calls[:3]!r}"
    )
    assert not (tmp_path / ".agentm").exists()
    assert writer._setup_done is False  # type: ignore[attr-defined]
