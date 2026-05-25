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
