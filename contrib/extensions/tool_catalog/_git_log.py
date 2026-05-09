"""Git introspection helpers for the tool catalog extension package."""

from __future__ import annotations

import subprocess
from pathlib import Path


def list_history(path: str, *, limit: int, root: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "git",
            "log",
            "--format=%H%x00%an%x00%aI%x00%s",
            "-n",
            str(limit),
            "--",
            path,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "<no output>"
        raise RuntimeError(stderr)
    out: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line:
            continue
        sha, author, timestamp, message = line.split("\x00", 3)
        out.append(
            {
                "sha": sha,
                "author": author,
                "timestamp": timestamp,
                "message": message,
            }
        )
    return out
