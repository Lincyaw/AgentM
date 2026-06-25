"""Shared protocol, dataclass, and ARL helpers for benchmark adapters."""

from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# Task spec -- uniform representation across all bench formats
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Format-neutral description of a benchmark task."""

    name: str
    prompt: str
    image: str = ""
    path: str = ""
    difficulty: str = ""
    category: str = ""
    # SWE-bench extras
    instance_id: str = ""
    base_commit: str = ""
    repo: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bench adapter protocol
# ---------------------------------------------------------------------------

class BenchAdapter(Protocol):
    """Pluggable adapter for benchmark formats."""

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        """Find tasks from a local repo path or HuggingFace dataset name."""
        ...

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        """Return the Docker image name/tag for this task."""
        ...

    def supports_build(self) -> bool:
        """Whether this format supports local image building."""
        ...

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        """Run evaluation in the sandbox and return score dict."""
        ...

    def format_score_line(self, r: dict) -> str:
        """Format a one-line summary of a task result for console output."""
        ...

    def summary_header(self) -> str:
        """Return the column header line for the summary table."""
        ...

    def summary_row(self, name: str, r: dict) -> str:
        """Return one row of the summary table."""
        ...

    def summary_footer(self, results: dict[str, dict]) -> str:
        """Return the footer/totals line for the summary table."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def image_name(task_name: str, registry: str, prefix: str, tag: str) -> str:
    """Convention-based image name: {registry}/{prefix}-{task}:{tag}."""
    return f"{registry}/{prefix}-{task_name}:{tag}"


def upload_file_to_sandbox(session: object, path: str, content: bytes) -> None:
    """Upload a file to the sandbox via ARL WriteFile API."""
    session._client.upload_file(  # type: ignore[attr-defined]
        session._session_id, path,  # type: ignore[attr-defined]
        base64.b64encode(content).decode(),
        encoding="base64",
    )


def load_trace_tools(session_id: str) -> list[dict]:
    result = subprocess.run(
        ["agentm", "trace", "tools", "--session", session_id, "--format", "ndjson"],
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def replay_tools_to_sandbox(
    session: object, tools: list[dict], *, up_to_turn: int | None = None
) -> int:
    """Replay side-effect tool calls in a sandbox. Returns count replayed."""
    replayed = 0
    assistant_index = -1
    for t in tools:
        tool, args = t.get("tool"), t.get("args", {})

        if tool in ("edit", "write", "bash", "read", "glob", "grep"):
            if assistant_index < 0:
                assistant_index = 0

        if up_to_turn is not None and assistant_index > up_to_turn:
            break

        try:
            if tool == "edit":
                path = args.get("path", "")
                old, new = args.get("old_string", ""), args.get("new_string", "")
                if path and old:
                    rel = path.lstrip("/").removeprefix("app/")
                    cur = session._client.download_file(  # type: ignore[attr-defined]
                        session._session_id, rel  # type: ignore[attr-defined]
                    ).decode("utf-8", errors="replace")
                    upd = cur.replace(old, new, 1)
                    if upd != cur:
                        upload_file_to_sandbox(session, rel, upd.encode())
                        replayed += 1
            elif tool == "write":
                path = args.get("path", "")
                content = args.get("content", "")
                if path:
                    rel = path.lstrip("/").removeprefix("app/")
                    upload_file_to_sandbox(session, rel, content.encode())
                    replayed += 1
            elif tool == "bash":
                cmd = args.get("cmd", "")
                skip = [
                    "make grade", "make qemu", "qemu-system", "timeout",
                    "python3 ok", "make test", "ctest",
                ]
                if cmd and not any(k in cmd for k in skip):
                    session.execute([{  # type: ignore[attr-defined]
                        "name": "r", "command": ["bash", "-lc", cmd],
                        "work_dir": "/app",
                    }])
                    replayed += 1
        except Exception:  # noqa: S110
            pass
    return replayed


def replay_trajectory(session: object, session_id: str) -> int:
    tools = load_trace_tools(session_id)
    return replay_tools_to_sandbox(session, tools)
