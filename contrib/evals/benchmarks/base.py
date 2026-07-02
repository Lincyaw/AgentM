"""Shared protocol, dataclass, and ARL helpers for benchmark adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterator, Protocol

from loguru import logger


# ---------------------------------------------------------------------------
# Task spec -- uniform representation across all bench formats
# ---------------------------------------------------------------------------

@dataclass(slots=True)
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
        """Return the runtime Docker image (convention-named in our registry)."""
        ...

    def get_source_image(self, task: TaskSpec) -> str | None:
        """Return the upstream image URL for mirroring, or None if self-built."""
        ...

    def supports_build(self) -> bool:
        """Whether this format supports local image building."""
        ...

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        """Run evaluation in the sandbox and return score dict."""
        ...

    def is_pass(self, result: dict) -> bool:
        """Whether this single-attempt result counts as a pass."""
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

    def pass_at_k_header(self) -> str:
        """Column header for the pass@k table."""
        ...

    def pass_at_k_row(self, name: str, runs: list[dict]) -> tuple[str, dict]:
        """One row of the pass@k table. Returns (formatted_line, row_stats)."""
        ...

    def pass_at_k_footer(self, all_stats: list[dict], n_tasks: int) -> str:
        """Footer/totals for the pass@k table."""
        ...


class PrivateEvalBuilder(Protocol):
    """Optional adapter hook for packaging private evaluator assets."""

    def get_eval_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        """Return the private evaluator Docker image name."""
        ...

    def build_eval_image(
        self,
        task: TaskSpec,
        *,
        base_image: str,
        eval_image: str,
        push: bool = False,
    ) -> None:
        """Build one private evaluator image for a task."""
        ...


class PrivateEvalRunner(Protocol):
    """Optional adapter hook for running private-container evaluation."""

    def private_eval_container(
        self,
        task: TaskSpec,
        registry: str,
        prefix: str,
        tag: str,
        *,
        container: str,
        image_pull_policy: str,
    ) -> dict[str, Any]:
        """Return the ARL private container spec for evaluation."""
        ...

    def evaluate_private_container(
        self,
        session: object,
        task: TaskSpec,
        *,
        container: str = "eval",
        timeout: int = 300,
    ) -> dict:
        """Evaluate using a configured private container."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_UPLOAD_CHUNK_BYTES = 1024 * 1024

def image_name(task_name: str, registry: str, prefix: str, tag: str) -> str:
    """Convention-based image name: {registry}/{prefix}-{task}:{tag}."""
    return f"{registry}/{prefix}-{task_name}:{tag}"


def eval_image_name(task_name: str, registry: str, prefix: str, tag: str) -> str:
    """Convention-based private evaluator image name."""
    return f"{registry}/{prefix}-{task_name}-eval:{tag}"


def _iter_upload_chunks(content: bytes) -> Iterator[bytes]:
    for offset in range(0, len(content), _UPLOAD_CHUNK_BYTES):
        yield content[offset : offset + _UPLOAD_CHUNK_BYTES]


def upload_file_to_sandbox(session: object, path: str, content: bytes) -> None:
    """Upload a file into the sandbox's /app tree via ARL's workspace file API."""
    rel_path = str(PurePosixPath(path.lstrip("/")))
    target = f"/app/{rel_path}"
    expected_sha256 = hashlib.sha256(content).hexdigest()
    upload_rel = ""
    for attempt in range(1, 5):
        upload_rel = f".agentm_eval_uploads/{uuid.uuid4().hex}"
        try:
            session.upload_file(  # type: ignore[attr-defined]
                upload_rel,
                _iter_upload_chunks(content),
                sha256=expected_sha256,
            )
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt - 1))
    source_candidates = [
        f"/workspace/{upload_rel}",
        f"/app/{upload_rel}",
    ]
    candidate_tests = " ".join(shlex.quote(candidate) for candidate in source_candidates)
    script = f"""
set -e
src=""
for candidate in {candidate_tests}; do
  if [ -f "$candidate" ]; then
    src="$candidate"
    break
  fi
done
if [ -z "$src" ]; then
  echo "uploaded file not found in sandbox workspace" >&2
  exit 1
fi
mkdir -p "$(dirname -- {shlex.quote(target)})"
cat "$src" > {shlex.quote(target)}
rm -f "$src"
"""
    result = session.execute([{  # type: ignore[attr-defined]
        "name": "upload-copy",
        "command": ["bash", "-lc", script],
        "work_dir": "/app",
    }])
    if not result.results or result.results[0].output.exit_code != 0:
        stderr = result.results[0].output.stderr if result.results else "no result"
        raise RuntimeError(f"failed to place uploaded file at {target}: {stderr}")


def read_file_from_sandbox(session: object, path: str) -> bytes:
    """Read a file from the sandbox's /app tree without using workspace file paths."""
    rel_path = str(PurePosixPath(path.lstrip("/")))
    target = f"/app/{rel_path}"
    result = session.execute([{  # type: ignore[attr-defined]
        "name": "read-file",
        "command": ["bash", "-lc", f"base64 -w0 -- {shlex.quote(target)}"],
        "work_dir": "/app",
    }])
    if not result.results or result.results[0].output.exit_code != 0:
        stderr = result.results[0].output.stderr if result.results else "no result"
        raise FileNotFoundError(stderr or target)
    return base64.b64decode(result.results[0].output.stdout)


def load_trace_tools(session_id: str) -> list[dict]:
    result = subprocess.run(
        ["agentm", "trace", "tools", "--session", session_id, "--format", "ndjson"],
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def _should_skip_replay_bash(cmd: str) -> bool:
    lower = cmd.lower()
    skip = (
        "qemu",
        "make grade",
        "timeout",
        "python3 ok",
        "make test",
        "ctest",
    )
    return any(token in lower for token in skip)


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
                    cur = read_file_from_sandbox(session, rel).decode(
                        "utf-8", errors="replace"
                    )
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
                if cmd and not _should_skip_replay_bash(cmd):
                    session.execute([{  # type: ignore[attr-defined]
                        "name": "r", "command": ["bash", "-lc", cmd],
                        "work_dir": "/app",
                    }])
                    replayed += 1
        except Exception as exc:  # noqa: S110
            logger.debug("Replay step {} failed: {}", tool, exc)
    return replayed


def replay_trajectory(session: object, session_id: str) -> int:
    tools = load_trace_tools(session_id)
    return replay_tools_to_sandbox(session, tools)
