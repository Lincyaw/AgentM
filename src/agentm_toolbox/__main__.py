"""CLI entry point for sandbox execution.

Usage::

    python3 -m agentm_toolbox read '{"path": "/repo/file.py", "offset": 10}'
    printf '%s' '{"path": "/repo/file.py"}' | python3 -m agentm_toolbox read -
    python3 -m agentm_toolbox write '{"path": "/repo/file.py", "content_file": "/tmp/.upload"}'
    python3 -m agentm_toolbox edit '{"path": "/repo/file.py", "old_string": "x", "new_string": "y"}'

State is persisted under ``/tmp/.agentm-toolbox/`` between invocations so
read-before-write guards work across exec calls. Hosts should pass
``_state_namespace`` or ``_state_file`` to isolate independent sessions and
environments that share one toolbox installation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
import sys
from typing import Iterator

from agentm_toolbox._file_ops import FileToolbox
from agentm_toolbox._state import ReadStateStore

_STATE_DIR = "/tmp/.agentm-toolbox"


def _state_file(
    *,
    state_file: str | None,
    state_namespace: str | None,
    cwd: str,
) -> str:
    if state_file:
        return state_file
    namespace = (
        state_namespace
        or os.environ.get("AGENTM_TOOLBOX_STATE_NAMESPACE")
        or f"cwd:{os.path.abspath(cwd)}"
    )
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    state_dir = os.environ.get("AGENTM_TOOLBOX_STATE_DIR", _STATE_DIR)
    return os.path.join(state_dir, f"{digest}.json")


@contextmanager
def _locked_state(path: str) -> Iterator[None]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lock_path = f"{path}.lock"
    with open(lock_path, "a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise RuntimeError(f"cannot lock toolbox state {path!r}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "text": "usage: agentm_toolbox <read|write|edit> [json_args]",
                    "is_error": True,
                }
            )
        )
        sys.exit(1)

    tool_name = sys.argv[1]
    raw_args = sys.argv[2] if len(sys.argv) > 2 else "{}"
    if raw_args == "-":
        raw_args = sys.stdin.read()
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        print(json.dumps({"text": f"invalid JSON args: {exc}", "is_error": True}))
        sys.exit(1)
    if not isinstance(args, dict):
        print(json.dumps({"text": "tool args must be a JSON object", "is_error": True}))
        sys.exit(1)

    state_file = args.pop("_state_file", os.environ.get("AGENTM_TOOLBOX_STATE_FILE"))
    state_namespace = args.pop("_state_namespace", None)
    cwd = args.pop("_cwd", os.getcwd())
    state_path = _state_file(
        state_file=state_file,
        state_namespace=state_namespace,
        cwd=cwd,
    )

    if tool_name == "write" and "content_file" in args:
        cf = args.pop("content_file")
        try:
            with open(cf, encoding="utf-8") as handle:
                args["content"] = handle.read()
            os.unlink(cf)
        except (OSError, UnicodeError) as exc:
            print(
                json.dumps(
                    {
                        "text": f"Failed to consume content_file {cf!r}: {exc}",
                        "is_error": True,
                    }
                )
            )
            sys.exit(1)

    with _locked_state(state_path):
        state = ReadStateStore.load_from(state_path)
        toolbox = FileToolbox(
            cwd=cwd,
            max_size=args.pop("_max_size", 262_144),
            require_read=args.pop("_require_read", True),
            default_limit=args.pop("_default_limit", 250),
            state=state,
        )
        fn = getattr(toolbox, tool_name, None)
        if fn is None:
            print(
                json.dumps(
                    {"text": f"unknown tool: {tool_name!r}", "is_error": True}
                )
            )
            sys.exit(1)

        result = fn(**args)
        toolbox.state.save_to(state_path)
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
