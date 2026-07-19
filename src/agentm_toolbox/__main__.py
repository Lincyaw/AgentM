"""CLI entry point for sandbox execution.

Usage::

    python3 -m agentm_toolbox read '{"path": "/repo/file.py", "offset": 10}'
    printf '%s' '{"path": "/repo/file.py"}' | python3 -m agentm_toolbox read -
    python3 -m agentm_toolbox write '{"path": "/repo/file.py", "content_file": "/tmp/.upload"}'
    python3 -m agentm_toolbox edit '{"path": "/repo/file.py", "old_string": "x", "new_string": "y"}'

State is persisted to ``/tmp/.agentm-toolbox/state.json`` between
invocations so read-before-write guards work across exec calls.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

from agentm_toolbox._file_ops import FileToolbox
from agentm_toolbox._state import ReadStateStore

_STATE_FILE = "/tmp/.agentm-toolbox/state.json"


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"text": "usage: agentm_toolbox <read|write|edit> [json_args]", "is_error": True}))
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

    state = ReadStateStore.load_from(_STATE_FILE)
    cwd = args.pop("_cwd", os.getcwd())
    toolbox = FileToolbox(
        cwd=cwd,
        max_size=args.pop("_max_size", 262_144),
        require_read=args.pop("_require_read", True),
        default_limit=args.pop("_default_limit", 250),
        state=state,
    )

    if tool_name == "write" and "content_file" in args:
        cf = args.pop("content_file")
        try:
            args["content"] = open(cf).read()
        except Exception as exc:
            print(json.dumps({"text": f"Failed to read content_file {cf!r}: {exc}", "is_error": True}))
            sys.exit(1)
        finally:
            try:
                os.unlink(cf)
            except OSError:
                pass

    fn = getattr(toolbox, tool_name, None)
    if fn is None:
        print(json.dumps({"text": f"unknown tool: {tool_name!r}", "is_error": True}))
        sys.exit(1)

    result = fn(**args)
    toolbox.state.save_to(_STATE_FILE)
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
