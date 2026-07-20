"""Subprocess worker for :mod:`agentm.execution.process`."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path

from agentm.core.abi.tool import ToolOutcome, ToolResult
from agentm.execution.wire import decode_tool_arguments, encode_tool_output


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: python -m agentm.execution.worker "
            "module:function args.json result.json",
            file=sys.stderr,
        )
        return 2
    entrypoint, args_path, result_path = args
    try:
        result = asyncio.run(_run(entrypoint, Path(args_path)))
        Path(result_path).write_text(encode_tool_output(result), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"agentm tool worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


async def _run(
    entrypoint: str,
    args_path: Path,
) -> ToolResult | ToolOutcome:
    module_name, _, function_name = entrypoint.partition(":")
    if not module_name or not function_name:
        raise ValueError("entrypoint must be 'module:function'")
    module = importlib.import_module(module_name)
    fn = module.__dict__.get(function_name)
    if not callable(fn):
        raise TypeError(f"process entrypoint is not callable: {entrypoint}")
    args = decode_tool_arguments(args_path.read_text(encoding="utf-8"))
    result = fn(args)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, (ToolResult, ToolOutcome)):
        raise TypeError(
            "process entrypoint must return ToolResult or ToolOutcome, got "
            f"{type(result).__name__}"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
