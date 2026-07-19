"""Subprocess worker for :mod:`agentm.execution.process`."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print("usage: python -m agentm.execution.worker module:function args.json result.pickle", file=sys.stderr)
        return 2
    entrypoint, args_path, result_path = args
    try:
        result = asyncio.run(_run(entrypoint, Path(args_path)))
        with Path(result_path).open("wb") as handle:
            pickle.dump(result, handle)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"agentm tool worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


async def _run(entrypoint: str, args_path: Path) -> Any:
    module_name, _, function_name = entrypoint.partition(":")
    if not module_name or not function_name:
        raise ValueError("entrypoint must be 'module:function'")
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    args = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(args, dict):
        raise ValueError("tool args must be a JSON object")
    result = fn(args)
    if inspect.isawaitable(result):
        return await result
    return result


if __name__ == "__main__":
    raise SystemExit(main())
