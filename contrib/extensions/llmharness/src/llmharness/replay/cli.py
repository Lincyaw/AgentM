"""``llmharness-replay`` — replay one captured phase invocation.

Subcommands
-----------
* ``extractor`` — rerun an extractor record; prints submit_events args.
* ``auditor``  — rerun an auditor record; prints submit_verdict args.
* ``list``     — index a sidecar file by phase / turn / status / latency.

Each subcommand supports ``--diff`` to compare replay output against the
recorded output side-by-side, ``--provider`` to swap LLM, and
``--prompt-override`` to swap system prompt — the three knobs we want
when bisecting an auditor regression.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .engine import PhaseResult
from .record import ReplayRecord, iter_records, read_records
from .runner import replay_auditor_record, replay_extractor_record


def _parse_provider(spec: str | None) -> tuple[str, dict[str, Any]] | None:
    """``module`` or ``module:json_config`` → coerced provider tuple."""
    if not spec:
        return None
    if ":" in spec:
        module, payload = spec.split(":", 1)
        try:
            cfg = json.loads(payload)
        except json.JSONDecodeError as exc:
            sys.exit(f"--provider config is not valid JSON: {exc}")
        if not isinstance(cfg, dict):
            sys.exit("--provider config JSON must be an object")
        return module.strip(), cfg
    return spec.strip(), {}


def _load_prompt_override(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        sys.exit(f"--prompt-override file not found: {p}")
    return p.read_text(encoding="utf-8")


def _pick_record(
    path: Path, *, phase: str, turn: int | None, index: int | None
) -> ReplayRecord:
    records = read_records(path, phase=phase)  # type: ignore[arg-type]
    if not records:
        sys.exit(f"no {phase} records in {path}")
    if turn is not None:
        candidates = [r for r in records if r.turn_index == turn]
        if not candidates:
            sys.exit(f"no {phase} record with turn_index={turn} in {path}")
        return candidates[-1]
    if index is not None:
        try:
            return records[index]
        except IndexError:
            sys.exit(f"--index {index} out of range (0..{len(records) - 1})")
    return records[-1]


def _print_result(result: PhaseResult, recorded: ReplayRecord, *, diff: bool) -> None:
    print(f"# status:      {result.status}")
    print(f"# latency_ms:  {result.latency_ms}")
    if result.error:
        print(f"# error:       {result.error}")
    if result.output is not None:
        print("\n--- replay output ---")
        print(json.dumps(result.output, ensure_ascii=False, indent=2, default=str))
    if diff:
        print("\n--- recorded output ---")
        print(
            json.dumps(recorded.output or {}, ensure_ascii=False, indent=2, default=str)
        )


def _cmd_extractor(args: argparse.Namespace) -> None:
    record = _pick_record(
        Path(args.record), phase="extractor", turn=args.turn, index=args.index
    )
    result = asyncio.run(
        replay_extractor_record(
            record,
            cwd=str(args.cwd),
            provider_override=_parse_provider(args.provider),
            prompt_override=_load_prompt_override(args.prompt_override),
        )
    )
    _print_result(result, record, diff=args.diff)


def _cmd_auditor(args: argparse.Namespace) -> None:
    record = _pick_record(
        Path(args.record), phase="auditor", turn=args.turn, index=args.index
    )
    result = asyncio.run(
        replay_auditor_record(
            record,
            cwd=str(args.cwd),
            provider_override=_parse_provider(args.provider),
            prompt_override=_load_prompt_override(args.prompt_override),
        )
    )
    _print_result(result, record, diff=args.diff)


def _cmd_list(args: argparse.Namespace) -> None:
    rows: list[tuple[int, str, int, str, int]] = []
    for i, rec in enumerate(iter_records(Path(args.record))):
        if args.phase and rec.phase != args.phase:
            continue
        rows.append((i, rec.phase, rec.turn_index, rec.status, rec.latency_ms))
    if not rows:
        sys.exit(f"no matching records in {args.record}")
    print(f"{'idx':>4}  {'phase':<10}  {'turn':>4}  {'status':<12}  {'latency_ms':>10}")
    for i, phase, turn, status, latency in rows:
        print(f"{i:>4}  {phase:<10}  {turn:>4}  {status:<12}  {latency:>10}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmharness-replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common_picker_args: list[tuple[tuple[str, ...], dict[str, Any]]] = [
        (("--record",), {"required": True, "help": "Path to audit_replay/<id>.jsonl"}),
        (("--turn",), {"type": int, "default": None, "help": "Pick by turn_index (last match wins)"}),
        (("--index",), {"type": int, "default": None, "help": "Pick by 0-based filtered index"}),
        (("--cwd",), {"default": ".", "help": "Working directory for the replay session"}),
        (("--provider",), {"default": None, "help": "module or module:{json_cfg}"}),
        (("--prompt-override",), {"default": None, "help": "Path to system-prompt override"}),
        (("--diff",), {"action": "store_true", "help": "Also print recorded output"}),
    ]

    s = sub.add_parser("extractor", help="rerun an extractor record")
    for flags, opts in common_picker_args:
        s.add_argument(*flags, **opts)
    s.set_defaults(func=_cmd_extractor)

    s = sub.add_parser("auditor", help="rerun an auditor record")
    for flags, opts in common_picker_args:
        s.add_argument(*flags, **opts)
    s.set_defaults(func=_cmd_auditor)

    s = sub.add_parser("list", help="enumerate records in a sidecar file")
    s.add_argument("--record", required=True)
    s.add_argument("--phase", choices=["extractor", "auditor"], default=None)
    s.set_defaults(func=_cmd_list)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
