# code-health: ignore-file[AM025] -- replaying untyped recorded payloads
"""Offline replay: run the trigger layer over recorded session databases.

This is the calibration bench. Every heuristic change to ``commands`` /
``signals`` / ``triggers`` is accepted by replaying the corpus and reading
the emissions — each firing prints the evidence behind it, so a reviewer
judges emissions, not aggregate counts.

Usage:
    python -m policy_engine replay <db-file-or-sessions-dir> [--min-events N]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .signals import MUTATING_TOOLS, TrajectoryState
from .triggers import StructuralTriggers, load_items


def _load_rows(db_path: Path) -> list[tuple[str, str, int | None, int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT tool_name, args_json, exit_code, turn "
            "FROM policy_tool_events WHERE phase='post' "
            "AND (tool_name = 'bash' OR result_json IS NULL "
            "     OR result_json NOT LIKE '%\"is_error\": true%') "
            "ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def replay_session(db_path: Path, items_path: Path) -> list[str]:
    rows = _load_rows(db_path)
    state = TrajectoryState()
    triggers = StructuralTriggers(items=load_items(items_path))
    emissions: list[str] = []

    def check(*, stopping: bool) -> None:
        firing = triggers.evaluate(state, stopping=stopping)
        if firing is not None:
            facts = "\n".join(f"      {fact}" for fact in firing.facts)
            emissions.append(
                f"  [{firing.item.item_id}] turn={state.turn} "
                f"stopping={stopping}\n{facts}"
            )

    for tool_name, args_json, exit_code, turn in rows:
        state.turn = turn
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            continue
        if tool_name == "bash":
            raw = args.get("cmd")
            if isinstance(raw, str) and raw.strip():
                state.feed_bash(raw, exit_code)
                check(stopping=False)
        elif tool_name in MUTATING_TOOLS:
            path = args.get("path") or args.get("file_path")
            if isinstance(path, str) and path:
                state.feed_mutation(path)

    check(stopping=True)  # the stop decision
    for fact in triggers.critic_evidence(state):
        emissions.append(f"  [critic-evidence] {fact}")
    return emissions


def main() -> int:
    parser = argparse.ArgumentParser(prog="policy_engine")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="replay recorded sessions")
    replay.add_argument("target", type=Path)
    replay.add_argument("--min-events", type=int, default=20)
    args = parser.parse_args()

    items_path = Path(__file__).parent / "checklist.yaml"
    targets = (
        sorted(args.target.glob("*.db")) if args.target.is_dir() else [args.target]
    )
    for db_path in targets:
        rows = _load_rows(db_path)
        if len(rows) < args.min_events:
            continue
        emissions = replay_session(db_path, items_path)
        status = f"{len(emissions)} emission(s)" if emissions else "silent"
        print(f"{db_path.stem}: {status}")
        for emission in emissions:
            print(emission)
    return 0


if __name__ == "__main__":
    sys.exit(main())
