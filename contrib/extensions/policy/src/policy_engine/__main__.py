# code-health: ignore-file[AM025] -- replaying untyped recorded payloads
"""Offline analysis over recorded sessions, through the data plane.

Two commands, both consuming the SAME plane the live watcher uses:

    replay <db-or-dir>   re-derive facts per session, walk the trajectory
                         and print every trigger emission with evidence —
                         the calibration bench for any heuristic change
    query <db> "<sql>"   the unified ad-hoc query entry over the fact
                         schema (plane_* tables, v_* views)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .plane import DataPlane
from .triggers import TriggerEngine, load_items


def replay_session(db_path: Path, items_path: Path) -> list[str]:
    plane = DataPlane.snapshot(db_path)
    rows = plane.fetch_raw()
    triggers = TriggerEngine(items=load_items(items_path))
    emissions: list[str] = []

    def check(prefix_end: int, *, stopping: bool, turn: int) -> None:
        plane.rebuild(rows[:prefix_end])
        firing = triggers.evaluate_inject(plane, stopping=stopping)
        if firing is not None:
            facts = "\n".join(f"      {fact}" for fact in firing.facts)
            emissions.append(
                f"  [{firing.item.item_id}] turn={turn} stopping={stopping}\n{facts}"
            )

    for index, row in enumerate(rows):
        if row[2] == "bash":
            check(index + 1, stopping=False, turn=row[1])

    plane.rebuild(rows)
    final_turn = rows[-1][1] if rows else 0
    check(len(rows), stopping=True, turn=final_turn)
    open_items = triggers.open_critic_items(plane)
    if open_items:
        ids = ", ".join(item.item_id for item in open_items)
        emissions.append(f"  [critic-gate-open x{len(open_items)}] {ids}")
    for fact in triggers.critic_evidence(plane):
        emissions.append(f"  [critic-evidence] {fact}")
    plane.close()
    return emissions


def main() -> int:
    parser = argparse.ArgumentParser(prog="policy_engine")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="replay recorded sessions")
    replay.add_argument("target", type=Path)
    replay.add_argument("--min-events", type=int, default=20)
    query = sub.add_parser("query", help="ad-hoc SQL over the fact schema")
    query.add_argument("db", type=Path)
    query.add_argument("sql")
    args = parser.parse_args()

    items_path = Path(__file__).parent / "checklist.yaml"

    if args.command == "query":
        plane = DataPlane.snapshot(args.db)
        plane.rebuild()
        for row in plane.query(args.sql):
            print("|".join(str(value) for value in row))
        plane.close()
        return 0

    targets = (
        sorted(args.target.glob("*.db")) if args.target.is_dir() else [args.target]
    )
    for db_path in targets:
        probe = DataPlane.snapshot(db_path)
        row_count = len(probe.fetch_raw())
        probe.close()
        if row_count < args.min_events:
            continue
        emissions = replay_session(db_path, items_path)
        status = f"{len(emissions)} emission(s)" if emissions else "silent"
        print(f"{db_path.stem}: {status}")
        for emission in emissions:
            print(emission)
    return 0


if __name__ == "__main__":
    sys.exit(main())
