"""Stamp an extra ``tags`` entry onto rcabench-platform's ``data`` table so
a fixed sample set can be re-evaluated by tag alone.

Why: the AgentM rca:harness.sync baseline was measured on the first 50
ops-lite samples; future strategy iterations need to compare against
the same 50 cases. ``rca llm-eval`` filters samples by ``data.tags``
(JSON array; OR-matched), so we add a unique tag (``ops-lite-fixed-50``
by default) to exactly those 50 rows and reuse it from the eval config.

The tag write is idempotent: rows that already carry the tag are left
untouched, and rows whose ``source`` is not in the input list have any
prior copies of the tag removed (so the on-disk fixture file is the
single source of truth — drift between file and DB is impossible).

Usage::

    uv run python contrib/scenarios/rca/eval/scripts/retag_fixed_set.py \\
        --fixture contrib/scenarios/rca/eval/fixtures/ops-lite-fixed-50.txt \\
        --dataset ops-lite \\
        --tag ops-lite-fixed-50

Pass ``--dry-run`` to print what would change without writing.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _read_fixture(path: Path) -> list[str]:
    sources: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(line)
    if not sources:
        raise SystemExit(f"fixture {path} is empty")
    return sources


def _ensure_tag(tags_json: str | None, tag: str) -> tuple[str, bool]:
    """Return (new_json, changed) — append ``tag`` if absent."""

    if tags_json:
        try:
            tags = json.loads(tags_json)
        except json.JSONDecodeError:
            tags = []
    else:
        tags = []
    if not isinstance(tags, list):
        tags = []
    if tag in tags:
        return json.dumps(tags), False
    tags.append(tag)
    return json.dumps(tags), True


def _remove_tag(tags_json: str | None, tag: str) -> tuple[str | None, bool]:
    """Return (new_json, changed) — drop ``tag`` if present."""

    if not tags_json:
        return tags_json, False
    try:
        tags = json.loads(tags_json)
    except json.JSONDecodeError:
        return tags_json, False
    if not isinstance(tags, list) or tag not in tags:
        return tags_json, False
    tags = [t for t in tags if t != tag]
    return json.dumps(tags), True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", default="eval.db", type=Path)
    p.add_argument("--dataset", default="ops-lite")
    p.add_argument(
        "--fixture",
        default=Path("contrib/scenarios/rca/eval/fixtures/ops-lite-fixed-50.txt"),
        type=Path,
    )
    p.add_argument("--tag", default="ops-lite-fixed-50")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    expected = set(_read_fixture(args.fixture))
    con = sqlite3.connect(args.db)
    rows = con.execute(
        "select id, source, tags from data where dataset=?", (args.dataset,)
    ).fetchall()
    rows_by_source = {source: (rid, tags) for rid, source, tags in rows}

    missing = sorted(expected - set(rows_by_source))
    if missing:
        print(
            f"ERROR: fixture references {len(missing)} sources not present in "
            f"data.dataset={args.dataset!r}: {missing[:3]}{'...' if len(missing) > 3 else ''}",
            file=sys.stderr,
        )
        return 2

    add: list[tuple[int, str]] = []
    drop: list[tuple[int, str | None]] = []
    for source, (rid, tags) in rows_by_source.items():
        if source in expected:
            new_json, changed = _ensure_tag(tags, args.tag)
            if changed:
                add.append((rid, new_json))
        else:
            new_json, changed = _remove_tag(tags, args.tag)
            if changed:
                drop.append((rid, new_json))

    print(f"will tag {len(add)} rows with {args.tag!r}; will untag {len(drop)} rows")
    if args.dry_run:
        return 0
    if add:
        con.executemany("update data set tags=? where id=?", [(t, i) for i, t in add])
    if drop:
        con.executemany("update data set tags=? where id=?", [(t, i) for i, t in drop])
    con.commit()
    final = con.execute(
        "select count(*) from data where dataset=? and tags like ?",
        (args.dataset, f'%"{args.tag}"%'),
    ).fetchone()[0]
    print(f"final: {final} rows in {args.dataset} carry tag {args.tag!r}")
    if final != len(expected):
        print(
            f"WARNING: expected {len(expected)} tagged rows, got {final}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
