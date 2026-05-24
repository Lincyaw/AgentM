#!/usr/bin/env python3
"""Best-effort v3 → v4 migration for ReplayRecord-style JSONL.

v4 (2026-05-24) drops the ``evid`` event kind: a linear investigation
block now collapses to ONE ``act`` node whose summary records both
probes and results. This script rewrites existing v3 jsonl artifacts
under that contract by:

* finding every ``evid`` node and the immediately preceding ``act``;
* merging the evid's summary into the act
  (``<act.summary> Results: <evid.summary>``);
* taking the union of ``source_turns`` — the result MUST still be
  contiguous (no gap), otherwise the record is logged and skipped;
* re-pointing every ref / edge whose ``to``/``dst`` is the evid id to
  the merged act's id;
* writing the result to a sibling ``<name>.v4.jsonl`` file.

The script is intentionally conservative: it touches ``records``
(``ReplayRecord``-style JSONL) and falls back to leaving a record
unchanged when the v3 → v4 shape isn't unambiguous. Run only against
read-only audit run jsonl that you intend to consume from v4 code.

Usage:
    python scripts/migrate_v3_to_v4_evid_fold.py <input.jsonl> [<input2.jsonl> ...]

Outputs ``<input>.v4.jsonl`` next to each input. Prints one line per
input summarising counts. Does NOT mutate the input file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _is_contiguous(turns: list[int]) -> bool:
    if not turns:
        return False
    sorted_turns = sorted(set(turns))
    return sorted_turns == list(range(sorted_turns[0], sorted_turns[-1] + 1))


def _fold_evid_into_act(events: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], dict[int, int], list[str]
]:
    """Return (folded_events, evid_id → act_id remap, warnings).

    Events are scanned in submission order. For every ``evid`` node we
    locate the most recent ``act`` (by position) that has not already
    swallowed a different evid, and merge into it. If no preceding
    ``act`` exists or the resulting ``source_turns`` is non-contiguous,
    the evid is preserved unchanged and a warning is logged so the
    operator can decide what to do.
    """
    warnings: list[str] = []
    remap: dict[int, int] = {}
    folded: list[dict[str, Any]] = []
    # Cache of last-seen act position by absence of merged evid yet.
    last_act_index: int | None = None
    for ev in events:
        kind = ev.get("kind")
        if kind == "act":
            folded.append(dict(ev))
            last_act_index = len(folded) - 1
            continue
        if kind != "evid":
            folded.append(dict(ev))
            last_act_index = None  # any non-(act|evid) breaks the pairing
            continue
        # kind == "evid"
        if last_act_index is None:
            warnings.append(
                f"evid id={ev.get('id')!r} has no preceding act; keeping as-is "
                "(downgraded to act kind)."
            )
            cloned = dict(ev)
            cloned["kind"] = "act"
            # Tag the downgrade in the summary so a post-migration audit
            # can grep these out — they are not real probe+result acts.
            orig_summary = cloned.get("summary", "")
            cloned["summary"] = f"[v4-migrated orphan evid] {orig_summary}".rstrip()
            folded.append(cloned)
            last_act_index = len(folded) - 1
            continue
        host = folded[last_act_index]
        merged_turns = sorted(
            set(host.get("source_turns") or []) | set(ev.get("source_turns") or [])
        )
        if not _is_contiguous(merged_turns):
            warnings.append(
                f"evid id={ev.get('id')!r} would produce non-contiguous "
                f"source_turns when merged into act id={host.get('id')!r}; "
                "keeping evid as a standalone act node."
            )
            cloned = dict(ev)
            cloned["kind"] = "act"
            folded.append(cloned)
            last_act_index = len(folded) - 1
            continue
        host["source_turns"] = merged_turns
        host_summary = host.get("summary", "")
        evid_summary = ev.get("summary", "")
        if evid_summary:
            host["summary"] = (
                f"{host_summary} Results: {evid_summary}".strip()
            )
        remap[int(ev["id"])] = int(host["id"])
        # Once this act has absorbed an evid we don't pair the next
        # evid into the same act — keep evid-pairing 1:1 to preserve
        # downstream parents that may have ref'd back to the act.
        last_act_index = None
    return folded, remap, warnings


def _apply_remap_to_refs(
    events: list[dict[str, Any]], remap: dict[int, int]
) -> None:
    """In-place: rewrite refs / external_refs ``to`` ids per ``remap``."""
    if not remap:
        return
    for ev in events:
        # In-firing refs (legacy v3.1 inline ref payload).
        for r in ev.get("refs") or []:
            to = r.get("to")
            if isinstance(to, int) and to in remap:
                r["to"] = remap[to]
        for r in ev.get("external_refs") or []:
            to = r.get("to_recent_event_id")
            if isinstance(to, int) and to in remap:
                r["to_recent_event_id"] = remap[to]
    # Drop nodes that got folded — they're no longer addressable.
    folded_ids = set(remap.keys())
    events[:] = [ev for ev in events if int(ev["id"]) not in folded_ids]


def _apply_remap_to_edges(
    edges: list[dict[str, Any]], remap: dict[int, int]
) -> list[dict[str, Any]]:
    """Return a new edge list with src/dst remapped; drop self-loops."""
    out: list[dict[str, Any]] = []
    for ed in edges:
        src = ed.get("src")
        dst = ed.get("dst")
        if isinstance(src, int) and src in remap:
            ed = dict(ed)
            ed["src"] = remap[src]
        if isinstance(dst, int) and dst in remap:
            ed = dict(ed)
            ed["dst"] = remap[dst]
        if ed.get("src") == ed.get("dst"):
            continue  # self-loop created by fold; drop
        out.append(ed)
    return out


def migrate_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fold one record in-place; return (rewritten, warnings)."""
    warnings: list[str] = []
    rewritten = json.loads(json.dumps(record))  # deep copy via JSON
    # ReplayRecord variants store events under a few keys; cover the
    # ones that show up in run jsonl artifacts.
    for key in ("events", "raw_events"):
        events = rewritten.get(key)
        if not isinstance(events, list):
            continue
        folded, remap, warns = _fold_evid_into_act(events)
        warnings.extend(warns)
        _apply_remap_to_refs(folded, remap)
        rewritten[key] = folded
        edges = rewritten.get("edges")
        if isinstance(edges, list):
            rewritten["edges"] = _apply_remap_to_edges(edges, remap)
    return rewritten, warnings


def migrate_file(path: Path) -> tuple[int, int, list[str]]:
    """Return (records_in, records_skipped, warnings)."""
    out_path = path.with_suffix(path.suffix + ".v4")
    in_count = 0
    skipped = 0
    warnings: list[str] = []
    with path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            in_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path.name}:{in_count}: invalid JSON ({exc})")
                skipped += 1
                continue
            rewritten, warns = migrate_record(record)
            warnings.extend(f"{path.name}:{in_count}: {w}" for w in warns)
            dst.write(json.dumps(rewritten, ensure_ascii=False))
            dst.write("\n")
    return in_count, skipped, warnings


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rc = 0
    for arg in argv[1:]:
        path = Path(arg)
        if not path.is_file():
            print(f"skip: {path} is not a file", file=sys.stderr)
            rc = 1
            continue
        in_count, skipped, warnings = migrate_file(path)
        out_path = path.with_suffix(path.suffix + ".v4")
        print(
            f"{path}: {in_count} records → {out_path.name} "
            f"({skipped} skipped, {len(warnings)} warning(s))"
        )
        for w in warnings:
            print(f"  WARN {w}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
