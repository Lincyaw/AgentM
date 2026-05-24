"""Group ``.agentm/observability/*.jsonl`` files by OTel ``trace_id``.

Post the OTel-native unification (commit landing 2026-05-12), every
event row carries a ``trace_id`` that is shared across the entire agent
tree (parent session + spawned extractor / auditor children). This
script:

  * Reads each rollout's ``trace_id`` directly from ``eval.db``
    (``evaluation_data.trace_id`` — populated by the AgentM eval adapter
    since the trace_id forwarding patch landed).
  * Groups all in-window JSONL files by their first row's ``trace_id``.
  * Emits one entry per sample with: the parent span (session.start
    where ``parent_span_id`` is null), all child sessions (everything
    else under the same trace_id, keyed by ``purpose`` for
    extractor / auditor), and a stable list of file paths.

Usage:

    uv run python contrib/scenarios/rca/eval/scripts/build_trace_map.py \\
        --exp-id agentm-rca-opslite-fixed-50-baseline \\
        --out eval-data/agentm-rca-opslite-fixed-50-baseline/trace_map.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from agentm.core.abi import TraceReader


def _first_line_attrs(jsonl_path: Path) -> dict | None:
    """Walk the file for the first ``agentm.session.start`` log record.

    The session.start identity is emitted as a log record with
    ``eventName == "agentm.session.start"``; :class:`TraceReader` walks
    OTLP/JSON ndjson lazily and yields unwrapped attribute dicts. Returns
    a dict shaped to keep the downstream call sites readable
    (``trace_id`` + ``span_id`` + ``attributes`` keys, all derived from
    the OTLP body / attributes).
    """
    if not jsonl_path.is_file():
        return None
    for record in TraceReader(jsonl_path).iter_log_records(
        name="agentm.session.start"
    ):
        body_dict = record.body if isinstance(record.body, dict) else {}
        return {
            "trace_id": (
                body_dict.get("root_session_id")
                or record.attributes.get("agentm.session.root_id")
            ),
            "span_id": (
                body_dict.get("session_id")
                or record.attributes.get("agentm.session.id")
            ),
            "attributes": body_dict,
        }
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp-id", required=True)
    p.add_argument(
        "--observability-dir",
        type=Path,
        default=Path(".agentm/observability"),
    )
    p.add_argument("--db", type=Path, default=Path("eval.db"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    rolled = [
        (source, trace_id)
        for source, trace_id in con.execute(
            "select source, trace_id from evaluation_data "
            "where exp_id=? and response is not null",
            (args.exp_id,),
        )
        if trace_id
    ]
    print(f"DB rollouts with trace_id: {len(rolled)}")
    if not rolled:
        print(
            "no rolled-out samples carry trace_id; either the run pre-dates "
            "the trace_id forwarding patch or eval.db is from a different exp.",
        )
        return 1
    # Look up every session.start in the observability dir; bucket by
    # ``trace_id``. Files with no first-row session.start are skipped
    # (extension-install-only traces from failed spawns, partial writes,
    # etc.).
    trace_to_sessions: dict[str, list[dict]] = defaultdict(list)
    for jf in args.observability_dir.glob("*.jsonl"):
        row = _first_line_attrs(jf)
        if row is None:
            continue
        attrs = row.get("attributes") or {}
        trace_to_sessions[row["trace_id"]].append(
            {
                "session_id": attrs.get("session_id") or row.get("span_id"),
                "parent_session_id": attrs.get("parent_session_id"),
                "purpose": attrs.get("purpose"),
                "scenario": attrs.get("scenario"),
                "path": str(jf),
            }
        )
    print(f"jsonl files scanned: {sum(len(v) for v in trace_to_sessions.values())}")

    mapping = []
    missing: list[str] = []
    for source, trace_id in rolled:
        sessions = trace_to_sessions.get(trace_id, [])
        if not sessions:
            missing.append(source)
            continue
        parent_records = [s for s in sessions if s["parent_session_id"] is None]
        if not parent_records:
            # No row with null parent — pick the session with the same
            # session_id as the one rcabench-platform stored (it's the
            # rollout's root span by construction).
            parent_records = [s for s in sessions if s["session_id"] == trace_id[:16]]
        parent = parent_records[0] if parent_records else {
            "session_id": trace_id,
            "path": str(args.observability_dir / f"{trace_id}.jsonl"),
        }
        children_by_purpose: dict[str, list[dict]] = defaultdict(list)
        for s in sessions:
            if s is parent or s["parent_session_id"] is None:
                continue
            children_by_purpose[s["purpose"] or "unknown"].append(s)

        mapping.append(
            {
                "source": source,
                "trace_id": trace_id,
                "parent_session_id": parent["session_id"],
                "parent_trace_path": parent["path"],
                "extractor_sessions": [
                    s["session_id"] for s in children_by_purpose.get("cognitive_audit_extractor", [])
                ],
                "auditor_sessions": [
                    s["session_id"] for s in children_by_purpose.get("cognitive_audit_auditor", [])
                ],
                "other_children": {
                    k: [s["session_id"] for s in v]
                    for k, v in children_by_purpose.items()
                    if k not in {"cognitive_audit_extractor", "cognitive_audit_auditor"}
                },
                "all_session_paths": [s["path"] for s in sessions],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2))
    print(f"matched {len(mapping)} / {len(rolled)} samples; missing={len(missing)}")
    if missing:
        print("missing:", missing)

    if mapping:
        ex = [len(m["extractor_sessions"]) for m in mapping]
        au = [len(m["auditor_sessions"]) for m in mapping]
        print(
            f"extractor children per sample: min={min(ex)} mean={statistics.mean(ex):.1f} "
            f"max={max(ex)} total={sum(ex)}"
        )
        print(
            f"auditor children per sample:   min={min(au)} mean={statistics.mean(au):.1f} "
            f"max={max(au)} total={sum(au)}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
