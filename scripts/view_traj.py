#!/usr/bin/env python3
"""Browse fixed-50 trajectories by exp_id.

Bridges `eval.db.evaluation_data.exp_id` ↔ `evaluation_data.trace_id`
↔ `.agentm/observability/<root_span_id>.jsonl`. Optimised for "I just
ran a fixed-50 batch; help me eyeball what audit produced."

Sub-commands
------------
  list   <exp_id> [--stage judged|init]
         One row per case: trace_id, source, correct, root file.

  files  <trace_id>
         Every session jsonl that shares this trace_id (root + audit
         children), with purpose tag.

  main   <trace_id>
         Main agent's per-turn tool sequence + the final
         `submit_final_report` arguments if present.

  audit  <trace_id> [--phase extractor|auditor]
         For each audit child: turn count, tools called, and the
         submit_events / submit_verdict payload (truncated to 4 KB).

  judge  <trace_id>
         eval.db judge verdict (correct / reasoning / extracted_final_answer).

  diag   <trace_id>
         Any `emit:diagnostic source=llmharness.audit` spans across the
         whole trace — surfaces the silent-fail paths that the
         _record_failure → DiagnosticEvent leg now exposes.

Example
-------
    # newest experiments first
    scripts/view_traj.py list agentm-rca-opslite-fixed-50-ops-fix
    # pick a trace, see what the audit child actually said
    scripts/view_traj.py audit ee407c24913c4e93a6acf8bf022cbb7b
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / ".agentm" / "observability"
DB = ROOT / "eval.db"
TRUNC = 4096


def _conn() -> sqlite3.Connection:
    if not DB.exists():
        sys.exit(f"eval.db not found at {DB}")
    return sqlite3.connect(DB)


def _trace_files(trace_id: str) -> list[tuple[str, Path]]:
    """Return [(purpose, path)] for every jsonl whose first line carries
    this root_session_id. O(n) over the obs dir; cheap enough for now."""
    out: list[tuple[str, Path]] = []
    for f in OBS.glob("*.jsonl"):
        try:
            with f.open() as fh:
                first = json.loads(fh.readline())
        except Exception:
            continue
        a = first.get("attributes") or {}
        if a.get("root_session_id") != trace_id:
            continue
        out.append((a.get("purpose") or "main", f))
    out.sort(key=lambda x: (x[0] != "root", x[1].stat().st_mtime))
    return out


def _is_root(path: Path) -> bool:
    try:
        with path.open() as fh:
            first = json.loads(fh.readline())
    except Exception:
        return False
    return (first.get("attributes") or {}).get("purpose") == "root"


def cmd_list(args: argparse.Namespace) -> None:
    con = _conn()
    rows = con.execute(
        "select trace_id, source, correct, extracted_final_answer "
        "from evaluation_data where exp_id=? and stage=? "
        "order by id",
        (args.exp_id, args.stage),
    ).fetchall()
    if not rows:
        sys.exit(f"no rows for exp_id={args.exp_id} stage={args.stage}")
    correct = sum(1 for r in rows if r[2])
    print(f"exp_id={args.exp_id}  stage={args.stage}  cases={len(rows)}  "
          f"correct={correct} ({correct / len(rows):.0%})")
    print(f"{'trace_id':>34}  {'correct':>7}  {'source':<40}  root_file")
    for trace_id, source, ok, _ans in rows:
        root_file = ""
        if trace_id:
            for p, f in _trace_files(trace_id):
                if p == "root":
                    root_file = f.name
                    break
        ok_str = "Y" if ok else ("-" if ok is None else "N")
        print(f"{trace_id or '<missing>':>34}  {ok_str:>7}  {(source or '')[:40]:<40}  {root_file}")


def cmd_files(args: argparse.Namespace) -> None:
    files = _trace_files(args.trace_id)
    if not files:
        sys.exit(f"no jsonl found for trace_id={args.trace_id}")
    by = Counter(p for p, _ in files)
    print(f"trace_id={args.trace_id}  total={len(files)}  by_purpose={dict(by)}")
    for purpose, path in files:
        print(f"  {purpose:30}  {path.name}  ({path.stat().st_size:>9} B)")


def _read_root(trace_id: str) -> Path:
    for p, f in _trace_files(trace_id):
        if p == "root":
            return f
    sys.exit(f"no root session jsonl for trace_id={trace_id}")


def cmd_main(args: argparse.Namespace) -> None:
    root = _read_root(args.trace_id)
    print(f"# root file: {root}")
    final_args: dict | None = None
    for line in root.open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("kind") == "turn.summary":
            a = d["attributes"]
            tools = a.get("tool_calls") or []
            print(f"t{a['turn_index']:02d} ({a['duration_ns'] / 1e9:5.1f}s, "
                  f"in={a.get('input_tokens', 0):>5} out={a.get('output_tokens', 0):>4}): {tools}")
        if d.get("kind") == "event.dispatch" and "emit:tool_call" in (d.get("name") or ""):
            ev = (d.get("attributes") or {}).get("event") or {}
            if ev.get("tool_name") in ("submit_final_report", "finalize", "submit_answer"):
                final_args = ev.get("args")
    if final_args is not None:
        print("\n# final tool args:")
        print(json.dumps(final_args, ensure_ascii=False, indent=2, default=str)[:TRUNC])


def _audit_children(trace_id: str, phase: str | None) -> list[Path]:
    files = _trace_files(trace_id)
    keep = []
    for purpose, f in files:
        if not purpose.startswith("cognitive_audit_"):
            continue
        if phase == "extractor" and purpose != "cognitive_audit_extractor":
            continue
        if phase == "auditor" and purpose != "cognitive_audit_auditor":
            continue
        keep.append(f)
    return keep


def cmd_audit(args: argparse.Namespace) -> None:
    files = _audit_children(args.trace_id, args.phase)
    if not files:
        sys.exit(f"no audit children for trace_id={args.trace_id} phase={args.phase}")
    for f in files:
        purpose = ""
        llm = 0
        tools: list[str] = []
        submit_args: list[tuple[str, dict]] = []
        ended = False
        for line in f.open():
            try:
                d = json.loads(line)
            except Exception:
                continue
            k = d.get("kind") or ""
            n = d.get("name") or ""
            a = d.get("attributes") or {}
            if k == "session.start":
                purpose = a.get("purpose", "")
            if k == "llm.request.start":
                llm += 1
            if k == "session.end":
                ended = True
            if k == "event.dispatch" and "emit:tool_call" in n:
                ev = a.get("event") or {}
                tn = ev.get("tool_name") or "?"
                tools.append(tn)
                if tn in ("submit_events", "submit_verdict"):
                    submit_args.append((tn, ev.get("args") or {}))
        print(f"\n=== {f.name}  purpose={purpose}  turns={llm}  ended={ended} ===")
        print(f"    tools: {tools}")
        for tn, payload in submit_args:
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            if len(text) > TRUNC:
                text = text[:TRUNC] + f"\n... [truncated, full {len(text)} chars]"
            print(f"    --- {tn} args ---\n{text}")


def cmd_judge(args: argparse.Namespace) -> None:
    con = _conn()
    row = con.execute(
        "select exp_id, source, correct, extracted_final_answer, reasoning, eval_metrics "
        "from evaluation_data where trace_id=? and stage='judged'",
        (args.trace_id,),
    ).fetchone()
    if row is None:
        sys.exit(f"no judged row for trace_id={args.trace_id}")
    exp_id, source, correct, ans, reasoning, metrics = row
    print(f"exp_id={exp_id}")
    print(f"source={source}")
    print(f"correct={correct}")
    print(f"extracted_final_answer:\n{ans}\n")
    print(f"reasoning:\n{reasoning}\n")
    print(f"eval_metrics:\n{metrics}")


def cmd_diag(args: argparse.Namespace) -> None:
    files = _trace_files(args.trace_id)
    if not files:
        sys.exit(f"no jsonl found for trace_id={args.trace_id}")
    hits = 0
    for _, f in files:
        for line in f.open():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("kind") != "event.dispatch":
                continue
            if "emit:diagnostic" not in (d.get("name") or ""):
                continue
            ev = (d.get("attributes") or {}).get("event") or {}
            if ev.get("source") != "llmharness.audit":
                continue
            hits += 1
            print(f"[{f.name}] {ev.get('level'):8} {ev.get('message')}")
    print(f"\n# total llmharness.audit diagnostics: {hits}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list cases for an exp_id")
    s.add_argument("exp_id")
    s.add_argument("--stage", default="judged", choices=["judged", "init", "rolled_out"])
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("files", help="all jsonl files in a trace")
    s.add_argument("trace_id")
    s.set_defaults(func=cmd_files)

    s = sub.add_parser("main", help="main agent tool sequence")
    s.add_argument("trace_id")
    s.set_defaults(func=cmd_main)

    s = sub.add_parser("audit", help="audit children + submit payloads")
    s.add_argument("trace_id")
    s.add_argument("--phase", choices=["extractor", "auditor"], default=None)
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("judge", help="eval.db judge verdict")
    s.add_argument("trace_id")
    s.set_defaults(func=cmd_judge)

    s = sub.add_parser("diag", help="llmharness.audit diagnostic spans")
    s.add_argument("trace_id")
    s.set_defaults(func=cmd_diag)

    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    ns.func(ns)
