"""``llmharness review`` — read-only inspector for the cognitive-audit graph.

The graph and verdicts the adapter writes live as custom entries inside the
main AgentM session JSONL (``~/.agentm/sessions/--<cwd-slug>--/*.jsonl``),
intermixed with messages. This CLI extracts and pretty-prints them so you
can audit a run without writing ad-hoc ``jq`` queries.

It is read-only — it never imports the AgentM harness or mutates the
session file. The session file format is the contract; if AgentM changes
it, this tool only needs to track ``type``/``payload`` shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from .audit import entry_types as _et
from .schema import Event, Verdict

_AUDIT_EVENT_ENTRY_TYPE = _et.AUDIT_EVENT
_VERDICT_ENTRY_TYPE = _et.VERDICT
_EXTRACTOR_CURSOR_ENTRY_TYPE = _et.EXTRACTOR_CURSOR
_MESSAGE_ENTRY_TYPE = _et.MESSAGE

_RECENT_GRAPH_SLICE_FOR_EXTRACTOR = _et.RECENT_GRAPH_SLICE_FOR_EXTRACTOR
_RECENT_VERDICTS_FOR_AUDITOR = _et.RECENT_VERDICTS_FOR_AUDITOR


# --- session-file resolution ------------------------------------------------
#
# These helpers must stay in lockstep with
# ``agentm.harness.session_manager.SessionManager.default_session_dir`` and
# ``_find_most_recent``. We reproduce the rule here rather than import to
# keep this CLI a read-only tool with no AgentM-runtime dependency.


def _default_sessions_root() -> Path:
    return Path.home() / ".agentm" / "sessions"


def _cwd_slug(cwd: Path) -> str:
    s = str(cwd.resolve()).strip(os.sep).replace(os.sep, "-") or "root"
    return f"--{s}--"


def _resolve_session_file(arg: str | None, cwd: Path) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_file():
            raise FileNotFoundError(f"session file not found: {arg}")
        return p
    directory = _default_sessions_root() / _cwd_slug(cwd)
    if not directory.is_dir():
        raise FileNotFoundError(f"no session directory for cwd {cwd} (looked in {directory})")
    files = sorted(
        (p for p in directory.glob("*.jsonl") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"no session files under {directory}")
    return files[0]


# --- parsing ----------------------------------------------------------------


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _collect(path: Path) -> tuple[list[Event], list[Verdict], int]:
    """Return (events_sorted_by_id, verdicts_in_production_order, cursor_count)."""
    events: list[Event] = []
    verdicts: list[Verdict] = []
    cursor_count = 0
    for rec in _iter_records(path):
        t = rec.get("type")
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue
        if t == _AUDIT_EVENT_ENTRY_TYPE:
            try:
                events.append(Event.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif t == _VERDICT_ENTRY_TYPE:
            try:
                verdicts.append(Verdict.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif t == _EXTRACTOR_CURSOR_ENTRY_TYPE:
            cursor_count += 1
    events.sort(key=lambda e: e.id)
    return events, verdicts, cursor_count


# --- rendering --------------------------------------------------------------


def _fmt_event(e: Event) -> str:
    refs = ",".join(str(r) for r in e.refs) if e.refs else "-"
    turns = ",".join(str(t) for t in e.source_turns) if e.source_turns else "-"
    return (
        f"[event {e.id:>3}] {e.kind.value:<10} "
        f"refs=[{refs}]  turns=[{turns}]\n"
        f"           {e.summary}"
    )


def _fmt_verdict(idx: int, v: Verdict) -> str:
    head = f"[verdict #{idx}] surface_reminder={v.surface_reminder}"
    if v.surface_reminder:
        matched = ",".join(str(i) for i in v.matched_event_ids)
        head += f"  matched=[{matched}]"
    body = ""
    if v.surface_reminder and v.reminder_text:
        body = f"\n           reminder: {v.reminder_text}"
    return head + body


def _render_text(
    events: list[Event],
    verdicts: list[Verdict],
    cursor_count: int,
    session_file: Path,
    out: TextIO,
    skip_events: bool,
    skip_verdicts: bool,
) -> None:
    drift_n = sum(1 for v in verdicts if v.surface_reminder)
    out.write(f"session: {session_file}\n")
    out.write(
        f"events={len(events)}  verdicts={len(verdicts)} (drift={drift_n})  "
        f"extractor_runs={cursor_count}\n\n"
    )
    if skip_events and skip_verdicts:
        return
    # Two passes: events in id order, then verdicts in production order.
    # The graph and the verdict stream are conceptually different views
    # of the same session; printing them sequentially is clearer than
    # synthesizing a positional interleave.
    if not skip_events:
        for e in events:
            out.write(_fmt_event(e))
            out.write("\n\n")
    if not skip_verdicts:
        for vi, v in enumerate(verdicts):
            out.write(_fmt_verdict(vi, v))
            out.write("\n\n")


def _render_json(
    events: list[Event],
    verdicts: list[Verdict],
    cursor_count: int,
    session_file: Path,
    out: TextIO,
    skip_events: bool,
    skip_verdicts: bool,
) -> None:
    payload: dict[str, Any] = {
        "session_file": str(session_file),
        "extractor_runs": cursor_count,
    }
    if not skip_events:
        payload["events"] = [e.to_dict() for e in events]
    if not skip_verdicts:
        payload["verdicts"] = [v.to_dict() for v in verdicts]
    json.dump(payload, out, indent=2, ensure_ascii=False)
    out.write("\n")


# --- dataset export ---------------------------------------------------------


def _to_extractor_msg(payload: Any, index: int) -> dict[str, Any] | None:
    """Reshape one persisted message record into the extractor input format.

    Runtime extractor receives ``{index, role, content[]}`` (see
    ``adapters.agentm._serialize_message_for_extractor``). The persisted
    payload is the same shape minus ``index`` plus ``timestamp``; we add
    ``index`` and drop ``timestamp`` so dataset cases match runtime exactly.
    """
    if not isinstance(payload, dict):
        return None
    role = payload.get("role")
    content = payload.get("content")
    if not isinstance(role, str) or not isinstance(content, list):
        return None
    blocks = [b for b in content if isinstance(b, dict)]
    if not blocks:
        return None
    return {"index": index, "role": role, "content": blocks}


def _walk_dataset(
    path: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Replay the session JSONL, yielding (kind, case) for each firing.

    ``kind`` is ``"extractor"`` or ``"auditor"``. Each ``extractor_cursor``
    entry caps a firing: every ``audit_event`` since the previous cursor
    is that firing's output, and the message slice referenced by
    ``last_turn_index`` (relative to the prior cursor) is its input. Each
    ``verdict`` entry is one auditor firing.

    Two-pass: ``AgentSession.prompt`` flushes new messages to the
    SessionManager in batch *after* the loop returns, while the audit
    pipeline appends entries inline through the same loop. Their physical
    order in the JSONL therefore differs from their semantic order — all
    audit entries land before any of the run's messages — so we collect
    messages first, then walk audit entries against the complete list.
    """
    messages: list[dict[str, Any]] = []
    for rec in _iter_records(path):
        if rec.get("type") == _MESSAGE_ENTRY_TYPE:
            payload = rec.get("payload")
            messages.append(payload if isinstance(payload, dict) else {})

    audit_events: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    pending_extractor_events: list[dict[str, Any]] = []
    last_cursor_turn_index = -1

    for rec in _iter_records(path):
        t = rec.get("type")
        payload = rec.get("payload")
        if t == _MESSAGE_ENTRY_TYPE:
            continue
        if not isinstance(payload, dict):
            continue
        if t == _AUDIT_EVENT_ENTRY_TYPE:
            pending_extractor_events.append(payload)
            continue
        if t == _EXTRACTOR_CURSOR_ENTRY_TYPE:
            this_turn_index = payload.get("last_turn_index")
            if not isinstance(this_turn_index, int):
                # Malformed cursor; commit pending events so they aren't
                # lost from the running graph but skip building a case.
                audit_events.extend(pending_extractor_events)
                pending_extractor_events = []
                continue
            window_lo = max(last_cursor_turn_index + 1, 0)
            window_hi = this_turn_index + 1
            new_turns: list[dict[str, Any]] = []
            for idx in range(window_lo, min(window_hi, len(messages))):
                shaped = _to_extractor_msg(messages[idx], idx)
                if shaped is not None:
                    new_turns.append(shaped)
            yield (
                "extractor",
                {
                    "input": {
                        "new_turns": new_turns,
                        "recent_graph": audit_events[-_RECENT_GRAPH_SLICE_FOR_EXTRACTOR:],
                    },
                    "output": {"events": pending_extractor_events},
                    "meta": {
                        "turn_window": [last_cursor_turn_index + 1, this_turn_index],
                        "extraction_run_id": payload.get("extraction_run_id"),
                    },
                },
            )
            audit_events.extend(pending_extractor_events)
            pending_extractor_events = []
            last_cursor_turn_index = this_turn_index
            continue
        if t == _VERDICT_ENTRY_TYPE:
            yield (
                "auditor",
                {
                    "input": {
                        "graph": list(audit_events),
                        "recent_verdicts": verdicts[-_RECENT_VERDICTS_FOR_AUDITOR:],
                    },
                    "output": {"verdict": payload},
                },
            )
            verdicts.append(payload)
            continue


def _stream_dataset_to_files(path: Path, out_dir: Path) -> tuple[int, int]:
    """Stream ``_walk_dataset`` straight to extractor.jsonl + auditor.jsonl.

    Returns ``(extractor_count, auditor_count)``. Memory stays O(running
    graph + messages) regardless of total case count, so a 1000-turn
    session does not blow up the exporter.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_count = aud_count = 0
    with (
        (out_dir / "extractor.jsonl").open("w", encoding="utf-8") as ext_fh,
        (out_dir / "auditor.jsonl").open("w", encoding="utf-8") as aud_fh,
    ):
        for kind, case in _walk_dataset(path):
            line = json.dumps(case, ensure_ascii=False, default=str)
            if kind == "extractor":
                ext_fh.write(line)
                ext_fh.write("\n")
                ext_count += 1
            else:
                aud_fh.write(line)
                aud_fh.write("\n")
                aud_count += 1
    return ext_count, aud_count


# --- dataset review ---------------------------------------------------------


def _resolve_dataset_files(path: Path, kind_filter: str | None) -> list[tuple[str, Path]]:
    """Return [(kind, file)] in display order for review-dataset.

    Single-file PATH: kind is inferred from the filename stem; ``--kind``
    must agree if given. Directory PATH: pick up extractor.jsonl /
    auditor.jsonl when present, optionally filtered by ``--kind``.
    """
    if path.is_file():
        # Stem must equal one of the canonical kinds, e.g. ``extractor.jsonl``
        # or ``auditor.jsonl``. Substring match would mis-classify
        # ``my_extractor_v2.jsonl`` and silently pick the first hit on
        # ambiguous names like ``auditor_extractor.jsonl``.
        kind = path.stem if path.stem in ("extractor", "auditor") else None
        if kind is None:
            raise FileNotFoundError(
                f"cannot infer kind from filename {path.name}; "
                "expected extractor.jsonl or auditor.jsonl"
            )
        if kind_filter and kind != kind_filter:
            return []
        return [(kind, path)]
    if not path.is_dir():
        raise FileNotFoundError(f"path not found: {path}")
    out: list[tuple[str, Path]] = []
    for kind in ("extractor", "auditor"):
        if kind_filter and kind != kind_filter:
            continue
        candidate = path / f"{kind}.jsonl"
        if candidate.is_file():
            out.append((kind, candidate))
    if not out:
        raise FileNotFoundError(f"no extractor.jsonl/auditor.jsonl under {path}")
    return out


def _summarize_case(kind: str, case: dict[str, Any]) -> str:
    inp = case.get("input") or {}
    out = case.get("output") or {}
    if kind == "extractor":
        meta = case.get("meta") or {}
        window = meta.get("turn_window")
        new_turns = inp.get("new_turns") or []
        recent = inp.get("recent_graph") or []
        events = out.get("events") or []
        kinds = ",".join(e.get("kind", "?") for e in events) or "-"
        return (
            f"window={window}  new_turns={len(new_turns)}  "
            f"recent_graph={len(recent)}  → events={len(events)} [{kinds}]"
        )
    graph = inp.get("graph") or []
    rv = inp.get("recent_verdicts") or []
    verdict = out.get("verdict") or {}
    surface_reminder = verdict.get("surface_reminder")
    reminder_text = (verdict.get("reminder_text") or "").replace("\n", " ")
    if len(reminder_text) > 80:
        reminder_text = reminder_text[:77] + "..."
    tail = f"  reminder={reminder_text!r}" if reminder_text else ""
    return (
        f"graph={len(graph)}  recent_verdicts={len(rv)}  "
        f"→ surface_reminder={surface_reminder}{tail}"
    )


def _run_review_dataset(args: argparse.Namespace) -> int:
    try:
        files = _resolve_dataset_files(Path(args.path), args.kind)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.case is not None and len(files) > 1:
        print(
            "error: --case requires --kind when PATH is a directory with both files.",
            file=sys.stderr,
        )
        return 2
    for kind, file in files:
        cases = list(_iter_records(file))
        header = f"=== {kind}.jsonl  ({len(cases)} cases)  {file} ==="
        if args.case is not None:
            n = args.case
            if not 1 <= n <= len(cases):
                print(
                    f"error: --case {n} out of range (1..{len(cases)})",
                    file=sys.stderr,
                )
                return 2
            sys.stdout.write(header + "\n")
            json.dump(
                cases[n - 1],
                sys.stdout,
                indent=None if args.json else 2,
                ensure_ascii=False,
            )
            sys.stdout.write("\n")
            continue
        sys.stdout.write(header + "\n")
        if args.full:
            for i, case in enumerate(cases, 1):
                sys.stdout.write(f"--- case {i} ---\n")
                json.dump(case, sys.stdout, indent=2, ensure_ascii=False)
                sys.stdout.write("\n")
            continue
        for i, case in enumerate(cases, 1):
            sys.stdout.write(f"  [{i:>3}] {_summarize_case(kind, case)}\n")
    return 0


# --- entry point ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmharness",
        description="Inspect cognitive-audit event graph + verdicts from a session JSONL.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    review = sub.add_parser(
        "review",
        help="Print the event graph and verdicts of a harness-instrumented session.",
    )
    review.add_argument(
        "session",
        nargs="?",
        help="Path to a session .jsonl file. Defaults to the latest session for --cwd.",
    )
    review.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory whose latest session is loaded when SESSION is omitted.",
    )
    review.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text.")
    review.add_argument("--events-only", action="store_true", help="Hide verdicts.")
    review.add_argument("--verdicts-only", action="store_true", help="Hide events.")

    dataset = sub.add_parser(
        "dataset",
        help=(
            "Export per-firing extractor / auditor training cases from a "
            "session JSONL (best paired with rca:harness:sync runs)."
        ),
    )
    dataset.add_argument(
        "session",
        nargs="?",
        help="Path to a session .jsonl file. Defaults to the latest for --cwd.",
    )
    dataset.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory whose latest session is used when SESSION omitted.",
    )
    dataset.add_argument(
        "--out",
        required=True,
        help="Output directory; receives extractor.jsonl and auditor.jsonl.",
    )

    review_html = sub.add_parser(
        "review-html",
        help=(
            "Generate a self-contained HTML page with trajectory + event "
            "graph + verdicts (three-pane interactive view)."
        ),
    )
    review_html.add_argument(
        "session",
        nargs="?",
        help="Path to a session .jsonl file. Defaults to the latest for --cwd.",
    )
    review_html.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory whose latest session is used when SESSION omitted.",
    )
    review_html.add_argument(
        "--out",
        required=True,
        help="Output HTML file path.",
    )

    review_ds = sub.add_parser(
        "review-dataset",
        help=(
            "Pretty-print exported extractor.jsonl / auditor.jsonl. Default is "
            "a one-line-per-case index; use --case N to pretty-print one case."
        ),
    )
    review_ds.add_argument(
        "path",
        help=(
            "extractor.jsonl, auditor.jsonl, or a directory containing one or "
            "both (kind is inferred from the filename)."
        ),
    )
    review_ds.add_argument(
        "--kind",
        choices=("extractor", "auditor"),
        help="Restrict output to one kind when PATH is a directory.",
    )
    review_ds.add_argument(
        "--case",
        type=int,
        help="Pretty-print only this 1-based case number.",
    )
    review_ds.add_argument(
        "--full",
        action="store_true",
        help="Pretty-print every case (equivalent to `jq .`).",
    )
    review_ds.add_argument(
        "--json",
        action="store_true",
        help="With --case, emit raw JSON for piping into jq.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "review":
        try:
            session_file = _resolve_session_file(args.session, Path(args.cwd))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        events, verdicts, cursor_count = _collect(session_file)
        renderer = _render_json if args.json else _render_text
        renderer(
            events,
            verdicts,
            cursor_count,
            session_file,
            sys.stdout,
            skip_events=args.verdicts_only,
            skip_verdicts=args.events_only,
        )
        return 0

    if args.cmd == "review-html":
        try:
            session_file = _resolve_session_file(args.session, Path(args.cwd))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        from .web import build_review_html

        out_path = Path(args.out)
        build_review_html(session_file, out_path)
        print(f"session: {session_file}\nwrote: {out_path}")
        return 0

    if args.cmd == "review-dataset":
        return _run_review_dataset(args)

    if args.cmd == "dataset":
        try:
            session_file = _resolve_session_file(args.session, Path(args.cwd))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        out_dir = Path(args.out)
        ext_n, aud_n = _stream_dataset_to_files(session_file, out_dir)
        print(
            f"session: {session_file}\n"
            f"wrote: {out_dir / 'extractor.jsonl'} ({ext_n} cases)\n"
            f"wrote: {out_dir / 'auditor.jsonl'} ({aud_n} cases)"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
