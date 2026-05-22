#!/usr/bin/env python3
"""Run the auditor standalone over already-extracted event graphs.

Sits parallel to ``rerun_extractor.py`` but for the *auditor* side:
v22-style runs are extractor-only (no auditor records), so
``replay_auditor_record`` can't be used. This script reads the
post-extractor event graph from ``cases/<row>/event_graph/after_extractor_NNN.json``,
builds auditor extensions via ``compose_auditor_extensions``, and pumps
a synthetic payload through ``run_phase_standalone`` with a chosen
auditor prompt (typically ``auditor_detective.md``).

Concurrency is across rows, matching ``rerun_extractor.py``. Provider
config comes from the raw records.jsonl (any extractor record) so the
ticket/base_url match the captured run; ``WARPGATE_TICKET`` /
``OPENAI_BASE_URL`` env vars override.

Outputs in two places (both written by default):

1. **Debug output** at ``<out-root>/<row>/{verdict.json,messages.jsonl}``
   plus ``<out-root>/summary.json`` — full transcript, raw tool args,
   prompt snapshot. For human review of the audit run.

2. **Frontend-visible firing** at
   ``<cases-root>/<row>/auditor/<NNN>_turn_<T>.json`` shaped exactly
   like ``AuditorFiring`` in the aegis-ui schema. The case's
   ``meta.json`` is patched to bump ``auditor_firings`` /
   ``surfaced_reminders`` / ``silent_verdicts``. The /cases UI
   ``mode=auditor`` view picks these up automatically once the cases/
   dir is re-uploaded to blob.

Pass ``--skip-case-write`` to only produce the debug output.

Usage::

    scripts/audit-graph.py \\
        --cases-root runs/iters/v22/cases \\
        --raw-root runs/iters/v22/raw/openrca-2-lite-n500-t20 \\
        --rows-file runs/iters/pinned-10.txt \\
        --prompt src/llmharness/audit/auditor/prompts/auditor_detective.md \\
        --out runs/audit/detective-v1 \\
        --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from llmharness.audit.auditor.extensions import compose_auditor_extensions  # noqa: E402
from llmharness.audit.auditor.submit_tool import SUBMIT_VERDICT_TOOL_NAME  # noqa: E402
from llmharness.replay.record import iter_records  # noqa: E402
from llmharness.schema import Edge, Event  # noqa: E402
from llmharness.tools.engine import run_phase_standalone  # noqa: E402


def _read_pinned_rows(path: Path) -> list[int]:
    rows: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        rows.append(int(s))
    if not rows:
        raise ValueError(f"no row ids in {path}")
    return rows


def _provider_from_raw(
    raw_dir: Path,
    *,
    max_output_tokens: int | None,
) -> tuple[str, dict[str, Any]] | None:
    """Read provider config from the first record in <raw_dir>/records.jsonl,
    patched with env-supplied ``WARPGATE_TICKET`` / ``OPENAI_BASE_URL`` so a
    stale recorded ticket doesn't kill the run."""
    records_path = raw_dir / "records.jsonl"
    if not records_path.is_file():
        return None
    for rec in iter_records(records_path):
        raw = rec.provider
        if isinstance(raw, list) and len(raw) >= 2 and isinstance(raw[1], dict):
            module = raw[0]
            cfg = copy.deepcopy(raw[1])
            ticket = os.environ.get("WARPGATE_TICKET")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if ticket:
                dq = cfg.get("default_query")
                if isinstance(dq, dict):
                    dq["warpgate-ticket"] = ticket
                else:
                    cfg["default_query"] = {"warpgate-ticket": ticket}
            if base_url:
                cfg["base_url"] = base_url
            if max_output_tokens is not None:
                cfg["max_output_tokens"] = int(max_output_tokens)
            return module, cfg
    return None


def _load_graph(case_dir: Path) -> tuple[list[Event], list[Edge], int]:
    """Returns (events, edges, extractor_firing_seq).

    The seq is the sequence number of the extractor firing whose graph
    snapshot we audit — it becomes ``graph_snapshot_ref`` on the
    AuditorFiring so the UI can pivot back to the source firing.
    """
    eg_dir = case_dir / "event_graph"
    candidates = sorted(eg_dir.glob("after_extractor_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no event_graph for {case_dir}")
    last = candidates[-1]
    payload = json.loads(last.read_text(encoding="utf-8"))
    events = [Event.from_dict(e) for e in payload.get("events", [])]
    edges = [Edge.from_dict(e) for e in payload.get("edges", [])]
    # filename shape: after_extractor_NNN.json
    seq = int(last.stem.split("_")[-1])
    return events, edges, seq


def _next_auditor_seq_and_turn(case_dir: Path, fallback_turn: int) -> tuple[int, int]:
    """Pick the next auditor firing sequence (1-based) for this case and
    the turn_index to file it under.

    For now we fire after the *last* extractor — same turn_index as that
    firing — so the auditor view sits next to its source graph in the
    timeline. If prior auditor firings exist (rerun, multiple prompts),
    we append at the next sequence rather than overwriting.
    """
    aud_dir = case_dir / "auditor"
    existing = list(aud_dir.glob("*_turn_*.json")) if aud_dir.exists() else []
    next_seq = len(existing) + 1
    # Match the on-disk turn-index of the most recent extractor firing,
    # since this audit is "as if" it fired right after that extractor.
    ext_dir = case_dir / "extractor"
    if ext_dir.exists():
        ext_files = sorted(ext_dir.glob("*_turn_*.json"))
        if ext_files:
            try:
                return next_seq, int(ext_files[-1].stem.split("_turn_")[-1])
            except (ValueError, IndexError):
                pass
    return next_seq, fallback_turn


def _patch_meta(case_dir: Path, surfaced: bool) -> None:
    meta_path = case_dir / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["auditor_firings"] = int(meta.get("auditor_firings", 0)) + 1
    if surfaced:
        meta["surfaced_reminders"] = int(meta.get("surfaced_reminders", 0)) + 1
    else:
        meta["silent_verdicts"] = int(meta.get("silent_verdicts", 0)) + 1
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


async def _audit_one_row(
    *,
    row_id: int,
    case_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    prompt_text: str,
    cwd: str,
    semaphore: asyncio.Semaphore,
    max_output_tokens: int | None,
    write_to_case_dir: bool,
) -> dict[str, Any]:
    async with semaphore:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            events, edges, extractor_seq = _load_graph(case_dir)
        except FileNotFoundError as exc:
            err = {"row_id": row_id, "status": "no_graph", "error": str(exc)}
            (out_dir / "verdict.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return err

        provider = _provider_from_raw(raw_dir, max_output_tokens=max_output_tokens)

        extensions = compose_auditor_extensions(
            base_prompt=prompt_text,
            events=tuple(events),
            edges=tuple(edges),
            phases=(),
            findings=[],
            check_errors={},
            continuation_notes=[],
            tools=(SUBMIT_VERDICT_TOOL_NAME,),
        )

        payload = {
            "graph": [e.to_dict() for e in events],
            "recent_verdicts": [],
            "continuation_notes_from_prior_firing": [],
        }

        t0 = time.monotonic()
        result = await run_phase_standalone(
            cwd=cwd,
            extensions=extensions,
            provider=provider,
            payload=payload,
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_standalone",
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # ``submit_verdict`` tool args are ``{"verdict": {...real fields...}}``;
        # unwrap once so the inner block matches the AuditorFiring schema.
        outer = result.output if isinstance(result.output, dict) else None
        inner_verdict = (
            outer.get("verdict") if isinstance(outer, dict) and isinstance(outer.get("verdict"), dict) else None
        )

        verdict_record = {
            "row_id": row_id,
            "status": result.status,
            "error": result.error,
            "latency_ms": result.latency_ms,
            "wall_ms": elapsed_ms,
            "verdict": result.output,
        }
        (out_dir / "verdict.json").write_text(
            json.dumps(verdict_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Dump full transcript so we can inspect the model's reasoning,
        # not just the final tool args.
        msgs_path = out_dir / "messages.jsonl"
        with msgs_path.open("w", encoding="utf-8") as f:
            for msg in result.messages:
                try:
                    f.write(
                        json.dumps(
                            {
                                "role": getattr(msg, "role", None),
                                "content": getattr(msg, "content", None),
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                except (TypeError, ValueError):
                    continue

        # Write the frontend-visible AuditorFiring file into the case dir
        # so /cases UI ``mode=auditor`` view picks it up. Skip on failure
        # — leaving meta.json untouched is preferable to recording a
        # malformed firing.
        if write_to_case_dir and result.status == "ok" and inner_verdict is not None:
            aud_dir = case_dir / "auditor"
            aud_dir.mkdir(parents=True, exist_ok=True)
            seq, turn_index = _next_auditor_seq_and_turn(case_dir, fallback_turn=0)
            firing = {
                "phase": "auditor",
                "sequence": seq,
                "turn_index": turn_index,
                "ts_ns": time.time_ns(),
                "status": result.status,
                "error": result.error,
                "latency_ms": result.latency_ms,
                "input": {
                    "graph_snapshot_ref": extractor_seq,
                    "findings": [],
                    "continuation_notes": [],
                    "check_errors": {},
                    "tools_profile": "minimal",
                    "trajectory_snapshot_len": 0,
                    "prompt_variant": "detective",
                },
                "output": inner_verdict,
            }
            firing_path = aud_dir / f"{seq:03d}_turn_{turn_index:03d}.json"
            firing_path.write_text(
                json.dumps(firing, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _patch_meta(case_dir, surfaced=bool(inner_verdict.get("surface_reminder")))

        return verdict_record


async def _main_async(args: argparse.Namespace) -> int:
    cases_root = Path(args.cases_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    rows = _read_pinned_rows(Path(args.rows_file).resolve())
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt).resolve()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    (out_root / "prompt-used.md").write_text(prompt_text, encoding="utf-8")

    semaphore = asyncio.Semaphore(args.concurrency)
    cwd = str(Path.cwd())

    tasks: list[asyncio.Task[dict[str, Any]]] = []
    for row_id in rows:
        case_dir = cases_root / f"openrca-2-lite-n500-t20-row{row_id}"
        raw_dir = raw_root / str(row_id)
        out_dir = out_root / f"row{row_id}"
        tasks.append(
            asyncio.create_task(
                _audit_one_row(
                    row_id=row_id,
                    case_dir=case_dir,
                    raw_dir=raw_dir,
                    out_dir=out_dir,
                    prompt_text=prompt_text,
                    cwd=cwd,
                    semaphore=semaphore,
                    max_output_tokens=args.max_output_tokens,
                    write_to_case_dir=not args.skip_case_write,
                )
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    summary: list[dict[str, Any]] = []
    for row_id, res in zip(rows, results, strict=True):
        if isinstance(res, BaseException):
            summary.append({"row_id": row_id, "status": "exception", "error": repr(res)})
            continue
        # ``submit_verdict``'s tool args are ``{"verdict": {...real fields...}}``
        # so ``result.output`` is the outer wrapper; the actual verdict lives
        # one level in.
        outer = res.get("verdict") or {}
        v = outer.get("verdict") if isinstance(outer, dict) else None
        v = v if isinstance(v, dict) else {}
        summary.append(
            {
                "row_id": row_id,
                "status": res.get("status"),
                "error": res.get("error"),
                "latency_ms": res.get("latency_ms"),
                "surface_reminder": v.get("surface_reminder"),
                "reminder_text": v.get("reminder_text"),
                "matched_event_ids": v.get("matched_event_ids"),
            }
        )

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[audit-graph] wrote {out_root}/summary.json")
    for row in summary:
        flag = "🔔" if row.get("surface_reminder") else ("✅" if row.get("status") == "ok" else "❌")
        print(
            f"  {flag} row{row['row_id']}: status={row.get('status')} "
            f"reminder={bool(row.get('surface_reminder'))} "
            f"latency={row.get('latency_ms')}ms"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cases-root", required=True, help="cases/ dir from an extractor iter run")
    p.add_argument("--raw-root", required=True, help="raw records dir (for provider config)")
    p.add_argument("--rows-file", required=True, help="newline-separated row ids")
    p.add_argument("--prompt", required=True, help="path to auditor prompt .md")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="cap on model output tokens; detective prompts need 4-8k",
    )
    p.add_argument(
        "--skip-case-write",
        action="store_true",
        help="skip writing the AuditorFiring file + meta.json patch into "
        "the case dir; only produce the debug --out artefacts",
    )
    args = p.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
