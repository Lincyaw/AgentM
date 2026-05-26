#!/usr/bin/env python3
"""Re-run the extractor over a pinned row set with a chosen prompt.

Sits between `eval-db extract` (which we can't re-invoke because the
source eval.db isn't local anymore) and `llmharness-aggregate eval-db`.
For each row id in --rows-file, the script:

    1. reads <src-root>/<row>/{records.jsonl, meta.json, graph.json}
    2. replays every ``phase == extractor`` record with the chosen
       prompt-override (auditor records pass through untouched)
    3. writes the replayed records + meta to
       <out-root>/<row>/{records.jsonl, meta.json, graph.json}

Auditor records flow through as-is because the prompt iteration we're
optimising for is purely extractor-side. The aggregator can then turn
the per-row dirs into a canonical cases/ layout via::

    llmharness-aggregate eval-db --src <out-root> --out <cases-out>

Concurrency is across ROWS, not across firings within a row — each row
remains a sequential session, matching the live adapter's behaviour and
respecting provider rate limits. The default of 10 matches the pinned
row set so a typical iter run finishes in one batch.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Add the llmharness src to path so this script runs without install.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from llmharness.aggregate.collector import (  # noqa: E402
    _main_agent_messages as build_main_agent_messages,
)
from llmharness.replay.record import ReplayRecord, iter_records  # noqa: E402
from llmharness.replay.runner import replay_extractor_record  # noqa: E402


def _render_message_text(msg: dict[str, Any]) -> str:
    """Witness-substring text for a main-agent message (dict shape).

    Mirrors ``adapters/agentm.py:_render_message_text`` but operates on
    the dict-form messages we get from records.jsonl. Concatenates every
    text-bearing block (assistant text, thinking, tool-result text, user
    text) into a single string.
    """
    parts: list[str] = []
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            inner = block.get("content")
            if isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict):
                        sub_text = sub.get("text")
                        if isinstance(sub_text, str) and sub_text:
                            parts.append(sub_text)
                    elif isinstance(sub, str):
                        parts.append(sub)
            elif isinstance(inner, str):
                parts.append(inner)
            args = block.get("arguments")
            if isinstance(args, dict):
                try:
                    parts.append(json.dumps(args, ensure_ascii=False, default=str))
                except (TypeError, ValueError):
                    pass
    return " ".join(parts)


def _provider_override_from_env(
    record: ReplayRecord,
    *,
    max_output_tokens: int | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Patch the recorded provider config with current-env credentials.

    The recorded ``provider`` field bakes in the ``warpgate-ticket`` that
    was live when ``eval-db extract`` originally ran. Tickets expire, so
    replaying months later returns HTTP 307 → /@warpgate/login.
    If ``WARPGATE_TICKET`` is set in the current environment, we copy the
    recorded provider config and swap in the fresh ticket.
    ``OPENAI_BASE_URL`` is also overridden when present, in case the
    gateway IP has moved.
    ``max_output_tokens`` overrides the AgentM-default 4096 — long
    trajectories (e.g. ``--full-trajectory`` mode) need a larger budget
    so the model can finish writing the block_plan + events payload.
    Returns None when there's nothing to patch, so the runner falls back
    to the record's provider unchanged.
    """
    raw = record.provider
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    module, cfg = raw[0], raw[1]
    if not isinstance(module, str) or not isinstance(cfg, dict):
        return None

    ticket = os.environ.get("WARPGATE_TICKET")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not ticket and not base_url and max_output_tokens is None:
        return None

    patched = copy.deepcopy(cfg)
    if ticket:
        default_query = patched.get("default_query")
        if isinstance(default_query, dict):
            default_query["warpgate-ticket"] = ticket
        else:
            patched["default_query"] = {"warpgate-ticket": ticket}
    if base_url:
        patched["base_url"] = base_url
    if max_output_tokens is not None:
        patched["max_output_tokens"] = int(max_output_tokens)
    return module, patched


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


async def _rerun_one_row(
    *,
    row_id: int,
    src_dir: Path,
    out_dir: Path,
    prompt_override: str,
    cwd: str,
    semaphore: asyncio.Semaphore,
    witness_retry_budget: int,
    full_trajectory: bool,
    max_output_tokens: int | None,
) -> tuple[int, int, int, int]:
    """Returns (row_id, n_replayed_extractor, n_passthrough_auditor, n_ok)."""
    async with semaphore:
        records_in = src_dir / "records.jsonl"
        if not records_in.is_file():
            raise FileNotFoundError(f"no records.jsonl for row {row_id}: {records_in}")

        out_dir.mkdir(parents=True, exist_ok=True)
        # Preserve meta + graph so the aggregator can pick them up.
        for sidecar in ("meta.json", "graph.json"):
            src = src_dir / sidecar
            if src.is_file():
                shutil.copy2(src, out_dir / sidecar)

        # Read the full record stream once so we can build the canonical
        # main-agent trajectory before walking firings. The trajectory is
        # the ground truth: every extractor request rebuilds new_turns,
        # turn_texts, recent_graph, and next_event_id from it rather than
        # trusting the captured payload (which froze under the OLD local-
        # id schema and is now stale).
        all_records = list(iter_records(records_in))
        extractor_records = [r for r in all_records if r.phase == "extractor"]
        auditor_records = [r for r in all_records if r.phase == "auditor"]
        main_agent = build_main_agent_messages(auditor_records, extractor_records)
        main_by_index: dict[int, dict[str, Any]] = {
            int(m["index"]): m
            for m in main_agent
            if isinstance(m, dict) and isinstance(m.get("index"), int)
        }

        replayed = 0
        passthrough = 0
        ok = 0
        # Track post-replay events across firings so each replay sees a
        # consistent global id space and recent_graph.
        running_events: list[dict[str, Any]] = []
        out_path = out_dir / "records.jsonl"
        # full_trajectory mode: collapse every recorded extractor firing
        # into a single replay firing whose window spans the entire
        # main-agent trajectory. The trajectory itself is the only
        # ground truth; the live adapter's cadence is irrelevant for
        # offline re-extraction. Subsequent extractor records in the
        # source are dropped (we keep one). Recent_graph is empty for
        # this firing because there are no prior firings.
        seen_full_firing = False
        full_lo = min(main_by_index.keys()) if main_by_index else 0
        full_hi = max(main_by_index.keys()) if main_by_index else 0
        # Open in write mode so re-running the script is idempotent.
        with out_path.open("w", encoding="utf-8") as fh:
            for record in all_records:
                if record.phase != "extractor":
                    fh.write(record.to_jsonl())
                    fh.write("\n")
                    passthrough += 1
                    continue

                if full_trajectory:
                    if seen_full_firing:
                        # Drop additional recorded firings — the single
                        # collapsed firing has already covered everything.
                        continue
                    seen_full_firing = True
                    window_lo = full_lo
                    window_hi = full_hi
                else:
                    # Derive this firing's window from the captured
                    # payload's new_turns indices — the trigger boundary
                    # is the LIVE adapter's choice, and the main_agent
                    # slice between those indices is the canonical input.
                    captured_new = record.payload.get("new_turns")
                    if not (
                        isinstance(captured_new, list)
                        and captured_new
                        and isinstance(captured_new[0], dict)
                        and isinstance(captured_new[-1], dict)
                        and isinstance(captured_new[0].get("index"), int)
                        and isinstance(captured_new[-1].get("index"), int)
                    ):
                        raise ValueError(
                            f"row={row_id} firing={record.turn_index}: cannot "
                            "derive window — payload.new_turns missing or "
                            "lacks integer .index fields"
                        )
                    window_lo = int(captured_new[0]["index"])
                    window_hi = int(captured_new[-1]["index"])
                new_turns = [
                    main_by_index[i]
                    for i in range(window_lo, window_hi + 1)
                    if i in main_by_index
                ]

                # turn_texts for the firing window + every recent_graph
                # entry's source_turns (witness must reach the source side
                # of cross-firing refs).
                next_event_id = (
                    max((e["id"] for e in running_events), default=0) + 1
                )
                rebuilt_recent: list[dict[str, Any]] = [
                    {
                        "id": e["id"],
                        "kind": e["kind"],
                        "summary": e["summary"],
                        "source_turns": list(e.get("source_turns") or []),
                    }
                    for e in running_events
                ]
                in_window_turn_texts: dict[int, str] = {
                    i: _render_message_text(main_by_index[i])
                    for i in range(window_lo, window_hi + 1)
                    if i in main_by_index
                }
                prior_turn_texts: dict[int, str] = {}
                for e in running_events:
                    for t in e.get("source_turns") or []:
                        if not isinstance(t, int):
                            continue
                        if t in in_window_turn_texts or t in prior_turn_texts:
                            continue
                        msg = main_by_index.get(t)
                        if msg is not None:
                            prior_turn_texts[t] = _render_message_text(msg)

                rebuilt_payload: dict[str, Any] = {
                    "next_event_id": next_event_id,
                    "new_turns": new_turns,
                    "recent_graph": rebuilt_recent,
                }
                rebuilt_extras: dict[str, Any] = dict(record.extras)
                rebuilt_extras["turn_window_json"] = json.dumps(
                    new_turns, ensure_ascii=False, default=str
                )
                rebuilt_extras["turn_texts"] = {
                    str(k): v for k, v in in_window_turn_texts.items()
                }
                rebuilt_extras["prior_turn_texts"] = {
                    str(k): v for k, v in prior_turn_texts.items()
                }

                patched_record = ReplayRecord(
                    phase=record.phase,
                    turn_index=record.turn_index,
                    session_id=record.session_id,
                    trace_id=record.trace_id,
                    ts_ns=record.ts_ns,
                    compose_kwargs=record.compose_kwargs,
                    payload=rebuilt_payload,
                    provider=record.provider,
                    output=record.output,
                    status=record.status,
                    error=record.error,
                    latency_ms=record.latency_ms,
                    extras=rebuilt_extras,
                )

                t0 = time.monotonic()
                try:
                    result = await replay_extractor_record(
                        patched_record,
                        cwd=cwd,
                        prompt_override=prompt_override,
                        provider_override=_provider_override_from_env(
                            record, max_output_tokens=max_output_tokens
                        ),
                    )
                    status = result.status
                    output = result.output
                    error = result.error
                    latency_ms = result.latency_ms
                except Exception as exc:  # noqa: BLE001 — record failure, keep going
                    status = "spawn_error"
                    output = None
                    error = f"{type(exc).__name__}: {exc}"
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    print(
                        f"  row={row_id} firing={record.turn_index} FAILED: {error}",
                        file=sys.stderr,
                    )

                if status == "ok" and output:
                    for ev in output.get("events") or []:
                        if isinstance(ev, dict) and isinstance(ev.get("id"), int):
                            running_events.append(ev)

                replacement = ReplayRecord(
                    phase=record.phase,
                    turn_index=record.turn_index,
                    session_id=record.session_id,
                    trace_id=record.trace_id,
                    ts_ns=record.ts_ns,
                    compose_kwargs=record.compose_kwargs,
                    payload=rebuilt_payload,
                    provider=record.provider,
                    output=output,
                    status=status,
                    error=error,
                    latency_ms=latency_ms,
                    extras=rebuilt_extras,
                )
                fh.write(replacement.to_jsonl())
                fh.write("\n")
                replayed += 1
                if status == "ok":
                    ok += 1

        print(
            f"  row={row_id}  extractor_replayed={replayed}  "
            f"ok={ok}  auditor_passthrough={passthrough}",
            file=sys.stderr,
        )
        return row_id, replayed, passthrough, ok


async def _main(args: argparse.Namespace) -> int:
    src_root: Path = args.src_root
    out_root: Path = args.out_root
    rows = _read_pinned_rows(args.rows_file)
    missing = [r for r in rows if not (src_root / str(r) / "records.jsonl").is_file()]
    if missing:
        print(f"missing source records for rows: {missing}", file=sys.stderr)
        return 4

    prompt_path: Path = args.prompt
    if not prompt_path.is_file():
        print(f"--prompt not found: {prompt_path}", file=sys.stderr)
        return 4
    prompt_text = prompt_path.read_text(encoding="utf-8")

    out_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    cwd = str(args.cwd.resolve()) if args.cwd else str(Path.cwd())

    print(
        f"=> reruning extractor on {len(rows)} rows  src={src_root}  "
        f"out={out_root}  prompt={prompt_path}  concurrency={args.concurrency}",
        file=sys.stderr,
    )

    tasks = [
        _rerun_one_row(
            row_id=r,
            src_dir=src_root / str(r),
            out_dir=out_root / str(r),
            prompt_override=prompt_text,
            cwd=cwd,
            semaphore=semaphore,
            witness_retry_budget=args.witness_retry_budget,
            full_trajectory=args.full_trajectory,
            max_output_tokens=args.max_output_tokens,
        )
        for r in rows
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        for f in failures:
            print(f"  ROW FAILED: {f}", file=sys.stderr)
        return 2

    total_repl = sum(t[1] for t in results)  # type: ignore[index]
    total_pass = sum(t[2] for t in results)  # type: ignore[index]
    total_ok = sum(t[3] for t in results)  # type: ignore[index]
    print(
        f"\nDone. Replayed {total_repl} extractor records "
        f"(ok={total_ok}), {total_pass} auditor passthroughs.",
        file=sys.stderr,
    )
    if total_repl > 0 and total_ok == 0:
        print(
            "ERROR: every extractor replay failed (likely auth / provider "
            "config missing — check OPENAI_API_KEY, OPENAI_BASE_URL, "
            "WARPGATE_TICKET, OPENAI_VERIFY_SSL). Aborting before aggregate.",
            file=sys.stderr,
        )
        return 5
    print(
        f"Next step:\n"
        f"  llmharness-aggregate eval-db --src {out_root} --out <cases-out>",
        file=sys.stderr,
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Existing eval-db extract root (contains <row>/records.jsonl).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Destination dir; will mirror <row>/records.jsonl shape.",
    )
    ap.add_argument(
        "--rows-file",
        type=Path,
        required=True,
        help="File with one row id per line (e.g. runs/iters/pinned-10.txt).",
    )
    ap.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Prompt file used as extractor system prompt override.",
    )
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument(
        "--witness-retry-budget",
        dest="witness_retry_budget",
        type=int,
        default=3,
        help=(
            "How many times to bounce back a submission with non-empty "
            "dropped_edges so the LLM can re-cite. Default 3."
        ),
    )
    ap.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Working dir for the spawned AgentSession (default: $PWD).",
    )
    ap.add_argument(
        "--full-trajectory",
        dest="full_trajectory",
        action="store_true",
        help=(
            "Run ONE extractor firing per row covering the entire "
            "main-agent trajectory, instead of replaying the captured "
            "10-turn cadence. The live adapter's firing cadence is "
            "irrelevant for offline re-extraction; this option lets "
            "the LLM see the whole investigation arc at once so it "
            "can identify cross-firing threads."
        ),
    )
    ap.add_argument(
        "--max-output-tokens",
        dest="max_output_tokens",
        type=int,
        default=None,
        help=(
            "Override the LLM's max output tokens (AgentM default 4096). "
            "Long trajectories under --full-trajectory need a larger "
            "budget so the block_plan + events JSON can finish; row9 "
            "in v18 lost its events array to a 4096-token truncation "
            "mid-plan. Recommended: 16384 for full-trajectory mode."
        ),
    )
    args = ap.parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
