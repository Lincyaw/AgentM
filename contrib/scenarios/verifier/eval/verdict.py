"""Verdict extraction: parse hop/judge verdicts from observability JSONL."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _read_jsonl_records(path: Path) -> list[dict]:
    """Tolerantly read a possibly-corrupt observability JSONL.

    Under heavy concurrency the observability writer can interleave large
    OTLP records — two objects end up on one physical line (a missing
    ``\\n``) or the tail is truncated mid-write. ``agentm trace`` then
    refuses to parse the whole file and silently yields nothing, dropping a
    verdict the hop agent actually submitted. We re-split the raw byte
    stream with a streaming decoder so concatenated objects are recovered
    and only the unparseable garbage is dropped.
    """
    dec = json.JSONDecoder()
    raw = path.read_text(errors="ignore")
    recs: list[dict] = []
    i, n = 0, len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(raw, i)
            if isinstance(obj, dict):
                recs.append(obj)
            i = end
        except json.JSONDecodeError:
            nl = raw.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
    return recs


def _is_jsonl_clean(path: Path) -> bool:
    """True iff every non-blank line parses as standalone JSON."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
    except Exception:  # noqa: BLE001
        return False
    return True


def extract_hop_verdict(
    obs_dir: Path,
    tool: str = "submit_hop_verdict",
    require_key: str = "verdict",
) -> dict | None:
    """Extract the last accepted tool result via ``agentm trace tools``.

    Shells out to the CLI rather than sniffing raw JSONL, so only accepted
    tool *results* are considered (rejected tool-call arguments are ignored).
    ``tool``/``require_key`` let the same path read the judge review tool
    (``submit_judge_review`` keyed on ``remove``).

    A corrupt obs file (concurrent-write interleaving / truncation) is
    re-split into a clean temp file first — see ``_read_jsonl_records`` —
    so a submitted verdict is never lost to an unparseable neighbouring
    line. The CLI timeout is generous because parsing competes for CPU
    with the hop agents during a high-concurrency batch.
    """
    base = (
        ["agentm"]
        if shutil.which("agentm")
        else ["uv", "run", "--no-sync", "agentm"]
    )
    best: dict | None = None
    for f in sorted(obs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        src = f
        tmp: str | None = None
        if not _is_jsonl_clean(f):
            recs = _read_jsonl_records(f)
            if not recs:
                continue
            fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="obs-sane-")
            with os.fdopen(fd, "w") as out:
                for r in recs:
                    out.write(json.dumps(r) + "\n")
            src = Path(tmp)
        cmd = [
            *base, "trace", "tools",
            "--file", str(src),
            "--tool", tool,
            "--format", "ndjson",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
        except Exception:  # noqa: BLE001
            if tmp:
                Path(tmp).unlink(missing_ok=True)
            continue
        if tmp:
            Path(tmp).unlink(missing_ok=True)
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            result = row.get("result")
            if not result:
                continue
            # result is either a raw string or a ToolResult dict
            # {"content": [{"type":"text","text":"..."}], "is_error": ...}
            if isinstance(result, dict):
                if result.get("is_error"):
                    continue
                content = result.get("content", [])
                if content and isinstance(content[0], dict):
                    result = content[0].get("text", "")
                else:
                    continue
            if not isinstance(result, str) or not result:
                continue
            if '"error"' in result and f'"{require_key}"' not in result:
                continue
            try:
                obj = json.loads(result)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and require_key in obj:
                best = obj
    return best


def verdicts_from_trace(trace: dict) -> list[dict]:
    """Extract all hop verdicts from a propagation trace (workflow output).

    The workflow ``agent()`` return values are already captured in
    ``hop_log`` and ``node_evidence``. This replaces the legacy
    ``collect_all_verdicts`` which walked per-hop directories.
    """
    node_evidence = trace.get("node_evidence", {})
    verdicts: list[dict] = []
    for entry in trace.get("hop_log", []):
        to_svc = entry.get("to", "")
        verdict = entry.get("verdict", "")
        if not to_svc or verdict == "edge_sql":
            continue
        ev = node_evidence.get(to_svc, {})
        verdicts.append({
            "from": entry.get("from", ""),
            "to": to_svc,
            "verdict": verdict,
            "rationale": ev.get("rationale", ""),
            "claim": ev.get("claim", ""),
            "symptom_evidence": ev.get("symptom_evidence", []),
        })
    return verdicts


def collect_all_verdicts(out_dir: Path) -> list[dict]:
    """Collect hop verdicts — prefers propagation_trace.json (workflow
    output), falls back to legacy hops/ directory for old runs.
    """
    trace_path = out_dir / "propagation_trace.json"
    if trace_path.exists():
        try:
            trace = json.loads(trace_path.read_text())
            return verdicts_from_trace(trace)
        except Exception:  # noqa: BLE001
            pass

    hops_dir = out_dir / "hops"
    if not hops_dir.exists():
        return []

    verdicts: list[dict] = []
    for hop_dir in sorted(hops_dir.iterdir()):
        if not hop_dir.is_dir():
            continue
        parts = hop_dir.name.split("__", 1)
        if len(parts) != 2:
            continue
        from_svc, to_svc = parts

        verdict_data: dict | None = None
        vf = hop_dir / "verdict.json"
        if vf.exists():
            try:
                verdict_data = json.loads(vf.read_text())
            except Exception:  # noqa: BLE001
                pass

        if not verdict_data:
            obs_dir = hop_dir / ".agentm" / "observability"
            if obs_dir.exists():
                verdict_data = extract_hop_verdict(obs_dir)
                if verdict_data:
                    vf.write_text(json.dumps(
                        verdict_data, ensure_ascii=False, indent=2,
                    ))

        if verdict_data:
            verdicts.append({
                "from": from_svc,
                "to": to_svc,
                "verdict": verdict_data.get("verdict", "unknown"),
                "rationale": verdict_data.get("rationale", ""),
                "claim": verdict_data.get("claim", ""),
                "symptom_evidence": verdict_data.get("symptom_evidence", []),
            })
    return verdicts
