"""Judge agent: review all hop verdicts for a completed case."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from graph import SYNTHETIC, _duckdb_conn
from injection import get_injections
from verdict import collect_all_verdicts, extract_hop_verdict

REPO = Path(__file__).resolve().parents[4]


def _get_system_throughput(data_dir: Path) -> dict:
    """Compare load-generator root span counts between windows."""
    conn = _duckdb_conn(data_dir)
    result: dict = {}
    try:
        for w in ("normal", "abnormal"):
            row = conn.execute(
                f"SELECT COUNT(*) FROM {w}_traces "
                "WHERE (parent_span_id = '' OR parent_span_id IS NULL) "
                f"AND service_name IN {str(tuple(SYNTHETIC))}"
            ).fetchone()
            result[w] = row[0] if row else 0
    except Exception:  # noqa: BLE001
        pass
    conn.close()
    return result


def run_judge(
    case_dir: Path,
    run_dir: Path,
    *,
    budget: int = 20,
) -> dict:
    """Run a judge agent on a completed case to review all hop verdicts."""
    data_dir = case_dir.resolve()
    out = run_dir.resolve()

    # Collect inputs
    all_verdicts_path = out / "all_verdicts.json"
    if not all_verdicts_path.exists():
        all_verdicts = collect_all_verdicts(out)
        all_verdicts_path.write_text(
            json.dumps(all_verdicts, indent=2, ensure_ascii=False)
        )
    else:
        all_verdicts = json.loads(all_verdicts_path.read_text())

    trace = json.loads((out / "propagation_trace.json").read_text())
    confirmed = trace.get("confirmed_nodes", [])
    injections = get_injections(data_dir)
    throughput = _get_system_throughput(data_dir)

    # Build the judge prompt
    seeds = {i["target"] for i in injections}
    inj_lines = [
        f"- {i['target']} ({i['chaos_type']})" for i in injections
    ]

    tp_normal = throughput.get("normal", 0)
    tp_abnormal = throughput.get("abnormal", 0)
    tp_drop = ((tp_normal - tp_abnormal) / tp_normal * 100
               if tp_normal > 0 else 0)

    # Index hop verdicts by target for evidence lookup.
    verdict_by_target: dict[str, dict] = {v["to"]: v for v in all_verdicts}

    def _ev_claims(svc: str) -> str:
        v = verdict_by_target.get(svc, {})
        claims = [
            e.get("claim", "") for e in v.get("symptom_evidence", [])
            if e.get("claim")
        ]
        return "; ".join(claims[:4])

    confirmed_nonseed = [s for s in confirmed if s not in seeds]
    confirmed_lines: list[str] = []
    for s in confirmed_nonseed:
        v = verdict_by_target.get(s, {})
        frm = v.get("from", "?")
        confirmed_lines.append(
            f"- {frm} → **{s}**: {v.get('rationale', '(no rationale)')}\n"
            f"    evidence: {_ev_claims(s) or '(none)'}"
        )
    confirmed_block = "\n".join(confirmed_lines) or "(none)"

    rejected_lines: list[str] = []
    for v in all_verdicts:
        if v["verdict"] == "rejected" and v["to"] not in confirmed:
            rejected_lines.append(
                f"- {v['from']} → {v['to']}: {v['rationale']}"
            )
    rejected_block = "\n".join(rejected_lines) or "(none)"

    prompt = f"""\
You are the lead auditor of a fault-propagation graph that independent
hop agents built one edge at a time. Each hop agent ran careful
per-edge analysis — checking error rate, latency magnitude, fault
mechanism, and relationship direction — and already rejected
throughput-only drops and noise. **Their confirmations are
authoritative; you do NOT remove them.**

Your ONLY job is to catch what no single edge could see: a system-wide
CASCADE in which services the hop agents rejected for "fewer calls /
throughput drop" are in fact genuinely unavailable because the whole
system is collapsing.

## Fault injection
{chr(10).join(inj_lines)}

## System-wide load (the cascade signal)
- load-generator root spans: normal {tp_normal} → abnormal {tp_abnormal} (drop {tp_drop:.1f}%)
- If drop > 80%: the system is in cascading collapse. Review the
  rejected list and ADD any service that is down because the whole
  system is down (was actively serving, now silent/erroring).
- If drop <= 80%: there is NO cascade. ADD nothing — a throughput drop
  is just the caller sending fewer requests.

## Confirmed services (context — do NOT change these) ({len(confirmed_nonseed)})
{confirmed_block}

## Rejected services — ADD only under a real cascade ({len(rejected_lines)})
{rejected_block}

## Decide
- Leave `remove` EMPTY. The per-edge analysis is authoritative for
  what is degraded; second-guessing it from rationale text alone
  removes genuinely-degraded services and corrupts the graph.
- ADD a rejected service only if a system-wide cascade (loadgen drop
  > 80%) makes it genuinely unavailable, not merely less-called. Use
  `list_tables` / `query_sql` to confirm; state latencies in ms/s
  (duration is nanoseconds).

Most reviews add nothing. Call `submit_judge_review` with `add` (and
`remove` empty) plus `rationale`.
"""

    judge_dir = out / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["AGENTM_PROJECT_ROOT"] = str(REPO)
    env["AGENTM_RCA_DATA_DIR"] = str(data_dir)
    base = (
        ["agentm"]
        if shutil.which("agentm")
        else ["uv", "run", "--no-sync", "agentm"]
    )
    cmd = [
        *base, "--scenario", "verifier",
        "--model", env.get("AGENTM_MODEL", "doubao"),
        "--cwd", str(judge_dir),
        "--max-tool-calls", str(budget),
        "-p", prompt,
    ]

    with open(judge_dir / "stdout.log", "w") as fout, \
         open(judge_dir / "stderr.log", "w") as ferr:
        subprocess.run(cmd, env=env, stdout=fout, stderr=ferr)

    # Extract judge review (bidirectional: remove confirmed, add rejected)
    obs_dir = judge_dir / ".agentm" / "observability"
    verdict = (
        extract_hop_verdict(obs_dir, "submit_judge_review", "remove")
        if obs_dir.exists() else None
    )
    if verdict:
        (judge_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False)
        )

    confirmed_set = set(confirmed)
    # Judge is promotion-only: hop confirmations are authoritative. LLM
    # pruning from rationale text alone is unreliable (it removes genuine
    # SLOW/ERROR services), so we record any suggested removal for audit
    # but do NOT apply it.
    suggested_remove = sorted(({
        s.strip() for s in (verdict.get("remove", []) if verdict else [])
        if isinstance(s, str) and s.strip()
    } & confirmed_set) - seeds)
    removed: set[str] = set()
    # add: rejected services to promote, not already confirmed, not seeds.
    added = {
        s.strip() for s in (verdict.get("add", []) if verdict else [])
        if isinstance(s, str) and s.strip()
    }
    added -= confirmed_set
    added -= seeds

    final_confirmed = sorted((confirmed_set - removed) | added)
    final_propagated = [s for s in final_confirmed if s not in seeds]
    judge_rationale = verdict.get("rationale", "") if verdict else ""

    # Build per-node entry with provenance + evidence
    nodes: list[dict] = []
    for svc in final_confirmed:
        if svc in seeds:
            nodes.append({"service": svc, "source": "injection_seed"})
        elif svc in added:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "judge_promoted",
                "hop_verdict": hop_v.get("verdict", ""),
                "hop_rationale": hop_v.get("rationale", ""),
                "hop_evidence": hop_v.get("symptom_evidence", []),
                "judge_rationale": judge_rationale,
            })
        else:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "hop_agent",
                "rationale": hop_v.get("rationale", ""),
                "symptom_evidence": hop_v.get("symptom_evidence", []),
            })

    # Build edges, dropping any whose target the judge removed.
    final_set = set(final_confirmed)
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    for h in trace.get("hop_log", []):
        frm, to = h.get("from", ""), h.get("to", "")
        if not frm or not to or to not in final_set:
            continue
        v = h.get("verdict", "")
        if v in ("confirmed", "edge_sql") and (frm, to) not in seen_edges:
            seen_edges.add((frm, to))
            hop_v = verdict_by_target.get(to, {})
            edge_entry: dict = {
                "from": frm, "to": to,
                "source": "hop_agent" if v == "confirmed" else "edge_sql",
            }
            if v == "confirmed":
                edge_entry["rationale"] = hop_v.get("rationale", "")
                edge_entry["symptom_evidence"] = hop_v.get(
                    "symptom_evidence", [],
                )
            edges.append(edge_entry)

    # Edges for judge-promoted nodes
    for svc in added:
        promoted_v = verdict_by_target.get(svc)
        if promoted_v and (promoted_v["from"], svc) not in seen_edges:
            seen_edges.add((promoted_v["from"], svc))
            edges.append({
                "from": promoted_v["from"], "to": svc,
                "source": "judge_promoted",
                "hop_rationale": promoted_v.get("rationale", ""),
                "hop_evidence": promoted_v.get("symptom_evidence", []),
            })

    final = {
        "seeds": sorted(seeds),
        "confirmed_nodes": final_confirmed,
        "propagated": final_propagated,
        "edges": edges,
        "nodes": nodes,
        "judge_rationale": judge_rationale,
        "removed": sorted(removed),
        "suggested_remove": suggested_remove,
        "added": sorted(added),
    }
    (out / "final_propagation.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )

    return verdict or {}
