"""Judge agent prompt construction for the verifier_judge scenario."""

from __future__ import annotations


def build_judge_prompt(
    injections: list[dict[str, str]],
    confirmed: list[str],
    rejected_verdicts: list[dict],
    throughput: dict,
    seeds: set[str],
    verdict_by_target: dict[str, dict] | None = None,
) -> str:
    """Build the full user prompt for the judge review agent.

    *verdict_by_target* maps service name to its hop verdict dict (used for
    evidence lookup in the confirmed/rejected blocks). When ``None``, evidence
    claims are omitted.
    """
    verdict_by_target = verdict_by_target or {}

    inj_lines = [
        f"- {i['target']} ({i['chaos_type']})" for i in injections
    ]

    tp_normal = throughput.get("normal", 0)
    tp_abnormal = throughput.get("abnormal", 0)
    tp_drop = ((tp_normal - tp_abnormal) / tp_normal * 100
               if tp_normal > 0 else 0)

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
    for v in rejected_verdicts:
        rejected_lines.append(
            f"- {v['from']} → {v['to']}: {v['rationale']}"
        )
    rejected_block = "\n".join(rejected_lines) or "(none)"

    return f"""\
Review the fault-propagation graph built by independent hop agents.

## Fault injection
{chr(10).join(inj_lines)}

## System-wide load (the cascade signal)
- load-generator root spans: normal {tp_normal} → abnormal {tp_abnormal} (drop {tp_drop:.1f}%)
- Examine the data yourself to decide whether a system-wide cascade is
  occurring. A large throughput drop MAY indicate cascading collapse, but
  use your own judgement — query the rejected services' own metrics to
  confirm genuine unavailability before promoting.

## Confirmed services (context — do NOT change these) ({len(confirmed_nonseed)})
{confirmed_block}

## Rejected services — ADD only under a real cascade ({len(rejected_lines)})
{rejected_block}

## Decide
- Leave `remove` EMPTY. The per-edge analysis is authoritative for
  what is degraded; second-guessing it from rationale text alone
  removes genuinely-degraded services and corrupts the graph.
- ADD a rejected service only if genuine system-wide cascade makes it
  unavailable, not merely less-called. Use `list_tables` / `query_sql`
  to confirm; state latencies in ms/s (duration is nanoseconds).

Most reviews add nothing. Call `submit_judge_review` with `add` (and
`remove` empty) plus `rationale`.
"""


__all__ = ["build_judge_prompt"]
