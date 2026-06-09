"""Verdict extraction from workflow propagation graph output."""
from __future__ import annotations


def verdicts_from_trace(trace: dict) -> list[dict]:
    """Extract all hop verdicts from a propagation graph (workflow output).

    The workflow ``agent()`` return values are captured in ``hop_log``
    and ``node_evidence``.
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
