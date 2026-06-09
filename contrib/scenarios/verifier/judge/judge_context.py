"""Inject whole-graph review context into the judge agent session.

Reads structured config describing the propagation results (injections,
confirmed services, rejected verdicts, throughput) and builds the full
domain context, appending it to the system prompt so the agent starts
with complete case-specific knowledge.
"""
from __future__ import annotations

from typing import Required, TypedDict

from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="judge_context",
    description="Inject whole-graph review context into the judge agent.",
    registers=("event:before_agent_start",),
    config_schema={
        "type": "object",
        "properties": {
            "injections": {
                "type": "array",
                "items": {"type": "object"},
            },
            "confirmed": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rejected_verdicts": {
                "type": "array",
                "items": {"type": "object"},
            },
            "throughput": {
                "type": "object",
                "additionalProperties": True,
            },
            "seeds": {
                "type": "array",
                "items": {"type": "string"},
            },
            "verdict_by_target": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
        },
        "required": ["injections", "confirmed"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------


class Injection(TypedDict, total=False):
    target: Required[str]
    chaos_type: Required[str]
    params: str


class SymptomEvidence(TypedDict, total=False):
    sql: str
    claim: str


JudgeTargetVerdict = TypedDict(
    "JudgeTargetVerdict",
    {
        "from": str,
        "to": str,
        "verdict": str,
        "rationale": str,
        "symptom_evidence": list[SymptomEvidence],
    },
    total=False,
)


class ThroughputSummary(TypedDict, total=False):
    normal: float
    abnormal: float


class JudgeContextConfig(TypedDict, total=False):
    injections: Required[list[Injection]]
    confirmed: Required[list[str]]
    rejected_verdicts: list[JudgeTargetVerdict]
    throughput: ThroughputSummary
    seeds: list[str]
    verdict_by_target: dict[str, JudgeTargetVerdict] | None


def _build_judge_prompt(
    injections: list[Injection],
    confirmed: list[str],
    rejected_verdicts: list[JudgeTargetVerdict],
    throughput: ThroughputSummary,
    seeds: set[str],
    verdict_by_target: dict[str, JudgeTargetVerdict] | None = None,
) -> str:
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


# ---------------------------------------------------------------
# Atom install
# ---------------------------------------------------------------

def install(api: ExtensionAPI, config: JudgeContextConfig) -> None:
    injections = config.get("injections", [])
    confirmed = config.get("confirmed", [])
    if not injections:
        return

    seeds_list: list[str] = config.get("seeds") or [
        i["target"] for i in injections if i.get("target")
    ]
    seeds = set(seeds_list)

    context = _build_judge_prompt(
        injections=injections,
        confirmed=confirmed,
        rejected_verdicts=config.get("rejected_verdicts", []),
        throughput=config.get("throughput", {}),
        seeds=seeds,
        verdict_by_target=config.get("verdict_by_target"),
    )

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__ = ["MANIFEST", "install"]
