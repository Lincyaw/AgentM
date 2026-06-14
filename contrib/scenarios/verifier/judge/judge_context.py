"""Inject whole-graph review context into the judge agent session.

Reads structured config describing the propagation results (injections,
confirmed services, rejected verdicts with their fpg evidence) and
builds the full domain context, appending it to the system prompt so
the agent starts with complete case-specific knowledge.
"""
from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from fpg import Evidence


class InjectionInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target: str
    chaos_type: str
    params: str = ""


class TableProfile(BaseModel):
    """Mechanical profile of one normal/abnormal table pair."""

    model_config = ConfigDict(extra="ignore")
    columns: list[str] = Field(default_factory=list)
    # column -> window ("normal"/"abnormal") -> value -> count
    value_distributions: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )


class TargetVerdict(BaseModel):
    """One hop verdict as seen by the judge (fpg evidence attached)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    from_service: str = Field(default="", alias="from")
    to: str = ""
    verdict: str = ""
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class VanishedEndpoint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    span_name: str
    normal: int
    abnormal: int


class ThroughputSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    normal: float = 0.0
    abnormal: float = 0.0


class JudgeContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    injections: list[InjectionInfo]
    confirmed: list[str]
    inconclusive_verdicts: list[TargetVerdict] = Field(default_factory=list)
    rejected_verdicts: list[TargetVerdict] = Field(default_factory=list)
    throughput: ThroughputSummary = Field(default_factory=ThroughputSummary)
    seeds: list[str] = Field(default_factory=list)
    verdict_by_target: dict[str, TargetVerdict] = Field(default_factory=dict)
    # Mechanically profiled dataset shape, same as the hop agents get
    dataset_profile: dict[str, TableProfile] = Field(default_factory=dict)
    # Mechanically detected: per rejected service, endpoints with spans
    # in the normal window but ZERO in the abnormal window
    vanished_endpoints: dict[str, list[VanishedEndpoint]] = Field(
        default_factory=dict
    )
    # Mechanically derived: services no other (non-synthetic) service
    # calls — their callers are end users / the load generator
    entry_services: list[str] = Field(default_factory=list)


MANIFEST = ExtensionManifest(
    name="judge_context",
    description="Inject whole-graph review context into the judge agent.",
    registers=("event:before_agent_start",),
    config_schema=JudgeContextConfig,
)

# ---------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------

def _format_dist(dist: dict[str, int], limit: int = 6) -> str:
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    text = ", ".join(f"{v}={c}" for v, c in items[:limit])
    if len(items) > limit:
        text += f", … (+{len(items) - limit} more)"
    return text


def _format_dataset_profile(profile: dict[str, TableProfile]) -> str:
    lines: list[str] = []
    for base, tp in profile.items():
        lines.append(f"### {base} (tables normal_{base} / abnormal_{base})")
        lines.append("columns: " + ", ".join(tp.columns))
        for col, windows in tp.value_distributions.items():
            normal = _format_dist(windows.get("normal", {}))
            abnormal = _format_dist(windows.get("abnormal", {}))
            marker = (
                "  <-- CHANGED"
                if windows.get("normal", {}).keys()
                != windows.get("abnormal", {}).keys()
                else ""
            )
            lines.append(
                f"- {col}: normal [{normal}] | abnormal [{abnormal}]{marker}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_judge_prompt(
    injections: list[InjectionInfo],
    confirmed: list[str],
    inconclusive_verdicts: list[TargetVerdict],
    rejected_verdicts: list[TargetVerdict],
    throughput: ThroughputSummary,
    seeds: set[str],
    verdict_by_target: dict[str, TargetVerdict],
    dataset_profile: dict[str, TableProfile],
    vanished_endpoints: dict[str, list[VanishedEndpoint]],
    entry_services: list[str],
) -> str:
    inj_lines = [f"- {i.target} ({i.chaos_type})" for i in injections]

    tp_drop = (
        (throughput.normal - throughput.abnormal) / throughput.normal * 100
        if throughput.normal > 0 else 0
    )

    def _ev_claims(svc: str) -> str:
        v = verdict_by_target.get(svc)
        if v is None:
            return ""
        claims = [e.explanation for e in v.evidence if e.explanation]
        return "; ".join(claims[:4])

    confirmed_nonseed = [s for s in confirmed if s not in seeds]
    confirmed_lines: list[str] = [
        f"- **{s}** (injection seed)" for s in sorted(seeds) if s in confirmed
    ]
    for s in confirmed_nonseed:
        v = verdict_by_target.get(s)
        frm = v.from_service if v else "?"
        rationale = v.rationale if v else "(no rationale)"
        confirmed_lines.append(
            f"- {frm} → **{s}**: {rationale}\n"
            f"    evidence: {_ev_claims(s) or '(none)'}"
        )
    confirmed_block = "\n".join(confirmed_lines) or "(none)"

    inconclusive_lines = [
        f"- {v.from_service} → {v.to}: {v.rationale}"
        for v in inconclusive_verdicts
    ]
    inconclusive_block = "\n".join(inconclusive_lines) or "(none)"

    rejected_lines = [
        f"- {v.from_service} → {v.to}: {v.rationale}"
        for v in rejected_verdicts
    ]
    rejected_block = "\n".join(rejected_lines) or "(none)"

    entries = set(entry_services)
    vanished_block = ""
    if vanished_endpoints:
        lines = []
        for svc, eps in vanished_endpoints.items():
            tag = " (ENTRY service — its callers are end users)" if svc in entries else ""
            lines.append(f"- **{svc}**{tag}:")
            for ep in eps:
                lines.append(
                    f"    - {ep.span_name}: normal={ep.normal} -> abnormal=0"
                )
        vanished_block = (
            "\n## Vanished endpoints (mechanically detected on REJECTED "
            "services)\nEndpoints with spans in the normal window and ZERO "
            "in the abnormal window:\n" + "\n".join(lines) + "\n"
        )

    profile_block = ""
    if dataset_profile:
        profile_block = (
            "\n## Dataset shape (mechanically profiled — authoritative for "
            "THIS dataset)\n"
            + _format_dataset_profile(dataset_profile)
            + "\n"
        )

    return f"""\
Review the fault-propagation graph built by independent hop agents.

## Fault injection
{chr(10).join(inj_lines)}

## System-wide load (the cascade signal)
- load-generator root spans: normal {throughput.normal} → abnormal {throughput.abnormal} (drop {tp_drop:.1f}%)
- Examine the data yourself to decide whether a system-wide cascade is
  occurring. A large throughput drop MAY indicate cascading collapse, but
  use your own judgement — query the rejected services' own metrics to
  confirm genuine unavailability before promoting.
{profile_block}{vanished_block}
## Confirmed graph — do NOT change these; ALL of them (seeds included) \
are valid `via_service` anchors ({len(confirmed_nonseed) + len(seeds & set(confirmed))})
{confirmed_block}

## Inconclusive edges — hop agents could not decide ({len(inconclusive_lines)})
{inconclusive_block}

## Rejected services — review for suspicious rejections ({len(rejected_lines)})
{rejected_block}

## Global patterns to look for
Hop agents judge one edge at a time from the target's OWN aggregate
signals; two genuine degradation patterns are invisible at that zoom:

1. **System-wide cascade**: services rejected for "fewer calls /
   throughput drop" that are in fact unavailable because the whole
   system is collapsing. Confirm via load-generator root spans and the
   rejected services' own metrics.
2. **Fault-path flow disappearance at ENTRY services**: the harness
   already detected the vanished endpoints (section above) and marked
   which services are entries. For mid-chain services, "fewer calls"
   is indeed the caller's problem — do not promote those. But for an
   ENTRY service that rule CANNOT apply: its callers are end users,
   there is no upstream service to attribute the drop to. A vanished
   user-facing endpoint at an entry means the user flow itself broke
   — that is exactly how the injected fault surfaces as user impact.
   Your judgment call is only: does the vanished endpoint's chain
   lead to an injected fault (the path often names the dependency,
   e.g. /api/v1/foodservice/... -> ts-food-service; multi-step user
   flows also break when an EARLIER step in the same flow hits the
   fault)? If yes, promote the entry with predicate flow_interrupted,
   via_service = the confirmed service its broken flow depends on.
   Sibling endpoints speeding up strengthens the signal (fail-fast /
   load shift), it does not weaken it.

## Decide
- **Confirmations are final** — you only ADD or RE-EVALUATE.
  (`suggested_remove` is audit-only and never applied.)
- **`re_evaluate`** (PREFERRED): send inconclusive or suspiciously-
  rejected edges back to a hop agent with your global context. The
  hop agent will re-query data and make the final call. Provide a
  clear `context` explaining what to reconsider (e.g. "upstream
  ts-food-service is confirmed dead with URL mutation — zero spans on
  ts-preserve-service is expected cascade, not just fewer calls.
  Check the specific endpoint that routes through ts-food-service.").
- **`add`** (direct promotion): use only when you have enough
  global evidence without needing re-investigation (e.g. system-wide
  cascade with load-gen throughput collapse > 80%).
- Every `add` must name `via_service` and `predicate`. Every
  `re_evaluate` must name `via_service` and `context`.
- Verify with `list_tables` / `query_sql` before deciding.

Call `submit_judge_review` with your decisions.
"""

# ---------------------------------------------------------------
# Atom install
# ---------------------------------------------------------------

def install(api: ExtensionAPI, config: JudgeContextConfig) -> None:
    if not config.injections:
        return

    seeds = set(config.seeds) or {
        i.target for i in config.injections if i.target
    }

    context = _build_judge_prompt(
        injections=config.injections,
        confirmed=config.confirmed,
        inconclusive_verdicts=config.inconclusive_verdicts,
        rejected_verdicts=config.rejected_verdicts,
        throughput=config.throughput,
        seeds=seeds,
        verdict_by_target=config.verdict_by_target,
        dataset_profile=config.dataset_profile,
        vanished_endpoints=config.vanished_endpoints,
        entry_services=config.entry_services,
    )

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)

__all__: Final = ["MANIFEST", "install"]
