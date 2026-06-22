"""Fault propagation verification workflow (module mode), fpg-native.

The workflow treats verifier agents as callable map functions. Seed and
hop agents maximize discovery recall; gate agents audit each local
investigation; audit agents reduce the merged evidence ledger into a
global decision and concrete rework. The internal state is the FPG graph:
nodes are FPG EventNode dicts and edges are FPG Edge dicts, built
incrementally as seeds/hops confirm. Structural rules are enforced at
construction time:

  - injection seeds are verified independently, but a seed-to-seed edge may
    still be accepted when a hop agent confirms that one injected fault
    propagated into another injected target;
  - an edge that would close a cycle is never evaluated;
  - EVERY edge goes through a hop agent — including edges between two
    already-confirmed services (topological adjacency alone is not
    causal evidence);
  - global audit may request focused seed/hop rechecks or remove causal
    paths that fail the evidence audit.

Input via ``ctx.args`` (built by ``prepare.CaseContext.to_workflow_args``):
    data_dir, graph, injections, infra_nodes, fault_docs,
    budget, out_dir, skip_propagate, skip_judge,
    window               -- fpg TimeInterval dict for this case
    rel_mechanism        -- {rel_type: fpg edge mechanism}
    existing_state       -- {nodes, edges, verdicts} for skip_propagate

Output: dict with nodes (list of fpg EventNode dicts), edges (list of
fpg Edge dicts), verdicts, hop_log, rounds, gate_log, audit_rounds, and
the latest audit result.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Sequence
from typing import Annotated, Any, Literal, Required, TypeGuard, TypedDict, cast

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext
from pydantic import BaseModel, ConfigDict, Field

from .hop.hop_context import PriorVerdict


class Injection(TypedDict, total=False):
    target: Required[str]
    chaos_type: Required[str]
    params: str
    node_id: str
    subject: str
    target_entity: str
    effect_target: str
    edge_source: str
    edge_target: str


HopLogEntry = TypedDict(
    "HopLogEntry",
    {"round": int, "from": str, "to": str, "verdict": str},
)


class SeedResult(TypedDict, total=False):
    verdict: Required[str]
    effect_target: str | None
    predicate: str | None
    rationale: str
    evidence: list[dict[str, Any]]
    claim: str
    gate: dict[str, Any]
    _error: str


class HopResult(TypedDict, total=False):
    verdict: Required[str]
    predicate: str | None
    rationale: str
    evidence: list[dict[str, Any]]
    relationship: dict[str, Any] | None
    claim: str
    gate: dict[str, Any]
    _error: str


class ExecutionError(TypedDict):
    stage: str
    item: str
    reason: str


SeedParallelItem = tuple[Injection, SeedResult | None]
HopReworkParallelItem = tuple["HopRecheckRequest", HopResult | None]


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool = Field(
        description="True when the seed/hop investigation is complete enough "
        "for reduction."
    )
    retryable: bool = Field(
        default=False,
        description="True when rerunning the same seed/hop with focused "
        "feedback can plausibly repair the missing investigation.",
    )
    missing_checks: list[str] = Field(default_factory=list)
    retry_prompt: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""


class SeedRecheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["seed_recheck"]
    seed: str
    context: str = ""


class HopRecheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["hop_recheck"]
    from_service: str
    to_service: str
    context: str = ""


ReworkRequest = Annotated[
    SeedRecheckRequest | HopRecheckRequest,
    Field(discriminator="kind"),
]


class EdgeDrop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    seed: str | None = Field(
        default=None,
        description="Seed lineage for the invalid path, when known.",
    )
    path_id: str | None = Field(
        default=None,
        description="Candidate path id that contains this edge, when known.",
    )
    reason: str = ""


SeedCoverageStatus = Literal[
    "explains_entry",
    "local_only",
    "benign_or_no_effect",
    "needs_recheck",
    "invalid_path",
]


class AnomalyAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str
    meaningful_anomalies: list[str] = Field(default_factory=list)
    explained: list[str] = Field(default_factory=list)
    unexplained: list[str] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


class CausalAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    path: list[str] = Field(default_factory=list)
    verdict: Literal["valid", "invalid", "needs_recheck"] = "needs_recheck"
    invalid_reason: str | None = None
    weakest_edge: EdgeDrop | None = None
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


class SeedCoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: str
    coverage: SeedCoverageStatus
    explained_entry_observations: list[str] = Field(default_factory=list)
    local_effect_observations: list[str] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


class GlobalAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    seed_coverage: dict[str, SeedCoverageStatus] = Field(default_factory=dict)
    unexplained_anomalies: list[str] = Field(default_factory=list)
    invalid_causal_paths: list[str] = Field(default_factory=list)
    drop_edges: list[EdgeDrop] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    stop_reason: str | None = None
    rationale: str = ""


def _parallel_items(raw: object) -> list[Any]:
    """Normalize workflow parallel output after per-item agent failures."""
    return raw if isinstance(raw, list) else []


def _is_seed_parallel_item(value: object) -> TypeGuard[SeedParallelItem]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], dict)
        and (value[1] is None or isinstance(value[1], dict))
    )


def _is_hop_rework_parallel_item(
    value: object,
) -> TypeGuard[HopReworkParallelItem]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], HopRecheckRequest)
        and (value[1] is None or isinstance(value[1], dict))
    )


def _truncate_text(value: object, limit: int = 1600) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _compact_query(query: object) -> dict[str, str]:
    if not isinstance(query, dict):
        return {}
    statement = ""
    raw_statement = query.get("statement")
    if raw_statement is not None:
        statement = _truncate_text(raw_statement, 1200)
    language = query.get("language")
    return {
        "language": str(language or "sql"),
        "statement": statement,
    }


def _compact_evidence_item(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"value": _truncate_text(item, 400)}
    compact: dict[str, Any] = {}
    if item.get("explanation"):
        compact["explanation"] = _truncate_text(item["explanation"], 500)
    query = _compact_query(item.get("query"))
    if query:
        compact["query"] = query
    return compact


def _compact_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep retry context useful without replaying an entire trajectory."""
    compact: dict[str, Any] = {}
    for key in (
        "verdict",
        "effect_target",
        "predicate",
        "claim",
        "rationale",
        "investigation_coverage",
    ):
        value = result.get(key)
        if value:
            compact[key] = value if isinstance(value, dict) else _truncate_text(value)
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        compact["evidence"] = [
            _compact_evidence_item(item) for item in evidence[:10]
        ]
        if len(evidence) > 10:
            compact["evidence_truncated"] = len(evidence) - 10
    relationship = result.get("relationship")
    if isinstance(relationship, dict):
        compact["relationship"] = _compact_evidence_item(relationship)
    return compact


def _retry_context(
    *,
    base_context: str,
    label: str,
    submitted_result: dict[str, Any],
    gate: GateDecision,
) -> str:
    """Build the context passed to the next clean retry session."""
    payload = {
        "failed_attempt": label,
        "previous_submitted_result": _compact_agent_result(submitted_result),
        "gate_decision": {
            "accepted": gate.accepted,
            "confidence": gate.confidence,
            "missing_checks": gate.missing_checks,
            "rationale": _truncate_text(gate.rationale),
            "retry_prompt": _truncate_text(gate.retry_prompt or "", 2400),
        },
    }
    block = (
        "## Previous attempt context\n"
        "This is a new child session, but the previous attempt is summarized below. "
        "Use it as a map of what was already checked and why gate rejected it. "
        "Do not simply repeat the previous evidence; rerun or repair SQL as needed, "
        "then fill only the missing checks.\n"
        "```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + "\n```"
    )
    return (base_context + "\n\n" if base_context else "") + block


class PropagationResult(TypedDict, total=False):
    nodes: Required[list[dict[str, Any]]]
    edges: Required[list[dict[str, Any]]]
    verdicts: Required[dict[str, HopResult]]
    hop_log: Required[list[HopLogEntry]]
    rounds: Required[int]
    unreachable_seeds: list[str]
    reachability_warnings: list[str]
    seed_verdicts: dict[str, SeedResult]
    confirmed_seeds: list[str]
    gate_log: list[dict[str, Any]]
    audit: dict[str, Any]
    audit_rounds: list[dict[str, Any]]
    execution_errors: list[ExecutionError]


def _reaches(adj: dict[str, list[str]], src: str, dst: str) -> bool:
    stack, seen = [src], {src}
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _injection_node_id(inj: Injection) -> str:
    return inj.get("node_id") or inj["target"]


def _injection_subject(inj: Injection) -> str:
    return inj.get("subject") or inj.get("target_entity") or f"svc:{inj['target']}"


def _injection_effect_target(inj: Injection) -> str:
    return inj.get("effect_target") or inj["target"]


def _seed_effect_target(inj: Injection, verdict: SeedResult) -> str:
    """Return the observed service-side symptom target for a seed."""
    reported = verdict.get("effect_target")
    if isinstance(reported, str) and reported.strip():
        return reported.strip()
    return _injection_effect_target(inj)


def _is_link_injection(inj: Injection) -> bool:
    return _injection_subject(inj).startswith("link:")


def _fault_record(inj: Injection) -> list[str]:
    return [inj["chaos_type"], _injection_node_id(inj), inj.get("params", "")]


def _link_root_predicate(inj: Injection) -> str:
    chaos_type = inj.get("chaos_type", "").lower()
    if "partition" in chaos_type:
        return "network_partitioned"
    return "network_degraded"


def _node_from_seed(
    inj: Injection,
    verdict: SeedResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build an fpg EventNode dict from a confirmed seed verdict."""
    predicate = verdict.get("predicate") or (
        _link_root_predicate(inj) if _is_link_injection(inj) else "other"
    )
    node: dict[str, Any] = {
        "kind": "event",
        "id": _injection_node_id(inj),
        "subject": _injection_subject(inj),
        "predicate": predicate,
        "time": window,
        "grounding": "observed",
        "evidence": list(verdict.get("evidence", [])),
        "annotation": "auto",
    }
    if predicate == "other":
        node["description"] = verdict.get("rationale", verdict.get("claim", "inconclusive"))
    return node


def _node_from_link_effect(
    inj: Injection,
    verdict: SeedResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build the service-side symptom node for a confirmed link injection."""
    svc = _seed_effect_target(inj, verdict)
    predicate = verdict.get("predicate") or "flow_interrupted"
    node: dict[str, Any] = {
        "kind": "event",
        "id": svc,
        "subject": f"svc:{svc}",
        "predicate": predicate,
        "time": window,
        "grounding": "observed",
        "evidence": list(verdict.get("evidence", [])),
        "annotation": "auto",
    }
    if predicate == "other":
        node["description"] = verdict.get("rationale", verdict.get("claim", "inconclusive"))
    return node


def _node_from_verdict(
    svc: str,
    verdict: HopResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build an fpg EventNode dict from a confirmed hop verdict."""
    evidence = list(verdict.get("evidence", []))
    relationship = verdict.get("relationship")
    if relationship:
        evidence.append(
            {
                "query": relationship["query"],
                "explanation": "call relationship with the confirmed upstream: "
                + relationship.get("explanation", "see query"),
            }
        )
    predicate = verdict.get("predicate") or "other"
    node: dict[str, Any] = {
        "kind": "event",
        "id": svc,
        "subject": f"svc:{svc}",
        "predicate": predicate,
        "time": window,
        "grounding": "observed" if evidence else "latent",
        "evidence": evidence,
        "annotation": "auto",
    }
    if predicate == "other":
        node["description"] = verdict.get("claim") or verdict.get("rationale", "")
    return node


def _edge_dict(
    src: str,
    dst: str,
    rel_type: str,
    rel_mechanism: dict[str, str],
    claim: str,
) -> dict[str, Any]:
    mechanism = rel_mechanism.get(rel_type, "other")
    edge: dict[str, Any] = {
        "src": src,
        "dst": dst,
        "mechanism": mechanism,
        "verification": "consistency-checked",
    }
    if mechanism == "other":
        edge["description"] = (
            claim or f"relationship type {rel_type!r} outside the vocabulary"
        )
    elif claim:
        edge["description"] = claim
    return edge


def _dump_model(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


def _agent_task_prompt(label: str) -> str:
    return f"Run verifier task `{label}` using the injected atom_config context."


def _structured_task_prompt(label: str) -> str:
    return f"Run structured verifier task `{label}` using atom_config."


def _find_child_session(
    ctx: WorkflowContext,
    label: str,
) -> dict[str, Any] | None:
    for child in reversed(ctx.child_sessions):
        if child.get("trace_label") == label or child.get("workflow_node_id") == label:
            return dict(child)
    return None


async def run(ctx: WorkflowContext) -> PropagationResult:
    args = ctx.args
    skip_propagate: bool = args.get("skip_propagate", False)
    injections = cast(list[Injection], args["injections"])
    window = cast(dict[str, str], args["window"])
    rel_mechanism = cast(dict[str, str], args.get("rel_mechanism", {}))
    seeds = {_injection_node_id(inj) for inj in injections if inj.get("target")}

    nodes: dict[str, dict[str, Any]]  # svc -> fpg EventNode dict
    edges: list[dict[str, Any]]
    adj: dict[str, list[str]]  # accepted-edge adjacency, for cycle guard
    in_deg: dict[str, int]
    verdicts: dict[str, HopResult]  # "from__to" -> hop verdict
    hop_log: list[HopLogEntry]
    round_n: int
    node_fault: dict[str, list[str]]
    seed_verdicts: dict[str, SeedResult]
    confirmed_seed_ids: set[str]
    execution_errors: dict[str, ExecutionError] = {}

    graph: dict[str, list[list[str]]] = args["graph"]
    infra_set = set(args.get("infra_nodes", []))
    data_dir: str = args["data_dir"]
    skip_judge: bool = args.get("skip_judge", False)
    judge_model_raw = args.get("judge_model")
    judge_model = (
        judge_model_raw.strip()
        if isinstance(judge_model_raw, str) and judge_model_raw.strip()
        else None
    )
    fault_docs: dict[str, str] = args.get("fault_docs", {})
    audit_result: GlobalAudit | None = None
    audit_rounds_log: list[dict[str, Any]] = []
    gate_log: list[dict[str, Any]] = []
    gate_retries = int(args.get("gate_retries", 3))
    agent_retries = int(args.get("agent_retries", 3))
    max_audit_rounds = int(args.get("max_audit_rounds", 3))

    def _execution_key(stage: str, item: str) -> str:
        return stage + ":" + item

    def _record_execution_error(stage: str, item: str, reason: str) -> None:
        execution_errors[_execution_key(stage, item)] = {
            "stage": stage,
            "item": item,
            "reason": reason,
        }
        ctx.log(f"⚠ {stage} {item}: {reason}")

    def _clear_execution_error(stage: str, item: str) -> None:
        execution_errors.pop(_execution_key(stage, item), None)

    def _entry_services_from_graph() -> set[str]:
        callers: set[str] = set()
        callees: set[str] = set()
        for svc, neighbors in graph.items():
            for info in neighbors:
                if info[1] == "caller_to_callee":
                    callers.add(svc)
                    callees.add(info[0])
        explicit_entries = {
            svc for svc in callers | callees if svc in {"frontend", "ts-ui-dashboard"}
        }
        return explicit_entries or (callers - callees)

    entry_services = _entry_services_from_graph()

    all_faults = [
        _fault_record(inj)
        for inj in injections
        if inj.get("target")
    ]

    def _rebuild_adjacency() -> tuple[dict[str, list[str]], dict[str, int]]:
        a: dict[str, list[str]] = {}
        d: dict[str, int] = {}
        for e in edges:
            a.setdefault(e["src"], []).append(e["dst"])
            d[e["dst"]] = d.get(e["dst"], 0) + 1
        return a, d

    def _unreachable_seed_nodes() -> list[str]:
        fpg_adj: dict[str, set[str]] = {}
        for e in edges:
            fpg_adj.setdefault(e["src"], set()).add(e["dst"])

        unreachable: list[str] = []
        for seed_svc in sorted(confirmed_seed_ids):
            if seed_svc not in nodes:
                unreachable.append(seed_svc)
                continue
            visited: set[str] = set()
            queue = [seed_svc]
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                for nxt in fpg_adj.get(cur, set()):
                    queue.append(nxt)
            if not (visited & entry_services):
                unreachable.append(seed_svc)
        return unreachable

    propagation_roots: list[str] = []

    async def _gate_result(
        *,
        task_kind: str,
        label: str,
        task: dict[str, Any],
        submitted_result: dict[str, Any],
        child_session: dict[str, Any] | None,
        attempt: int,
    ) -> GateDecision | None:
        gate_label = f"gate-{label}"
        gate_result: AgentResult = await ctx.agent(
            _structured_task_prompt(gate_label),
            scenario="verifier/gate",
            model=judge_model,
            schema=GateDecision,
            atom_config={
                "duckdb_sql": {"data_dir": data_dir},
                "gate_context": {
                    "task_kind": task_kind,
                    "task": task,
                    "submitted_result": submitted_result,
                    "child_session": child_session,
                    "attempt": attempt,
                },
            },
            retry=agent_retries,
            trace_label=gate_label,
        )
        return gate_result if isinstance(gate_result, GateDecision) else None

    async def _verify_seed(
        inj: Injection,
        judge_context: str = "",
    ) -> tuple[Injection, SeedResult | None]:
        target = _injection_node_id(inj) if _is_link_injection(inj) else inj["target"]
        fault_kind = inj.get("chaos_type", "unknown")
        task = {
            "seed": _injection_node_id(inj),
            "target": target,
            "fault_kind": fault_kind,
            "params": inj.get("params", ""),
            "subject": _injection_subject(inj),
        }
        feedback = judge_context
        last_result: SeedResult | None = None
        for attempt in range(gate_retries + 1):
            label = f"seed-{_injection_node_id(inj)}-a{attempt}"
            result: AgentResult = await ctx.agent(
                _agent_task_prompt(label),
                scenario="verifier/seed",
                atom_config={
                    "duckdb_sql": {"data_dir": data_dir},
                    "seed_context": {
                        "target": target,
                        "fault_kind": fault_kind,
                        "params": inj.get("params", ""),
                        "fault_doc": fault_docs.get(fault_kind, ""),
                        "judge_context": feedback,
                    },
                    "seed_finalize": {"data_dir": data_dir},
                },
                retry=agent_retries,
                trace_label=label,
            )
            if not isinstance(result, dict):
                reason = "seed agent returned no structured result"
                _record_execution_error("seed", _injection_node_id(inj), reason)
                return inj, {
                    "verdict": "inconclusive",
                    "_error": reason,
                    "rationale": reason,
                }
            if result.get("verdict") not in {
                "confirmed",
                "rejected",
                "inconclusive",
            }:
                reason = f"seed agent returned invalid verdict: {result.get('verdict')!r}"
                _record_execution_error("seed", _injection_node_id(inj), reason)
                return inj, {
                    "verdict": "inconclusive",
                    "_error": reason,
                    "rationale": reason,
                }
            last_result = cast(SeedResult, result)
            child_session = _find_child_session(ctx, label)
            gate = await _gate_result(
                task_kind="seed",
                label=label,
                task=task,
                submitted_result=dict(last_result),
                child_session=child_session,
                attempt=attempt,
            )
            if gate is None:
                reason = "seed gate returned no structured decision"
                _record_execution_error("seed", _injection_node_id(inj), reason)
                last_result["verdict"] = "inconclusive"
                last_result["_error"] = reason
                last_result["rationale"] = (
                    reason + ": " + str(last_result.get("rationale", ""))
                )
                return inj, last_result
            gate_payload = gate.model_dump(mode="json")
            last_result["gate"] = gate_payload
            gate_log.append({
                "task": "seed",
                "label": label,
                "seed": _injection_node_id(inj),
                "gate": gate_payload,
            })
            if gate.accepted:
                _clear_execution_error("seed", _injection_node_id(inj))
                return inj, last_result
            if attempt >= gate_retries:
                reason = "seed gate rejected all retry attempts"
                _record_execution_error("seed", _injection_node_id(inj), reason)
                last_result["verdict"] = "inconclusive"
                last_result["_error"] = reason
                last_result["rationale"] = (
                    reason + ": " + str(last_result.get("rationale", ""))
                )
                return inj, last_result
            feedback = _retry_context(
                base_context=judge_context,
                label=label,
                submitted_result=dict(last_result),
                gate=gate,
            )
        return inj, last_result

    def _accept_seed_node(inj: Injection, seed_verdict: SeedResult) -> str:
        root_id = _injection_node_id(inj)
        nodes[root_id] = _node_from_seed(inj, seed_verdict, window)
        if not _is_link_injection(inj):
            propagation_roots.append(root_id)
            return root_id

        effect_target = _seed_effect_target(inj, seed_verdict)
        if effect_target not in nodes:
            nodes[effect_target] = _node_from_link_effect(
                inj,
                seed_verdict,
                window,
            )
        node_fault[effect_target] = _fault_record(inj)
        if effect_target not in adj.get(root_id, []):
            edges.append(
                _edge_dict(
                    root_id,
                    effect_target,
                    "link_to_service",
                    rel_mechanism,
                    "link fault manifests on the rule-bearing service side",
                )
            )
            adj.setdefault(root_id, []).append(effect_target)
            in_deg[effect_target] = in_deg.get(effect_target, 0) + 1
        propagation_roots.append(effect_target)
        return root_id

    async def _verify_hop(
        from_svc: str,
        to_svc: str,
        rel_type: str,
        *,
        judge_context: str = "",
        prior_verdict: PriorVerdict | None = None,
    ) -> HopResult | None:
        hop_fault = node_fault.get(
            from_svc,
            [all_faults[0][0], all_faults[0][1]],
        )
        edge_fault_kind = hop_fault[0]
        edge_fault_docs: dict[str, str] = {}
        if edge_fault_kind in fault_docs:
            edge_fault_docs[edge_fault_kind] = fault_docs[edge_fault_kind]
        task = {
            "from_service": from_svc,
            "to_service": to_svc,
            "rel_type": rel_type,
            "fault_kind": edge_fault_kind,
            "is_infra": to_svc in infra_set,
            "prior_verdict": prior_verdict.model_dump(mode="json")
            if prior_verdict
            else {},
        }
        feedback = judge_context
        last_result: HopResult | None = None
        for attempt in range(gate_retries + 1):
            label = f"hop-{from_svc}-to-{to_svc}-a{attempt}"
            result: AgentResult = await ctx.agent(
                _agent_task_prompt(label),
                scenario="verifier/hop",
                atom_config={
                    "duckdb_sql": {"data_dir": data_dir},
                    "hop_context": {
                        "from_service": from_svc,
                        "to_service": to_svc,
                        "rel_type": rel_type,
                        "fault_kind": edge_fault_kind,
                        "all_faults": [
                            (f[0], f[1], f[2] if len(f) > 2 else "")
                            for f in all_faults
                            if len(f) >= 2
                        ],
                        "fault_docs": edge_fault_docs,
                        "is_infra": to_svc in infra_set,
                        "upstream_evidence": nodes.get(from_svc),
                        "judge_context": feedback,
                        "prior_verdict": prior_verdict.model_dump(mode="json")
                        if prior_verdict
                        else None,
                    },
                    "hop_finalize": {"data_dir": data_dir},
                },
                retry=agent_retries,
                trace_label=label,
            )
            if not isinstance(result, dict):
                reason = "hop agent returned no structured result"
                _record_execution_error("hop", from_svc + "__" + to_svc, reason)
                return {
                    "verdict": "inconclusive",
                    "_error": reason,
                    "rationale": reason,
                }
            if result.get("verdict") not in {
                "confirmed",
                "rejected",
                "inconclusive",
            }:
                reason = f"hop agent returned invalid verdict: {result.get('verdict')!r}"
                _record_execution_error("hop", from_svc + "__" + to_svc, reason)
                return {
                    "verdict": "inconclusive",
                    "_error": reason,
                    "rationale": reason,
                }
            last_result = cast(HopResult, result)
            child_session = _find_child_session(ctx, label)
            gate = await _gate_result(
                task_kind="hop",
                label=label,
                task=task,
                submitted_result=dict(last_result),
                child_session=child_session,
                attempt=attempt,
            )
            if gate is None:
                reason = "hop gate returned no structured decision"
                _record_execution_error("hop", from_svc + "__" + to_svc, reason)
                last_result["_error"] = reason
                return last_result
            gate_payload = gate.model_dump(mode="json")
            last_result["gate"] = gate_payload
            gate_log.append({
                "task": "hop",
                "label": label,
                "from": from_svc,
                "to": to_svc,
                "gate": gate_payload,
            })
            if gate.accepted:
                _clear_execution_error("hop", from_svc + "__" + to_svc)
                return last_result
            if attempt >= gate_retries:
                reason = "hop gate rejected all retry attempts"
                ctx.log(f"⚠ hop {from_svc}__{to_svc}: {reason}")
                last_result["verdict"] = "inconclusive"
                last_result["rationale"] = (
                    reason + ": " + str(last_result.get("rationale", ""))
                )
                return last_result
            feedback = _retry_context(
                base_context=judge_context,
                label=label,
                submitted_result=dict(last_result),
                gate=gate,
            )
        return last_result

    def _accept_hop_result(
        from_svc: str,
        to_svc: str,
        rel_type: str,
        result: HopResult,
        *,
        claim_override: str = "",
    ) -> bool:
        if _reaches(adj, to_svc, from_svc):
            hop_log.append(
                {
                    "round": round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": "dropped_cycle",
                }
            )
            return False
        changed = False
        if to_svc not in nodes:
            nodes[to_svc] = _node_from_verdict(to_svc, result, window)
            node_fault[to_svc] = node_fault.get(
                from_svc,
                [all_faults[0][0], all_faults[0][1]],
            )
            changed = True
        if to_svc not in adj.get(from_svc, []):
            edges.append(
                _edge_dict(
                    from_svc,
                    to_svc,
                    rel_type,
                    rel_mechanism,
                    claim_override or str(result.get("claim", "")),
                )
            )
            adj.setdefault(from_svc, []).append(to_svc)
            in_deg[to_svc] = in_deg.get(to_svc, 0) + 1
            changed = True
        return changed

    checked_edges: set[str] = set()

    async def _propagate_from_roots(roots: Sequence[str]) -> bool:
        nonlocal round_n
        queue = [
            root for root in dict.fromkeys(roots)
            if root in nodes and root not in infra_set and root not in entry_services
        ]
        changed_any = False

        while queue:
            round_n += 1
            batch = list(queue)
            queue = []

            pending_hops: list[list[str]] = []
            for current in batch:
                for neighbor_info in graph.get(current, []):
                    neighbor = neighbor_info[0]
                    rel_type = neighbor_info[1]
                    edge_key = current + "__" + neighbor
                    if edge_key in checked_edges:
                        continue
                    checked_edges.add(edge_key)

                    # fpg DAG rule: never evaluate an edge that would
                    # close a cycle through already-accepted edges.
                    if neighbor in nodes and _reaches(adj, neighbor, current):
                        hop_log.append(
                            {
                                "round": round_n,
                                "from": current,
                                "to": neighbor,
                                "verdict": "skipped_cycle",
                            }
                        )
                        continue
                    pending_hops.append([current, neighbor, rel_type])

            if not pending_hops:
                continue

            ctx.log("Round " + str(round_n) + ": " + str(len(pending_hops)) + " hops")

            async def _make_hop_coro(
                from_svc: str,
                to_svc: str,
                rel_type: str,
            ) -> HopResult | None:
                return await _verify_hop(from_svc, to_svc, rel_type)

            coros: list[Awaitable[HopResult | None]] = [
                _make_hop_coro(item[0], item[1], item[2]) for item in pending_hops
            ]
            results = _parallel_items(await ctx.parallel(coros))

            for idx in range(len(pending_hops)):
                from_svc, to_svc, rel_type = pending_hops[idx]
                result = results[idx] if idx < len(results) else None
                edge_key = from_svc + "__" + to_svc
                if result is None:
                    _record_execution_error(
                        "hop",
                        edge_key,
                        "hop verifier task failed before returning a result",
                    )
                verdict = result.get("verdict") if isinstance(result, dict) else None
                hop_log.append(
                    {
                        "round": round_n,
                        "from": from_svc,
                        "to": to_svc,
                        "verdict": verdict if verdict else "no-result",
                    }
                )
                ctx.log(
                    "  "
                    + from_svc
                    + " -> "
                    + to_svc
                    + ": "
                    + (verdict if verdict else "no-result")
                )
                hop_result = cast(HopResult, result) if isinstance(result, dict) else None
                if hop_result and verdict:
                    verdicts[edge_key] = hop_result

                if verdict != "confirmed":
                    continue
                assert hop_result is not None
                was_new_node = to_svc not in nodes
                accepted = _accept_hop_result(
                    from_svc,
                    to_svc,
                    rel_type,
                    hop_result,
                )
                changed_any = accepted or changed_any
                if accepted and was_new_node:
                    if to_svc not in infra_set and to_svc not in entry_services:
                        queue.append(to_svc)

            if not queue:
                ctx.log("propagation frontier exhausted for this round")

        return changed_any

    if skip_propagate:
        existing = cast(dict[str, Any], args.get("existing_state", {}))
        nodes = {n["id"]: n for n in existing.get("nodes", [])}
        edges = list(existing.get("edges", []))
        verdicts = cast(dict[str, HopResult], existing.get("verdicts", {}))
        hop_log = cast(list[HopLogEntry], existing.get("hop_log", []))
        round_n = int(existing.get("rounds", 0))
        seed_verdicts = cast(dict[str, SeedResult], existing.get("seed_verdicts", {}))
        confirmed_seed_ids = set(existing.get("confirmed_seeds", []))
        for item in cast(list[ExecutionError], existing.get("execution_errors", [])):
            execution_errors[
                _execution_key(item["stage"], item["item"])
            ] = item
        if not confirmed_seed_ids and not seed_verdicts:
            confirmed_seed_ids = {seed for seed in seeds if seed in nodes}
        adj, in_deg = _rebuild_adjacency()
        node_fault = {
            _injection_node_id(inj): _fault_record(inj)
            for inj in injections
            if inj.get("target")
        }
        for inj in injections:
            if inj.get("target") and _is_link_injection(inj):
                node_fault[_injection_effect_target(inj)] = _fault_record(inj)
    else:
        # -- Phase 0: verify seeds ----------------------------------------
        ctx.phase("seed")
        node_fault = {
            _injection_node_id(inj): _fault_record(inj)
            for inj in injections
            if inj.get("target")
        }
        nodes = {}
        edges = []
        adj = {}
        in_deg = {}
        verdicts = {}
        hop_log = []
        round_n = 0
        seed_verdicts = {}
        confirmed_seed_ids = set()

        seed_injections = [inj for inj in injections if inj.get("target")]
        seed_coros: list[Awaitable[tuple[Injection, SeedResult | None]]] = [
            _verify_seed(inj) for inj in seed_injections
        ]
        seed_result_items = _parallel_items(await ctx.parallel(seed_coros))
        for idx, item in enumerate(seed_result_items):
            if _is_seed_parallel_item(item):
                continue
            if idx < len(seed_injections):
                _record_execution_error(
                    "seed",
                    _injection_node_id(seed_injections[idx]),
                    "seed verifier task failed before returning a result",
                )
        for inj in seed_injections[len(seed_result_items):]:
            _record_execution_error(
                "seed",
                _injection_node_id(inj),
                "seed verifier task was missing from parallel results",
            )
        seed_results = [
            item for item in seed_result_items
            if _is_seed_parallel_item(item)
        ]
        missing_seed_results = len(seed_coros) - len(seed_results)
        if missing_seed_results > 0:
            ctx.log(
                f"{missing_seed_results} seed verifier task(s) returned no usable result"
            )
        for inj, seed_verdict in seed_results:
            root_id = _injection_node_id(inj)
            if seed_verdict and seed_verdict.get("verdict") == "confirmed":
                root_id = _accept_seed_node(inj, seed_verdict)
                confirmed_seed_ids.add(root_id)
                _clear_execution_error("seed", root_id)
                ctx.log(
                    f"seed {root_id}: confirmed ({seed_verdict.get('predicate')})"
                )
            elif seed_verdict and seed_verdict.get("verdict") == "inconclusive":
                ctx.log(f"seed {root_id}: inconclusive — keeping for audit review")
            else:
                v = seed_verdict.get("verdict", "no result") if seed_verdict else "no result"
                ctx.log(f"seed {root_id}: {v} — skipping")

        for inj, seed_verdict in seed_results:
            if seed_verdict:
                seed_verdicts[_injection_node_id(inj)] = seed_verdict

        if not nodes:
            ctx.log("no seeds confirmed after seed map; audit may request rechecks")

        # -- Phase 1: propagate -------------------------------------------
        ctx.phase("propagate")
        await _propagate_from_roots(propagation_roots or list(nodes))

    def _case_summary() -> dict[str, Any]:
        return {
            "injections": injections,
            "entry_services": sorted(entry_services),
            "all_faults": all_faults,
        }

    def _graph_snapshot() -> dict[str, Any]:
        return {
            "nodes": [nodes[k] for k in sorted(nodes)],
            "edges": list(edges),
            "entry_services": sorted(entry_services),
        }

    def _ledger_snapshot() -> dict[str, Any]:
        return {
            "seed_verdicts": seed_verdicts,
            "hop_verdicts": verdicts,
            "hop_log": hop_log,
            "gate_log": gate_log,
            "confirmed_seeds": sorted(confirmed_seed_ids),
        }

    def _paths_from(seed: str, max_depth: int = 10) -> list[list[str]]:
        if seed not in nodes:
            return []
        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(seed, [seed])]
        while stack:
            cur, path = stack.pop()
            if cur in entry_services and len(path) > 1:
                paths.append(path)
                continue
            if len(path) >= max_depth:
                continue
            for nxt in adj.get(cur, []):
                if nxt in path:
                    continue
                stack.append((nxt, path + [nxt]))
        return paths

    def _candidate_paths() -> list[tuple[str, str, list[str]]]:
        out: list[tuple[str, str, list[str]]] = []
        for seed in sorted(seeds):
            for idx, path in enumerate(_paths_from(seed)):
                out.append((f"{seed}:path:{idx}", seed, path))
        return out

    def _rebuild_graph_indices() -> None:
        nonlocal adj, in_deg
        adj, in_deg = _rebuild_adjacency()

    def _prune_unreachable_nodes() -> None:
        reachable: set[str] = set()
        stack = [seed for seed in confirmed_seed_ids if seed in nodes]
        while stack:
            cur = stack.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            stack.extend(adj.get(cur, []))
        for node_id in list(nodes):
            if node_id not in reachable:
                nodes.pop(node_id, None)
        edges[:] = [
            e for e in edges
            if e.get("src") in nodes and e.get("dst") in nodes
        ]
        _rebuild_graph_indices()

    def _path_contains_edge(path: Sequence[str], src: str, dst: str) -> bool:
        return any(
            path[idx] == src and path[idx + 1] == dst
            for idx in range(len(path) - 1)
        )

    def _edge_used_by_other_seed(src: str, dst: str, seed: str | None) -> bool:
        if not seed:
            return False
        for other_seed in confirmed_seed_ids:
            if other_seed == seed:
                continue
            if any(_path_contains_edge(path, src, dst) for path in _paths_from(other_seed)):
                return True
        return False

    def _drop_matches_reported_lineage(drop: EdgeDrop) -> bool:
        if drop.path_id:
            return any(
                path_id == drop.path_id
                and _path_contains_edge(path, drop.src, drop.dst)
                for path_id, _, path in _candidate_paths()
            )
        if drop.seed:
            return any(
                _path_contains_edge(path, drop.src, drop.dst)
                for path in _paths_from(drop.seed)
            )
        return True

    def _drop_audit_edges(drop_edges: Sequence[EdgeDrop]) -> bool:
        if not drop_edges:
            return False
        doomed: set[tuple[str, str]] = set()
        for item in drop_edges:
            if not item.src or not item.dst:
                continue
            if not _drop_matches_reported_lineage(item):
                ctx.log(
                    f"  audit drop skipped for {item.src} -> {item.dst}: "
                    "edge is not on the reported seed/path lineage"
                )
                continue
            if _edge_used_by_other_seed(item.src, item.dst, item.seed):
                ctx.log(
                    f"  audit drop skipped for {item.src} -> {item.dst}: "
                    "edge is still used by another seed-to-entry path"
                )
                continue
            doomed.add((item.src, item.dst))
        if not doomed:
            return False
        before = len(edges)
        edges[:] = [
            e for e in edges
            if (str(e.get("src", "")), str(e.get("dst", ""))) not in doomed
        ]
        changed = len(edges) != before
        if changed:
            _rebuild_graph_indices()
            _prune_unreachable_nodes()
        return changed

    # -- Audit map/reduce + rework loop -----------------------------------
    async def _audit_agent(
        *,
        label: str,
        role: str,
        instruction: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> BaseModel | None:
        result: AgentResult = await ctx.agent(
            _structured_task_prompt(label),
            scenario="verifier/audit",
            model=judge_model,
            schema=schema,
            atom_config={
                "duckdb_sql": {"data_dir": data_dir},
                "audit_context": {
                    "role": role,
                    "instruction": instruction,
                    "payload": payload,
                },
            },
            retry=agent_retries,
            trace_label=label,
        )
        return result if isinstance(result, BaseModel) else None

    async def _apply_audit_rework(
        requests: Sequence[ReworkRequest],
    ) -> tuple[bool, list[dict[str, Any]]]:
        if not requests:
            return False, []
        rework_log: list[dict[str, Any]] = []
        changed = False
        inj_by_seed = {
            _injection_node_id(inj): inj
            for inj in injections
            if inj.get("target")
        }

        seed_requests = [
            req for req in requests
            if isinstance(req, SeedRecheckRequest) and req.seed in inj_by_seed
        ]
        new_roots: list[str] = []
        if seed_requests:
            ctx.phase("audit-seed-rework")
            ctx.log(f"Audit requested {len(seed_requests)} seed rechecks")
            seed_result_items = _parallel_items(await ctx.parallel([
                _verify_seed(inj_by_seed[req.seed], req.context)
                for req in seed_requests
            ]))
            for idx, item in enumerate(seed_result_items):
                if _is_seed_parallel_item(item):
                    continue
                if idx < len(seed_requests):
                    _record_execution_error(
                        "seed",
                        seed_requests[idx].seed,
                        "audit seed recheck failed before returning a result",
                    )
            for req in seed_requests[len(seed_result_items):]:
                _record_execution_error(
                    "seed",
                    req.seed,
                    "audit seed recheck was missing from parallel results",
                )
            seed_results = [
                item for item in seed_result_items
                if _is_seed_parallel_item(item)
            ]
            missing_seed_results = len(seed_requests) - len(seed_results)
            if missing_seed_results > 0:
                ctx.log(
                    f"{missing_seed_results} audit seed recheck task(s) "
                    "returned no usable result"
                )
            for inj, seed_verdict in seed_results:
                seed_id = _injection_node_id(inj)
                previous_seed_verdict = seed_verdicts.get(seed_id)
                if seed_verdict:
                    seed_verdicts[seed_id] = seed_verdict
                    if seed_verdict != previous_seed_verdict:
                        changed = True
                verdict = seed_verdict.get("verdict") if seed_verdict else "no-result"
                rework_log.append({
                    "kind": "seed_recheck",
                    "seed": seed_id,
                    "verdict": verdict,
                    "rationale": seed_verdict.get("rationale", "")
                    if seed_verdict
                    else "",
                })
                ctx.log(f"  seed recheck {seed_id}: {verdict}")
                if seed_verdict and verdict == "confirmed":
                    before_roots = len(propagation_roots)
                    accepted_seed = _accept_seed_node(inj, seed_verdict)
                    new_roots.extend(propagation_roots[before_roots:])
                    confirmed_seed_ids.add(accepted_seed)
                    _clear_execution_error("seed", accepted_seed)
                    changed = True

        hop_requests = [
            req for req in requests
            if isinstance(req, HopRecheckRequest)
            and req.from_service in nodes
        ]
        if hop_requests:
            ctx.phase("audit-hop-rework")
            ctx.log(f"Audit requested {len(hop_requests)} hop rechecks")

            async def _one_hop_rework(
                req: HopRecheckRequest,
            ) -> tuple[HopRecheckRequest, HopResult | None]:
                rel_type = next(
                    (
                        info[1]
                        for info in graph.get(req.from_service, [])
                        if info[0] == req.to_service
                    ),
                    "callee_to_caller",
                )
                prior = dict(verdicts.get(
                    req.from_service + "__" + req.to_service,
                    {},
                ))
                return req, await _verify_hop(
                    req.from_service,
                    req.to_service,
                    rel_type,
                    judge_context=req.context,
                    prior_verdict=PriorVerdict(
                        verdict=str(prior.get("verdict", "")),
                        rationale=str(prior.get("rationale", "")),
                    ),
                )

            hop_result_items = _parallel_items(await ctx.parallel([
                _one_hop_rework(req) for req in hop_requests
            ]))
            for idx, item in enumerate(hop_result_items):
                if _is_hop_rework_parallel_item(item):
                    continue
                if idx < len(hop_requests):
                    hop_req = hop_requests[idx]
                    _record_execution_error(
                        "hop",
                        hop_req.from_service + "__" + hop_req.to_service,
                        "audit hop recheck failed before returning a result",
                    )
            for hop_req in hop_requests[len(hop_result_items):]:
                _record_execution_error(
                    "hop",
                    hop_req.from_service + "__" + hop_req.to_service,
                    "audit hop recheck was missing from parallel results",
                )
            hop_results = [
                item for item in hop_result_items
                if _is_hop_rework_parallel_item(item)
            ]
            missing_hop_results = len(hop_requests) - len(hop_results)
            if missing_hop_results > 0:
                ctx.log(
                    f"{missing_hop_results} audit hop recheck task(s) "
                    "returned no usable result"
                )
            for hop_req, result in hop_results:
                from_svc = hop_req.from_service
                to_svc = hop_req.to_service
                hop_verdict: str | None = (
                    result.get("verdict") if isinstance(result, dict) else None
                )
                edge_key = from_svc + "__" + to_svc
                hop_log.append({
                    "round": round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": (hop_verdict or "no-result") + "(audit-rework)",
                })
                if isinstance(result, dict) and hop_verdict:
                    previous_hop_verdict = verdicts.get(edge_key)
                    verdicts[edge_key] = result
                    if result != previous_hop_verdict:
                        changed = True
                rework_log.append({
                    "kind": "hop_recheck",
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": hop_verdict or "no-result",
                    "rationale": result.get("rationale", "")
                    if isinstance(result, dict)
                    else "",
                })
                ctx.log(
                    f"  hop recheck {from_svc} -> {to_svc}: "
                    f"{hop_verdict or 'no-result'}"
                )
                if hop_verdict != "confirmed" or not isinstance(result, dict):
                    continue
                was_new_node = to_svc not in nodes
                rel_type = next(
                    (
                        info[1]
                        for info in graph.get(from_svc, [])
                        if info[0] == to_svc
                    ),
                    "callee_to_caller",
                )
                changed = _accept_hop_result(
                    from_svc,
                    to_svc,
                    rel_type,
                    result,
                    claim_override=hop_req.context[:200],
                ) or changed
                if was_new_node and to_svc not in infra_set and to_svc not in entry_services:
                    new_roots.append(to_svc)
        if new_roots:
            ctx.phase("audit-propagate-rework")
            ctx.log(f"Propagating from {len(set(new_roots))} audit-confirmed roots")
            changed = await _propagate_from_roots(new_roots) or changed
        return changed, rework_log

    def _close_audit_when_unconfirmed_seeds_are_resolved() -> None:
        nonlocal audit_result
        if audit_result is None or audit_result.accepted:
            return
        if audit_result.unexplained_anomalies:
            return

        unresolved_status = {"needs_recheck", "invalid_path"}
        unconfirmed = set(seeds) - confirmed_seed_ids
        unresolved = {
            seed
            for seed, coverage in audit_result.seed_coverage.items()
            if coverage in unresolved_status
        }
        if any(seed in confirmed_seed_ids for seed in unresolved):
            return

        for seed in confirmed_seed_ids:
            coverage = audit_result.seed_coverage.get(seed)
            if coverage in unresolved_status or coverage is None:
                return

        removable_invalid_paths = {
            path_id
            for path_id in audit_result.invalid_causal_paths
            if path_id.split(":path:", 1)[0] in unconfirmed
        }
        blocking_invalid_paths = [
            path_id
            for path_id in audit_result.invalid_causal_paths
            if path_id not in removable_invalid_paths
        ]
        if blocking_invalid_paths:
            return

        blocking_drops = [
            edge
            for edge in audit_result.drop_edges
            if edge.seed not in unconfirmed
        ]
        if blocking_drops:
            return

        blocking_rework = [
            req
            for req in audit_result.rework_requests
            if not (
                isinstance(req, SeedRecheckRequest)
                and req.seed in unconfirmed
            )
        ]
        if blocking_rework:
            return

        coverage_map = dict(audit_result.seed_coverage)
        for seed in unconfirmed:
            if coverage_map.get(seed) in {None, "needs_recheck", "invalid_path"}:
                coverage_map[seed] = "benign_or_no_effect"

        audit_result = audit_result.model_copy(
            update={
                "accepted": True,
                "seed_coverage": coverage_map,
                "invalid_causal_paths": blocking_invalid_paths,
                "drop_edges": blocking_drops,
                "rework_requests": [],
                "stop_reason": "closed_unconfirmed_seed_no_entry_effect",
                "rationale": (
                    audit_result.rationale
                    + "\n\nHarness closure: all meaningful entry anomalies are "
                    "explained by confirmed seed paths. Remaining rework applies "
                    "only to unconfirmed seeds whose candidate paths were invalid "
                    "or repeatedly failed gate review, so they are resolved as "
                    "benign_or_no_effect instead of blocking the accepted graph."
                ),
            }
        )

    if skip_judge:
        ctx.log("audit skipped by request")
    else:
        for audit_round in range(max_audit_rounds):
            ctx.phase("audit" if audit_round == 0 else f"audit-r{audit_round + 1}")
            graph_view = _graph_snapshot()
            ledger_view = _ledger_snapshot()
            case_view = _case_summary()

            anomaly_scopes = sorted(entry_services) or ["<entry-services-not-discovered>"]
            anomaly_results = _parallel_items(await ctx.parallel([
                _audit_agent(
                    label=f"audit-anomaly-{scope}-r{audit_round}",
                    role="anomaly_coverage",
                    instruction=(
                        "Inspect this entry/service scope. Identify meaningful "
                        "abnormal trace, metric, or log symptoms and decide "
                        "whether the current candidate graph explains each one."
                    ),
                    payload={
                        "scope": scope,
                        "case": case_view,
                        "candidate_graph": graph_view,
                        "evidence_ledger": ledger_view,
                    },
                    schema=AnomalyAuditReport,
                )
                for scope in anomaly_scopes
            ]))
            for idx, report in enumerate(anomaly_results):
                if report is not None:
                    continue
                if idx < len(anomaly_scopes):
                    _record_execution_error(
                        "audit",
                        f"anomaly:{anomaly_scopes[idx]}:r{audit_round}",
                        "anomaly audit task failed before returning a report",
                    )
            for scope in anomaly_scopes[len(anomaly_results):]:
                _record_execution_error(
                    "audit",
                    f"anomaly:{scope}:r{audit_round}",
                    "anomaly audit task was missing from parallel results",
                )
            anomaly_reports = [
                _dump_model(report) for report in anomaly_results if report is not None
            ]

            path_items = _candidate_paths()
            causal_results = _parallel_items(await ctx.parallel([
                _audit_agent(
                    label=f"audit-causal-{path_id}-r{audit_round}",
                    role="causal_path",
                    instruction=(
                        "Audit whether this single seed-to-entry path is a "
                        "coherent causal explanation. Reject paths that only "
                        "borrow another seed's symptoms or are merely "
                        "topologically reachable."
                    ),
                    payload={
                        "path_id": path_id,
                        "seed": seed,
                        "path": path,
                        "case": case_view,
                        "candidate_graph": graph_view,
                        "evidence_ledger": ledger_view,
                    },
                    schema=CausalAuditReport,
                )
                for path_id, seed, path in path_items
            ]))
            for idx, report in enumerate(causal_results):
                if report is not None:
                    continue
                if idx < len(path_items):
                    path_id, _, _ = path_items[idx]
                    _record_execution_error(
                        "audit",
                        f"causal:{path_id}:r{audit_round}",
                        "causal audit task failed before returning a report",
                    )
            for path_id, _, _ in path_items[len(causal_results):]:
                _record_execution_error(
                    "audit",
                    f"causal:{path_id}:r{audit_round}",
                    "causal audit task was missing from parallel results",
                )
            causal_reports = [
                _dump_model(report) for report in causal_results if report is not None
            ]

            audit_seeds = sorted(seeds)
            seed_audit_results = _parallel_items(await ctx.parallel([
                _audit_agent(
                    label=f"audit-seed-{seed}-r{audit_round}",
                    role="seed_coverage",
                    instruction=(
                        "Classify this seed as explains_entry, local_only, "
                        "benign_or_no_effect, needs_recheck, or invalid_path."
                    ),
                    payload={
                        "seed": seed,
                        "case": case_view,
                        "candidate_graph": graph_view,
                        "evidence_ledger": ledger_view,
                        "paths_from_seed": _paths_from(seed),
                        "causal_reports": [
                            report for report in causal_reports
                            if str(report.get("path_id", "")).startswith(seed + ":")
                        ],
                        "anomaly_reports": anomaly_reports,
                    },
                    schema=SeedCoverageReport,
                )
                for seed in audit_seeds
            ]))
            for idx, report in enumerate(seed_audit_results):
                if report is not None:
                    continue
                if idx < len(audit_seeds):
                    _record_execution_error(
                        "audit",
                        f"seed:{audit_seeds[idx]}:r{audit_round}",
                        "seed coverage audit task failed before returning a report",
                    )
            for seed in audit_seeds[len(seed_audit_results):]:
                _record_execution_error(
                    "audit",
                    f"seed:{seed}:r{audit_round}",
                    "seed coverage audit task was missing from parallel results",
                )
            seed_coverage_reports = [
                _dump_model(report)
                for report in seed_audit_results
                if report is not None
            ]

            reducer = await _audit_agent(
                label=f"audit-reducer-r{audit_round}",
                role="audit_reducer",
                instruction=(
                    "Merge the audit reports. Accept only when all meaningful "
                    "anomalies are explained or resolved and every seed has a "
                    "resolved coverage status. Otherwise emit concrete rework "
                    "requests and/or path-specific edge drops."
                ),
                payload={
                    "round": audit_round,
                    "case": case_view,
                    "candidate_graph": graph_view,
                    "evidence_ledger": ledger_view,
                    "anomaly_reports": anomaly_reports,
                    "causal_reports": causal_reports,
                    "seed_coverage_reports": seed_coverage_reports,
                },
                schema=GlobalAudit,
            )
            audit_result = reducer if isinstance(reducer, GlobalAudit) else None
            if audit_result is None:
                _record_execution_error(
                    "audit",
                    f"reducer:r{audit_round}",
                    "audit reducer returned no structured decision",
                )
            audit_payload = (
                audit_result.model_dump(mode="json") if audit_result else {}
            )
            ctx.log(
                "  audit reducer: "
                + ("accepted" if audit_result and audit_result.accepted else "needs rework")
            )

            dropped = _drop_audit_edges(audit_result.drop_edges if audit_result else [])
            is_last_audit_round = audit_round >= max_audit_rounds - 1
            if (
                is_last_audit_round
                and audit_result is not None
                and not audit_result.accepted
            ):
                ctx.log("audit reached max rounds; skipping further rework")
                rework_changed = False
                rework_log: list[dict[str, Any]] = []
            else:
                rework_changed, rework_log = await _apply_audit_rework(
                    audit_result.rework_requests if audit_result else []
                )
            audit_rounds_log.append({
                "round": audit_round + 1,
                "anomaly_reports": anomaly_reports,
                "causal_reports": causal_reports,
                "seed_coverage_reports": seed_coverage_reports,
                "audit": audit_payload,
                "dropped_edges": [
                    item.model_dump(mode="json")
                    for item in (audit_result.drop_edges if audit_result else [])
                ],
                "rework_results": rework_log,
            })

            if audit_result and audit_result.accepted:
                break
            if not dropped and not rework_changed:
                ctx.log("audit has no effective rework left; stopping audit loop")
                break

        _close_audit_when_unconfirmed_seeds_are_resolved()

    # fpg rule: in-degree >= 2 requires combine; each confirmed edge is
    # an independently sufficient path, so the combination is OR.
    for svc, node in nodes.items():
        if in_deg.get(svc, 0) >= 2:
            node["combine"] = "OR"
        else:
            node.pop("combine", None)

    def _audit_seed_coverage(seed: str) -> SeedCoverageStatus | None:
        if audit_result is None:
            return None
        return audit_result.seed_coverage.get(seed)

    # -- Validation: unresolved seeds should either reach entry or be audit-resolved.
    ctx.phase("validate")
    reachability_warnings = _unreachable_seed_nodes()
    resolved_non_entry = {"local_only", "benign_or_no_effect"}
    unreachable = [
        seed
        for seed in reachability_warnings
        if _audit_seed_coverage(seed) not in resolved_non_entry
    ]
    for seed_svc in sorted(seeds):
        if seed_svc not in confirmed_seed_ids:
            ctx.log(f"⚠ seed {seed_svc}: not confirmed")
            continue
        if seed_svc in reachability_warnings:
            coverage = _audit_seed_coverage(seed_svc)
            if coverage in resolved_non_entry:
                ctx.log(
                    f"ⓘ seed {seed_svc}: no entry path, resolved by audit as "
                    f"{coverage}"
                )
                continue
            ctx.log(
                f"⚠ seed {seed_svc}: no path to entry services "
                f"{sorted(entry_services)} in fpg"
            )

    if not unreachable:
        ctx.log("✓ no unresolved confirmed seeds lack entry-service coverage")

    result_out: PropagationResult = {
        "nodes": [nodes[k] for k in sorted(nodes)],
        "edges": edges,
        "verdicts": verdicts,
        "hop_log": hop_log,
        "rounds": round_n,
        "seed_verdicts": seed_verdicts,
        "confirmed_seeds": sorted(confirmed_seed_ids),
        "gate_log": gate_log,
    }
    if audit_result:
        result_out["audit"] = audit_result.model_dump(mode="json")
    if audit_rounds_log:
        result_out["audit_rounds"] = audit_rounds_log
    if execution_errors:
        result_out["execution_errors"] = [
            execution_errors[key] for key in sorted(execution_errors)
        ]
    if reachability_warnings:
        result_out["reachability_warnings"] = reachability_warnings
    if unreachable:
        result_out["unreachable_seeds"] = unreachable
    return result_out
