"""``rca_evidence_tools`` — LLM-facing tool surface for the rca_hfsm scenario.

Registers five tools that mutate L1 state (design §3.3 + §6):
``record_symptom`` / ``record_observation`` / ``propose_hypothesis`` /
``attach_check`` are direct routes for the common operators;
``propose_update(op, ...)`` is the escape hatch for confirm / refute /
refine / split / merge / supersede / suspend.

Every tool routes through ``api.get_service('rca.gate').apply(...)`` and
renders the ``UpdateResult`` as ``ToolResult.text`` (``status=applied|
downgraded|rejected`` plus reason on the latter two paths). All tools
register with ``metadata['idempotent']=False`` — they mutate L1 so
``rca_observation_cache`` must not memoise them.

LLM-vocabulary guidance lives in ``tool_schemas.PARAMS``; the
``attach_check.verdict_proposal`` description nudges the model toward the
§7.1 / §7.2 sentinels (``triggered`` / ``supports`` / ``steelman``) the
gate matches by word-boundary regex. Per CLAUDE.md "no preset enums",
this is schema guidance not enforcement.

contract: single MANIFEST + single ``install``; imports stdlib +
``agentm.core.abi.*`` + ``agentm.extensions`` + the scenario's pure
``schema`` / ``updates`` / ``tool_schemas`` modules; the gate is reached
strictly via ``api.get_service('rca.gate')``.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

from rca.hfsm.schema import (
    CheckResult,
    Hypothesis,
    Interpretation,
    Observation,
    Prediction,
    Symptom,
)
from rca.hfsm.tool_schemas import DESCRIPTIONS, PARAMS
from rca.hfsm.updates import UpdateProposal, UpdateResult

MANIFEST = ExtensionManifest(
    name="rca_evidence_tools",
    description=(
        "LLM-facing evidence tools (record_symptom / record_observation / "
        "propose_hypothesis / attach_check / propose_update) routing every "
        "graph mutation through the rca.gate service."
    ),
    registers=(
        "tool:record_symptom",
        "tool:record_observation",
        "tool:propose_hypothesis",
        "tool:attach_check",
        "tool:propose_update",
    ),
    config_schema={"type": "object", "properties": {}, "additionalProperties": False},
    requires=("rca_falsification_gate",),
)


# ``_tool_signature`` is the canonical hash. ``rca_observation_cache``
# re-implements it (forbids atom-to-atom imports); keep both copies in sync.
def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _tool_signature(tool_name: str, args: dict[str, Any]) -> str:
    """``sha256(tool_name + canonical_json(args))``."""

    return hashlib.sha256(
        (tool_name + ":" + _canonical_json(args)).encode("utf-8")
    ).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _render(result: UpdateResult) -> str:
    """Render a gate ``UpdateResult`` as ``status=... [id=...] [to=...] [reason=...]``."""

    if result.kind == "applied":
        return f"status=applied id={result.applied_id}"
    if result.kind == "downgraded":
        downgrade_op = result.downgrade.op if result.downgrade else "?"
        applied = f" id={result.applied_id}" if result.applied_id else ""
        return f"status=downgraded to={downgrade_op}{applied} reason={result.reason}"
    return f"status=rejected reason={result.reason}"


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _observation_from(payload: dict[str, Any]) -> Observation:
    text = str(payload.get("text", ""))
    source_call = str(payload.get("source_tool_call", ""))
    signature = _tool_signature(source_call, {"text": text}) if source_call else ""
    return Observation(
        id=_new_id("O"),
        text=text,
        source_tool_call=source_call,
        tool_signature=signature,
        related_symptoms=list(payload.get("related_symptoms", []) or []),
        related_predictions=list(payload.get("related_predictions", []) or []),
        ts=time.time(),
    )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    gate = api.get_service("rca.gate")
    if gate is None:
        raise RuntimeError(
            "rca_evidence_tools: rca.gate missing; install rca_falsification_gate first"
        )

    async def _record_symptom(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return _error("status=rejected reason=text must be non-empty")
        sym = Symptom(
            id=_new_id("S"),
            text=text,
            source=str(args.get("source", "user_intake")),
            ts=time.time(),
        )
        return _ok(
            _render(gate.apply(UpdateProposal(op="record_symptom", symptom=sym)))
        )

    async def _record_observation(args: dict[str, Any]) -> ToolResult:
        if not str(args.get("text", "")).strip():
            return _error("status=rejected reason=text must be non-empty")
        obs = _observation_from(args)
        return _ok(
            _render(
                gate.apply(UpdateProposal(op="record_observation", observation=obs))
            )
        )

    async def _propose_hypothesis(args: dict[str, Any]) -> ToolResult:
        claim = str(args.get("claim", "")).strip()
        if not claim:
            return _error("status=rejected reason=claim must be non-empty")
        h_id = _new_id("H")
        predictions: list[Prediction] = []
        for idx, p in enumerate(list(args.get("predictions", []) or [])):
            if not isinstance(p, dict) or p.get("polarity") not in (
                "positive",
                "negative",
            ):
                return _error(
                    f"status=rejected reason=predictions[{idx}] must be object with "
                    "polarity='positive'|'negative'"
                )
            predictions.append(
                Prediction(
                    id=_new_id("P"),
                    hypothesis_id=h_id,
                    claim=str(p.get("claim", "")),
                    polarity=p["polarity"],
                    test_plan=p.get("test_plan"),
                )
            )
        h = Hypothesis(
            id=h_id,
            claim=claim,
            predictions=predictions,
            rationale=str(args.get("rationale", "")),
        )
        return _ok(_render(gate.apply(UpdateProposal(op="propose", hypothesis=h))))

    async def _attach_check(args: dict[str, Any]) -> ToolResult:
        pid = str(args.get("prediction_id", ""))
        wsid = str(args.get("worker_session_id", ""))
        if not pid or not wsid:
            return _error(
                "status=rejected reason=prediction_id and worker_session_id must both be non-empty"
            )
        obs_rows: list[Observation] = []
        for idx, o in enumerate(args.get("observations", []) or []):
            if not isinstance(o, dict):
                return _error(
                    f"status=rejected reason=observations[{idx}] must be object"
                )
            obs_rows.append(_observation_from(o))
        interp_in = args.get("interpretation") or {}
        if not isinstance(interp_in, dict):
            return _error("status=rejected reason=interpretation must be an object")
        verdict = str(args.get("verdict_proposal", ""))
        mode = str(args.get("mode", "verify"))
        if mode == "steelman" and "steelman" not in verdict.lower():
            # Make the §7.2 sentinel visible without forcing the LLM to repeat it;
            # the gate matches on word-boundary so appending is safe.
            verdict = (verdict + " [steelman]").strip()
        check = CheckResult(
            id=_new_id("C"),
            prediction_id=pid,
            worker_session_id=wsid,
            observations=obs_rows,
            interpretation=Interpretation(
                proposed_update=str(interp_in.get("proposed_update", "")),
                reasoning=str(interp_in.get("reasoning", "")),
                confidence=str(interp_in.get("confidence", "")),
            ),
            verdict_proposal=verdict,
            ts=time.time(),
        )
        # Persist observations first so they can be cited / cached, then attach.
        for obs in obs_rows:
            gate.apply(UpdateProposal(op="record_observation", observation=obs))
        return _ok(
            _render(
                gate.apply(
                    UpdateProposal(op="attach_check", prediction_id=pid, check=check)
                )
            )
        )

    async def _propose_update(args: dict[str, Any]) -> ToolResult:
        op = str(args.get("op", "")).strip()
        if not op:
            return _error("status=rejected reason=op must be non-empty")
        if op in {"propose", "record_observation", "attach_check", "record_symptom"}:
            return _error(f"status=rejected reason=op={op!r} has a dedicated tool")
        proposal = _build_proposal_from_args(op, args)
        if isinstance(proposal, str):
            return _error(f"status=rejected reason={proposal}")
        return _ok(_render(gate.apply(proposal)))

    fns: dict[str, Any] = {
        "record_symptom": _record_symptom,
        "record_observation": _record_observation,
        "propose_hypothesis": _propose_hypothesis,
        "attach_check": _attach_check,
        "propose_update": _propose_update,
    }
    for name, fn in fns.items():
        api.register_tool(
            FunctionTool(
                name=name,
                description=DESCRIPTIONS[name],
                parameters=PARAMS[name],
                fn=fn,
                metadata={"idempotent": False, "rca_op": name},
            )
        )


# propose_update payload builders. Each branch returns either an
# ``UpdateProposal`` (success) or a string (validation error).


def _h_from_payload(
    payload: Any,
    *,
    parent_id: str | None = None,
    parent_ids: list[str] | None = None,
) -> Hypothesis | str:
    if not isinstance(payload, dict):
        return "child must be an object"
    claim = str(payload.get("claim", "")).strip()
    if not claim:
        return "child.claim must be non-empty"
    parents = list(parent_ids) if parent_ids else ([parent_id] if parent_id else [])
    return Hypothesis(
        id=str(payload.get("id") or _new_id("H")),
        claim=claim,
        parent_ids=parents,
        rationale=str(payload.get("rationale", "")),
    )


def _build_proposal_from_args(op: str, args: dict[str, Any]) -> UpdateProposal | str:
    target = str(args.get("target_id", "")).strip()
    if op in {"confirm", "refute"}:
        if not target:
            return f"op={op} requires target_id"
        return UpdateProposal(op=op, target_id=target)
    if op == "suspend":
        if not target:
            return "op=suspend requires target_id"
        return UpdateProposal(
            op="suspend", target_id=target, reason=str(args.get("reason", ""))
        )
    if op == "refine":
        if not target:
            return "op=refine requires target_id + child={claim,...}"
        child = _h_from_payload(args.get("child"), parent_id=target)
        if isinstance(child, str):
            return child
        return UpdateProposal(
            op="refine",
            hypothesis=child,
            target_id=target,
            reason=str(args.get("reason", "")),
        )
    if op == "supersede":
        cp = args.get("child") or {}
        rid = cp.get("id") if isinstance(cp, dict) else None
        if not target or not isinstance(rid, str) or not rid:
            return "op=supersede requires target_id + child={id:'<existing>'}"
        return UpdateProposal(
            op="supersede",
            target_id=target,
            hypothesis=Hypothesis(id=rid, claim=""),
        )
    if op == "split":
        children_in = list(args.get("children", []) or [])
        if not target or len(children_in) < 2:
            return "op=split requires target_id + children[len>=2]"
        children: list[Hypothesis] = []
        for c in children_in:
            built = _h_from_payload(c, parent_id=target)
            if isinstance(built, str):
                return f"op=split: {built}"
            children.append(built)
        return UpdateProposal(op="split", target_id=target, children=children)
    if op == "merge":
        sources = [
            str(sid) for sid in (args.get("sources", []) or []) if isinstance(sid, str)
        ]
        if len(sources) < 2:
            return "op=merge requires sources[len>=2] + child={claim,...}"
        merged = _h_from_payload(args.get("child"), parent_ids=sources)
        if isinstance(merged, str):
            return merged
        return UpdateProposal(op="merge", hypothesis=merged, sources=sources)
    return f"unknown op={op!r}"
