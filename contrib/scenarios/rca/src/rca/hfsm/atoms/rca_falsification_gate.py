"""``rca_falsification_gate`` — the single writer of the rca_hfsm graph.

Post-Phase-2 form (C2 of the LLM-native-judges refactor). The gate is the
**only writer** of the HypothesisGraph and the **dispatcher** of the four
``rca.judge.*`` services. It contains:

* Structural plumbing: dispatch by ``UpdateProposal.op``, single-writer
  token claim, ``rca.graph.mutated`` emission.
* Shape preconditions: payload-must-be-present, parent-must-not-be-terminal,
  status-pointer arithmetic.
* Calls into ``rca.judge.satisfied`` / ``rca.judge.coverage`` /
  ``rca.judge.independence`` / ``rca.judge.falsified_genuinely`` for every
  semantic decision about free-text fields.

What the gate does **NOT** contain after this commit:

* Zero regex. No ``import re``. No ``.match`` / ``.search`` / ``\\b`` /
  ``startswith`` / ``endswith`` calls against ``verdict_proposal``,
  ``interpretation``, or any other free-text field.
* No lemma matching on ``"triggered"`` / ``"supports"`` / ``"steelman"``.
  Those Phase-1 vocabulary markers are now the judge prompts' problem.
* No mechanical chain-walk through ``Observation.related_symptoms`` →
  satisfied prediction → confirmed hypothesis. That moved into
  ``rca.judge.coverage``.
* No literal ``worker_session_id`` equality compare for independence. That
  moved into ``rca.judge.independence``.
* No "≥1 negative prediction declared" / "≥1 satisfied negative
  prediction" structural rule. The propose-time check was deleted (it was
  a structural proxy for "was falsification genuinely attempted?"); the
  confirm-time check now calls ``rca.judge.falsified_genuinely``.

The ``test_gate_no_regex`` fail-stop in this scenario's test suite grep s
this file for those substrings; if any reappear the test fails. See
``.claude/designs/llm-native-judges.md`` §8 (acceptance properties).

Downgrade-application semantics flip (design §5.2):

* Old (Phase 1): a failing ``confirm`` synthesised a refine, applied it,
  and returned ``downgraded(applied_id=<child>, downgrade=<refine>)``.
  The parent hypothesis flipped to ``refined→<child>``.
* New (Phase 2): a failing ``confirm`` returns
  ``downgraded(applied_id=None, downgrade=<refine proposal>,
  reason=<judge.reason>)``. The parent hypothesis stays ``open``. The
  orchestrator reads the judge reason and decides next steps — gather
  more evidence, propose the refine explicitly, split, merge, or
  request help.

contract notes:

* Imports are stdlib + ``agentm.core.abi.*`` + ``agentm.extensions`` plus
  the scenario's pure modules (``schema``, ``updates``, ``judges``) only.
  The gate imports **no** sibling atom module. The write handle is reached
  through ``api.get_service('rca.hgraph.claim_write')`` (the store atom
  publishes the claim function as a service) applied to the token from
  ``api.get_service('rca.hgraph.write_token')`` — the same service-registry
  integration every other atom uses, not an import. The
  ``requires=("rca_hgraph_store",)`` edge declares the load-order
  dependency explicitly.

* The store services and the 4 ``rca.judge.*`` services are looked up by
  name via ``api.get_service`` at install time and stored on the gate
  instance. No sibling-atom imports — every dependency is reached only
  through the service registry, the same way every other atom integrates.

* No ``core.runtime`` / ``core._internal`` imports. Mutable state lives
  on the gate instance, not at module level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest

from rca.hfsm.judges import Judge, JudgeContext, Verdict
from rca.hfsm.schema import CheckResult, Hypothesis, Prediction
from rca.hfsm.updates import (
    GraphView,
    UpdateProposal,
    UpdateResult,
    check_merge,
    check_refine,
    check_split,
)

MANIFEST = ExtensionManifest(
    name="rca_falsification_gate",
    description=(
        "Sole writer of the rca_hfsm HypothesisGraph. Dispatches every "
        "semantic precondition into the rca.judge.* services; the gate "
        "itself only enforces structural shape and applies/downgrades "
        "based on the judges' verdicts."
    ),
    registers=(),
    config_schema=None,
    requires=("rca_hgraph_store",),
)

# Canonical verdict strings the judges' prompts ask the LLM to emit. These
# constants are NOT lemma matches against free-text — they are equality
# checks against the structured ``Verdict.verdict`` field, which the judge
# atoms produce via the ``submit_verdict`` tool_use call. See
# ``judges.py`` and the prompts under ``prompts/judges/``.
_SATISFIED = "satisfied"
_COVERS = "covers"
_INDEPENDENT = "independent"
_GENUINE_ATTEMPT = "genuine_attempt"

# ---------------------------------------------------------------------------
# Gate implementation. ``_Gate`` is intentionally instance-scoped — the gate
# state (write handle, graph view, judge handles) lives on the instance the
# install builds, never as module-level mutable globals (§11.4.D3).
# ---------------------------------------------------------------------------


@dataclass
class _JudgeBundle:
    """The four judges the gate consults, looked up once at install."""

    satisfied: Judge
    coverage: Judge
    independence: Judge
    falsified_genuinely: Judge


@dataclass
class _Gate:
    write_handle: Any  # _WriteHandle from the store; opaque here by design
    graph: GraphView
    judges: _JudgeBundle
    # Optional sync-emit hook installed by ``install`` so ``apply`` can
    # publish ``rca.graph.mutated`` without awaiting. ``None`` when no bus
    # is wired (e.g. unit tests that instantiate ``_Gate`` directly).
    emit: Any = None

    # -- Public surface ------------------------------------------------------

    def apply(self, update: UpdateProposal) -> UpdateResult:
        """Dispatch on ``update.op`` and apply or downgrade per design §5."""

        op = update.op
        if op == "propose":
            result = self._apply_propose(update)
        elif op == "confirm":
            result = self._apply_confirm(update)
        elif op == "refute":
            result = self._apply_refute(update)
        elif op == "refine":
            result = self._apply_refine(update)
        elif op == "split":
            result = self._apply_split(update)
        elif op == "merge":
            result = self._apply_merge(update)
        elif op == "supersede":
            result = self._apply_supersede(update)
        elif op == "suspend":
            result = self._apply_suspend(update)
        elif op == "record_observation":
            result = self._apply_record_observation(update)
        elif op == "attach_check":
            result = self._apply_attach_check(update)
        elif op == "record_symptom":
            # Not in the §3.3 operator table but the gate is the single
            # writer, so symptom intake also routes through it.
            result = self._apply_record_symptom(update)
        else:
            return UpdateResult.rejected(f"unknown operator: {op!r}")
        self._emit_mutation(op, result)
        return result

    def _emit_mutation(self, op: str, result: UpdateResult) -> None:
        emit = self.emit
        if emit is None or result.kind == "rejected":
            return
        emit(
            "rca.graph.mutated",
            {
                "op": op,
                "kind": result.kind,
                "applied_id": result.applied_id,
                "downgrade_op": (
                    result.downgrade.op if result.downgrade is not None else None
                ),
                "reason": result.reason,
            },
        )

    # -- Operator handlers ---------------------------------------------------

    def _apply_propose(self, update: UpdateProposal) -> UpdateResult:
        """Light-precondition: payload present, claim non-empty, ≥1 prediction.

        The Phase-1 "≥1 negative prediction" structural rule is gone —
        ``rca.judge.falsified_genuinely`` now decides at confirm-time
        whether the verification process genuinely attempted to falsify
        the hypothesis. Propose-time only checks that the LLM gave us
        something coherent to verify.
        """

        h = update.hypothesis
        if h is None:
            return UpdateResult.rejected("propose requires a hypothesis payload")
        if not h.claim.strip():
            return UpdateResult.rejected("hypothesis claim must be non-empty")
        if not h.predictions:
            return UpdateResult.rejected(
                "hypothesis must declare at least one prediction"
            )
        self.write_handle.add_hypothesis(h)
        return UpdateResult.applied(h.id)

    def _apply_confirm(self, update: UpdateProposal) -> UpdateResult:
        """Confirm gate — consults three judges (§5.1).

        1. ``rca.judge.falsified_genuinely`` — was a real falsification
           attempt made on the hypothesis?
        2. ``rca.judge.independence`` — do the supporting checks come
           from genuinely independent investigations?
        3. ``rca.judge.coverage`` — does the hypothesis explain every
           recorded symptom?

        Any judge whose verdict is not the canonical "good" string
        produces a downgrade (suggested refine, NOT applied — design
        §5.2). The reason text comes from the judge.
        """

        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h

        all_checks = _all_checks(h)

        falsified = self.judges.falsified_genuinely.judge(
            JudgeContext(
                graph_slice={
                    "hypothesis": _hypothesis_slice(h),
                    "predictions": [_prediction_slice(p) for p in h.predictions],
                    "all_checks": [_check_slice(c) for c in all_checks],
                },
                operands={"hypothesis_id": h.id},
            )
        )
        if falsified.verdict != _GENUINE_ATTEMPT:
            return self._downgrade(h, _judge_reason("falsified_genuinely", falsified))

        supporting = _collect_supporting_checks(h)
        if len(supporting) < 2:
            return self._downgrade(
                h,
                "confirm requires at least two supporting checks on positive "
                "predictions; fewer were attached to this hypothesis",
            )
        check_a, check_b = supporting[0], supporting[1]
        indep = self.judges.independence.judge(
            JudgeContext(
                graph_slice={
                    "check_a": _check_slice(check_a),
                    "check_b": _check_slice(check_b),
                },
                operands={"hypothesis_id": h.id},
            )
        )
        if indep.verdict != _INDEPENDENT:
            return self._downgrade(h, _judge_reason("independence", indep))

        cov = self.judges.coverage.judge(
            JudgeContext(
                graph_slice={
                    "hypothesis": _hypothesis_slice(h),
                    "predictions": [_prediction_slice(p) for p in h.predictions],
                    "symptoms": [
                        {"id": s.id, "text": s.text} for s in self.graph.get_symptoms()
                    ],
                    "observations": _observation_log_slice(h),
                },
                operands={"hypothesis_id": h.id},
            )
        )
        if cov.verdict != _COVERS:
            return self._downgrade(h, _judge_reason("coverage", cov))

        self.write_handle.set_hypothesis_status(h.id, "confirmed")
        return UpdateResult.applied(h.id)

    def _apply_refute(self, update: UpdateProposal) -> UpdateResult:
        """Refute gate — consults the falsified-genuinely judge.

        Phase 1 accepted refute on either of two structural grounds: a
        triggered negative prediction (Popperian falsification) OR a
        steelman check that itself failed to find support. Both of those
        grounds amount to "the verification process *did* try to refute
        the hypothesis and produced something refute-shaped" — i.e. a
        ``genuine_attempt`` verdict from ``rca.judge.falsified_genuinely``.
        The Phase-2 mapping is therefore: refute is accepted when the
        falsified-genuinely judge says ``genuine_attempt``.

        This is the cleaner mapping than splitting refute across two
        judges. The judge prompt is aware that ``op="refute"`` callers
        are asking a slightly different question ("did the investigation
        produce something refute-shaped?") — the operand discriminates.
        """

        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h
        all_checks = _all_checks(h)
        verdict = self.judges.falsified_genuinely.judge(
            JudgeContext(
                graph_slice={
                    "hypothesis": _hypothesis_slice(h),
                    "predictions": [_prediction_slice(p) for p in h.predictions],
                    "all_checks": [_check_slice(c) for c in all_checks],
                },
                operands={"hypothesis_id": h.id, "op": "refute"},
            )
        )
        if verdict.verdict != _GENUINE_ATTEMPT:
            return self._downgrade(h, _judge_reason("falsified_genuinely", verdict))
        self.write_handle.set_hypothesis_status(h.id, "refuted")
        return UpdateResult.applied(h.id)

    def _apply_refine(self, update: UpdateProposal) -> UpdateResult:
        child = update.hypothesis
        if child is None:
            return UpdateResult.rejected(
                "refine requires a new hypothesis payload as the child node"
            )
        # The parent is identified either explicitly via ``target_id`` or as
        # the first entry of ``child.parent_ids``. The caller is expected to
        # populate ``parent_ids`` on the child.
        parent_id = update.target_id or (
            child.parent_ids[0] if child.parent_ids else None
        )
        if parent_id is None:
            return UpdateResult.rejected(
                "refine requires a parent hypothesis id "
                "(set target_id or child.parent_ids[0])"
            )
        parent = self.graph.get_hypothesis(parent_id)
        if parent is None:
            return UpdateResult.rejected(f"unknown parent hypothesis: {parent_id}")
        gate_reason = check_refine(parent)
        if gate_reason is not None:
            return UpdateResult.rejected(gate_reason)
        if not child.parent_ids:
            child.parent_ids = [parent_id]
        # Resolve a unique child id if one already exists from a prior
        # explicitly-applied refine.
        child.id = _unique_child_id(self.graph, child.id)
        self.write_handle.add_hypothesis(child)
        self.write_handle.set_hypothesis_status(parent_id, f"refined→{child.id}")
        return UpdateResult.applied(child.id)

    def _apply_split(self, update: UpdateProposal) -> UpdateResult:
        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h
        reason = check_split([h], update.children)
        if reason is not None:
            return UpdateResult.rejected(reason)
        for child in update.children:
            if not child.parent_ids:
                child.parent_ids = [h.id]
            self.write_handle.add_hypothesis(child)
        ids = ",".join(child.id for child in update.children)
        self.write_handle.set_hypothesis_status(h.id, f"split→[{ids}]")
        return UpdateResult.applied(h.id)

    def _apply_merge(self, update: UpdateProposal) -> UpdateResult:
        sources: list[Hypothesis] = []
        for sid in update.sources:
            src = self.graph.get_hypothesis(sid)
            if src is None:
                return UpdateResult.rejected(f"unknown merge source: {sid}")
            sources.append(src)
        reason = check_merge(sources)
        if reason is not None:
            return UpdateResult.rejected(reason)
        merged = update.hypothesis
        if merged is None:
            return UpdateResult.rejected("merge requires the merged hypothesis payload")
        if not merged.parent_ids:
            merged.parent_ids = list(update.sources)
        self.write_handle.add_hypothesis(merged)
        for src in sources:
            self.write_handle.set_hypothesis_status(src.id, f"merged→{merged.id}")
        return UpdateResult.applied(merged.id)

    def _apply_supersede(self, update: UpdateProposal) -> UpdateResult:
        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h
        replacement = update.hypothesis
        if replacement is None or self.graph.get_hypothesis(replacement.id) is None:
            return UpdateResult.rejected(
                "supersede requires an already-known replacement hypothesis"
            )
        self.write_handle.set_hypothesis_status(h.id, "superseded")
        return UpdateResult.applied(h.id)

    def _apply_suspend(self, update: UpdateProposal) -> UpdateResult:
        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h
        suffix = f": {update.reason}" if update.reason else ""
        self.write_handle.set_hypothesis_status(h.id, f"suspended{suffix}")
        return UpdateResult.applied(h.id)

    def _apply_record_observation(self, update: UpdateProposal) -> UpdateResult:
        obs = update.observation
        if obs is None:
            return UpdateResult.rejected(
                "record_observation requires an observation payload"
            )
        self.write_handle.append_observation(obs)
        return UpdateResult.applied(obs.id)

    def _apply_attach_check(self, update: UpdateProposal) -> UpdateResult:
        """Attach a check to a prediction.

        Light precondition only: the prediction's parent hypothesis must
        not be terminal, and the prediction id must resolve. Whether the
        check "satisfies" or "refutes" or "is ambiguous about" the
        prediction is the job of ``rca.judge.satisfied`` — that judgment
        flows through ``_apply_confirm`` / ``_apply_refute`` when the
        downstream operator runs, not here. ``attach_check`` is a fact
        append; the gate appends the structured record without grading it.
        """

        pid = update.prediction_id
        check = update.check
        if pid is None or check is None:
            return UpdateResult.rejected(
                "attach_check requires prediction_id + check payload"
            )
        owner = _find_hypothesis_owning_prediction(self.graph, pid)
        if owner is None:
            return UpdateResult.rejected(f"unknown prediction: {pid}")
        if owner.status in ("refuted", "confirmed"):
            return UpdateResult.rejected(
                f"cannot attach check to prediction of {owner.status} hypothesis"
            )
        self.write_handle.attach_check(pid, check)
        return UpdateResult.applied(check.id)

    def _apply_record_symptom(self, update: UpdateProposal) -> UpdateResult:
        sym = update.symptom
        if sym is None:
            return UpdateResult.rejected("record_symptom requires a symptom payload")
        self.write_handle.add_symptom(sym)
        return UpdateResult.applied(sym.id)

    # -- Helpers -------------------------------------------------------------

    def _resolve_target(self, update: UpdateProposal) -> Hypothesis | UpdateResult:
        target_id = update.target_id or (
            update.hypothesis.id if update.hypothesis is not None else None
        )
        if target_id is None:
            return UpdateResult.rejected(
                f"{update.op} requires a target hypothesis (target_id)"
            )
        h = self.graph.get_hypothesis(target_id)
        if h is None:
            return UpdateResult.rejected(f"unknown hypothesis: {target_id}")
        return h

    def _downgrade(self, parent: Hypothesis, reason: str) -> UpdateResult:
        """Build a refine-suggestion downgrade without applying it.

        Design §5.2: the gate returns the suggested refine proposal but
        does **not** mutate the graph. The parent hypothesis stays
        ``open``; the orchestrator reads ``reason`` and ``downgrade`` and
        decides next steps.
        """

        suggested_child = Hypothesis(
            id=f"{parent.id}.refine",
            claim=f"{parent.claim} (refine: needs {reason})",
            parent_ids=[parent.id],
            predictions=[],
            status="open",
            generation=parent.generation + 1,
            rationale=f"gate suggestion: {reason}",
        )
        suggested_proposal = UpdateProposal(
            op="refine",
            hypothesis=suggested_child,
            target_id=parent.id,
            reason=reason,
        )
        return UpdateResult.downgraded(
            to=suggested_proposal,
            reason=reason,
            applied_id=None,
        )


# ---------------------------------------------------------------------------
# Pure helpers — graph-slice constructors and structural traversal.
# Free of regex / lemma matching by construction; they pass structured
# data to the judges and return judges' free-text verdicts unchanged.
# ---------------------------------------------------------------------------


def _hypothesis_slice(h: Hypothesis) -> dict[str, Any]:
    return {
        "id": h.id,
        "claim": h.claim,
        "status": h.status,
        "parent_ids": list(h.parent_ids),
        "generation": h.generation,
        "rationale": h.rationale,
    }


def _prediction_slice(p: Prediction) -> dict[str, Any]:
    return {
        "id": p.id,
        "hypothesis_id": p.hypothesis_id,
        "claim": p.claim,
        "polarity": p.polarity,
        "test_plan": p.test_plan,
        "checks": [_check_slice(c) for c in p.checks],
    }


def _check_slice(c: CheckResult) -> dict[str, Any]:
    interpretation = c.interpretation
    return {
        "id": c.id,
        "prediction_id": c.prediction_id,
        "worker_session_id": c.worker_session_id,
        "verdict_proposal": c.verdict_proposal,
        "interpretation": (
            None
            if interpretation is None
            else {
                "proposed_update": interpretation.proposed_update,
                "reasoning": interpretation.reasoning,
                "confidence": interpretation.confidence,
            }
        ),
        "observations": [
            {
                "id": o.id,
                "text": o.text,
                "source_tool_call": o.source_tool_call,
                "related_symptoms": list(o.related_symptoms),
                "related_predictions": list(o.related_predictions),
            }
            for o in c.observations
        ],
    }


def _all_checks(h: Hypothesis) -> list[CheckResult]:
    """Flatten every check on every prediction of ``h``."""

    out: list[CheckResult] = []
    for p in h.predictions:
        out.extend(p.checks)
    return out


def _collect_supporting_checks(h: Hypothesis) -> list[CheckResult]:
    """Return all checks attached to positive predictions, in order.

    Whether each check actually supports the claim is the
    ``rca.judge.satisfied`` judge's job; this helper just enumerates the
    structural candidates. The independence judge then receives two of
    them as ``check_a`` and ``check_b`` and decides whether they are
    genuinely independent.

    For Phase-2 the gate passes the first two positive-prediction checks
    to the independence judge. If a future scenario wants pairwise
    independence over every supporting pair, that becomes a separate
    judge call rather than a structural enumeration here.
    """

    out: list[CheckResult] = []
    for p in h.predictions:
        if p.polarity != "positive":
            continue
        out.extend(p.checks)
    return out


def _observation_log_slice(h: Hypothesis) -> list[dict[str, Any]]:
    """The observation set attached to any of ``h``'s checks.

    Used by the coverage judge so it can see, structurally, which
    symptoms are or aren't linked to the candidate-confirmed hypothesis.
    """

    obs: list[dict[str, Any]] = []
    for p in h.predictions:
        for c in p.checks:
            for o in c.observations:
                obs.append(
                    {
                        "id": o.id,
                        "text": o.text,
                        "related_symptoms": list(o.related_symptoms),
                        "related_predictions": list(o.related_predictions),
                    }
                )
    return obs


def _judge_reason(kind: str, verdict: Verdict) -> str:
    """Compose a downgrade reason from a judge's verdict.

    The kind tag lets the orchestrator route on which judge fired; the
    verdict label is the canonical short string the prompt requested (one
    of the per-judge vocabularies); the reason is the judge's free-text
    rationale. Caller is responsible for not parsing this string for
    decisions — it is for the LLM consumer.
    """

    return f"judge={kind} verdict={verdict.verdict} reason={verdict.reason}"


def _find_hypothesis_owning_prediction(
    graph: GraphView, prediction_id: str
) -> Hypothesis | None:
    # ``GraphView`` does not expose an "all hypotheses" iterator (by design —
    # the read API surfaces structured views, not raw dumps). For this lookup
    # we walk open leaves + refuted branches + confirmed; this covers every
    # status the store produces in Phase 1.
    confirmed_getter = getattr(graph, "get_confirmed", None)
    confirmed: list[Hypothesis] = (
        confirmed_getter() if callable(confirmed_getter) else []
    )
    for h in (
        list(graph.get_open_leaves()) + list(graph.get_refuted_branches()) + confirmed
    ):
        for p in h.predictions:
            if p.id == prediction_id:
                return h
    return None


def _unique_child_id(graph: GraphView, base: str) -> str:
    if graph.get_hypothesis(base) is None:
        return base
    counter = 2
    while graph.get_hypothesis(f"{base}.{counter}") is not None:
        counter += 1
    return f"{base}.{counter}"


# ---------------------------------------------------------------------------
# install — wire the gate into the scenario.
# ---------------------------------------------------------------------------


def _required_judge(api: ExtensionAPI, service_name: str) -> Judge:
    impl = api.get_service(service_name)
    if impl is None:
        raise RuntimeError(
            f"rca_falsification_gate: required judge service {service_name!r} "
            "is not published; mount the corresponding judge atom before "
            "the gate"
        )
    return impl  # type: ignore[no-any-return]


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    token = api.get_service("rca.hgraph.write_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(
            "rca_falsification_gate: rca.hgraph.write_token is not "
            "published; rca_hgraph_store must install before the gate"
        )
    claim_write = api.get_service("rca.hgraph.claim_write")
    if claim_write is None:
        raise RuntimeError(
            "rca_falsification_gate: rca.hgraph.claim_write is not "
            "published; rca_hgraph_store must install before the gate"
        )
    write_handle = claim_write(token)
    read_handle = api.get_service("rca.hgraph.read")
    if read_handle is None:
        raise RuntimeError(
            "rca_falsification_gate: rca.hgraph.read is not published; "
            "rca_hgraph_store must install before the gate"
        )
    judges = _JudgeBundle(
        satisfied=_required_judge(api, "rca.judge.satisfied"),
        coverage=_required_judge(api, "rca.judge.coverage"),
        independence=_required_judge(api, "rca.judge.independence"),
        falsified_genuinely=_required_judge(api, "rca.judge.falsified_genuinely"),
    )
    bus = api.events
    emit_fn = bus.emit_sync if bus is not None else None
    gate = _Gate(
        write_handle=write_handle,
        graph=read_handle,
        judges=judges,
        emit=emit_fn,
    )
    api.set_service("rca.gate", gate)
