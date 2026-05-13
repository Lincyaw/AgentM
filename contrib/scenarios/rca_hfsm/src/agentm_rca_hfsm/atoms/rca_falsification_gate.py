"""``rca_falsification_gate`` — the single writer of the rca_hfsm graph.

Implements design §7 in full. Acquires the store's write handle at install
time using the token the store publishes via ``rca.hgraph.write_token``
(design §7.4 single-writer property), then publishes ``rca.gate`` with a
single public method ``gate.apply(update) -> UpdateResult``.

Semantics (design §7.1, §7.2, §7.3):

* ``propose(H)`` with zero negative predictions → ``rejected`` with the
  precise design-§6.2 reason string.
* ``confirm(H)`` failing any §7.1 precondition (satisfied negative
  prediction / independent positive workers / coverage of all symptoms) →
  ``downgraded`` to ``refine(H, reason)``. The downgrade is APPLIED — the
  LLM gets back both the original failure AND the fact that the refine
  succeeded.
* ``refute(H)`` failing both §7.2 conditions (triggered negative OR
  steelman attempt) → ``downgraded`` to ``refine`` likewise.
* Light-precondition operators (``refine``/``split``/``merge``/
  ``supersede``/``suspend``/``record_observation``/``attach_check``) apply
  on success; on precondition failure → ``rejected`` (no downgrade — these
  operators are themselves graph-restructuring and have no "softer"
  alternative).

§11 contract notes:

* Imports are stdlib + ``agentm.core.abi.*`` + ``agentm.extensions`` plus
  the scenario's pure modules (``schema``, ``updates``) and the store atom
  module for its module-level ``claim_write_handle``. The store atom is
  the only sibling atom referenced — and only via a top-level function
  explicitly designed for the gate's use (commit 1 plan §"Files added"
  bullet 3). The validator's forbidden-prefix list rejects only
  ``agentm.extensions.builtin.*`` / ``_agentm_contrib__*`` /
  ``agentm._scenarios.*`` — scenario-local ``agentm_rca_hfsm.atoms.*`` is
  outside that list, so this import is legal.

* No ``core.runtime`` / ``core._internal`` imports. Mutable globals are
  not module-level (the gate state is per-install, stored on the gate
  instance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from agentm_rca_hfsm.atoms.rca_hgraph_store import claim_write_handle
from agentm_rca_hfsm.schema import Hypothesis
from agentm_rca_hfsm.updates import (
    GraphView,
    UpdateProposal,
    UpdateResult,
    check_confirm,
    check_merge,
    check_propose,
    check_refine,
    check_refute,
    check_split,
)


MANIFEST = ExtensionManifest(
    name="rca_falsification_gate",
    description=(
        "Sole writer of the rca_hfsm HypothesisGraph. Mediates every "
        "mutation through the §7 falsification preconditions and downgrades "
        "failing confirms/refutes to refine-with-reason."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=(),
)


# ---------------------------------------------------------------------------
# Gate implementation. ``_Gate`` is intentionally instance-scoped — the gate
# state (write handle, graph view) lives on the instance the install builds,
# never as module-level mutable globals (§11.4.D3).
# ---------------------------------------------------------------------------


@dataclass
class _Gate:
    write_handle: Any  # _WriteHandle from the store; opaque here by design
    graph: GraphView
    # Optional sync-emit hook installed by ``install`` so ``apply`` can
    # publish ``rca.graph.mutated`` without awaiting. ``None`` when no bus
    # is wired (e.g. unit tests that instantiate ``_Gate`` directly).
    emit: Any = None

    # -- Public surface ------------------------------------------------------

    def apply(self, update: UpdateProposal) -> UpdateResult:
        """Dispatch on ``update.op`` and apply or downgrade per §7."""

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
        # Single-line emit on success/downgrade so FSM policy can observe
        # graph mutations without polling the read API. Commit-4 addition.
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
        h = update.hypothesis
        if h is None:
            return UpdateResult.rejected("propose requires a hypothesis payload")
        reason = check_propose(h)
        if reason is not None:
            return UpdateResult.rejected(reason)
        self.write_handle.add_hypothesis(h)
        return UpdateResult.applied(h.id)

    def _apply_confirm(self, update: UpdateProposal) -> UpdateResult:
        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h
        reason = check_confirm(h, self.graph)
        if reason is not None:
            return self._downgrade_to_refine(h, reason)
        self.write_handle.set_hypothesis_status(h.id, "confirmed")
        return UpdateResult.applied(h.id)

    def _apply_refute(self, update: UpdateProposal) -> UpdateResult:
        h = self._resolve_target(update)
        if isinstance(h, UpdateResult):
            return h
        reason = check_refute(h, self.graph)
        if reason is not None:
            return self._downgrade_to_refine(h, reason)
        self.write_handle.set_hypothesis_status(h.id, "refuted")
        return UpdateResult.applied(h.id)

    def _apply_refine(self, update: UpdateProposal) -> UpdateResult:
        child = update.hypothesis
        if child is None:
            return UpdateResult.rejected(
                "refine requires a new hypothesis payload as the child node"
            )
        # The parent is identified either explicitly via ``target_id`` or as
        # the first entry of ``child.parent_ids``. Phase 1 expects the
        # caller to populate ``parent_ids`` on the child.
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
            return UpdateResult.rejected(
                "merge requires the merged hypothesis payload"
            )
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
        # ``hypothesis`` carries the superseding node; light precondition:
        # the new node must already exist in the graph (it was proposed
        # earlier and is therefore "a more precise sibling").
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
        pid = update.prediction_id
        check = update.check
        if pid is None or check is None:
            return UpdateResult.rejected(
                "attach_check requires prediction_id + check payload"
            )
        # Light precondition: the prediction's parent hypothesis must not be
        # terminal. The store raises on unknown prediction.
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
            return UpdateResult.rejected(
                "record_symptom requires a symptom payload"
            )
        self.write_handle.add_symptom(sym)
        return UpdateResult.applied(sym.id)

    # -- Helpers -------------------------------------------------------------

    def _resolve_target(
        self, update: UpdateProposal
    ) -> Hypothesis | UpdateResult:
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

    def _downgrade_to_refine(
        self, parent: Hypothesis, reason: str
    ) -> UpdateResult:
        """Synthesise and apply a ``refine`` carrying ``reason``.

        The downgrade is APPLIED — the LLM gets back ``UpdateResult.downgraded``
        with both the executed refine proposal and its ``applied_id`` (the
        child node) so a separate retry round-trip is unnecessary.
        """

        child = Hypothesis(
            id=f"{parent.id}.refine",
            claim=f"{parent.claim} — refined: needs {reason}",
            parent_ids=[parent.id],
            predictions=[],
            status="open",
            generation=parent.generation + 1,
            rationale=f"gate downgrade: {reason}",
        )
        refine_proposal = UpdateProposal(
            op="refine",
            hypothesis=child,
            target_id=parent.id,
            reason=reason,
        )
        # Apply directly (bypassing dispatch) so we cannot accidentally
        # recurse if a future gate change re-routes refine through apply().
        applied_id: str | None = None
        precondition = check_refine(parent)
        if precondition is None:
            # Resolve a unique child id if one already exists from a prior
            # downgrade — append a numeric suffix.
            child.id = _unique_child_id(self.graph, child.id)
            self.write_handle.add_hypothesis(child)
            self.write_handle.set_hypothesis_status(
                parent.id, f"refined→{child.id}",
            )
            applied_id = child.id
        return UpdateResult.downgraded(
            to=refine_proposal, reason=reason, applied_id=applied_id,
        )


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
    for h in list(graph.get_open_leaves()) + list(graph.get_refuted_branches()) + confirmed:
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    token = api.get_service("rca.hgraph.write_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(
            "rca_falsification_gate: rca.hgraph.write_token is not "
            "published; rca_hgraph_store must install before the gate"
        )
    write_handle = claim_write_handle(token)
    read_handle = api.get_service("rca.hgraph.read")
    if read_handle is None:
        raise RuntimeError(
            "rca_falsification_gate: rca.hgraph.read is not published; "
            "rca_hgraph_store must install before the gate"
        )
    # Wire emit through the bus's ``emit_sync`` so ``apply`` (synchronous)
    # can publish ``rca.graph.mutated`` without awaiting. The FSM policy
    # atom subscribes via ``api.on('rca.graph.mutated', ...)``.
    bus = api.events
    emit_fn = bus.emit_sync if bus is not None else None
    gate = _Gate(write_handle=write_handle, graph=read_handle, emit=emit_fn)
    api.set_service("rca.gate", gate)
