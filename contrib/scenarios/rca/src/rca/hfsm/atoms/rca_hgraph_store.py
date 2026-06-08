"""``rca_hgraph_store`` — L1 hypothesis-graph store for the rca_hfsm scenario.

Owns the per-trace ``HypothesisGraph`` (design §3) and the ``ObservationLog``
(design §3.2). Publishes a public read handle as the service
``rca.hgraph.read`` and a one-shot write handle that the falsification gate
atom claims via the ``rca.hgraph.claim_write`` service — the claim function is
delivered over the service registry (not import-reached) and consumes a
shared-secret token (design §7.4 — single-writer property).

The token mechanism is the structural reply to "any atom that calls
``api.get_service('rca.hgraph.write')`` becomes a writer". Instead, this atom
publishes a per-install random token via ``rca.hgraph.write_token`` plus the
claim function via ``rca.hgraph.claim_write``; the write handle is only
obtainable by calling that claim function with the token *once* per token.
Each call to :func:`install` mints a fresh token, so multiple sessions in the
same process each get an independent (token, handle) pair; the one-shot rule
is per token, not per process. A second claim of the same token (or any
unknown token) raises ``RuntimeError``. Publishing the claim function as a
service changes nothing security-wise: the token is already a public service
and the one-shot claim still enforces single-writer.

§11 single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*`` only. The sibling ``schema`` import is a pure-data
module, not an atom, and is permitted by §11.4 (atoms may import their
scenario's pure modules; the validator's atom-to-atom rule names only other
atom files).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Final

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from rca.hfsm.schema import (
    CheckResult,
    Hypothesis,
    Observation,
    Symptom,
)


MANIFEST = ExtensionManifest(
    name="rca_hgraph_store",
    description=(
        "Owns the per-trace HypothesisGraph + ObservationLog and exposes "
        "rca.hgraph.read publicly. The write handle is gated by a one-shot "
        "token claimed via the rca.hgraph.claim_write service."
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
# State held inside a single install. Each ``install()`` call constructs one
# ``_StoreState`` and registers a token in the module-level pending registry
# so the future gate atom can claim the corresponding write handle without
# importing this module's atom-internal types.
# ---------------------------------------------------------------------------


@dataclass
class _StoreState:
    symptoms: dict[str, Symptom] = field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    obs_by_signature: dict[str, Observation] = field(default_factory=dict)


class _ReadHandle:
    """Read API published as ``rca.hgraph.read``.

    Every getter returns a defensive copy of the live collection. Atoms in
    the scenario rely on the snapshot semantics: a worker brief built from
    ``get_open_leaves()`` must not mutate when the gate then advances state.
    """

    def __init__(self, state: _StoreState) -> None:
        self._state = state

    def get_symptoms(self) -> list[Symptom]:
        return list(self._state.symptoms.values())

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return self._state.hypotheses.get(hypothesis_id)

    def get_open_leaves(self) -> list[Hypothesis]:
        # An open leaf has status == "open" and no other hypothesis lists it
        # as a parent. This computation is structural (not a stored flag), so
        # the gate can update statuses without bookkeeping a derived index.
        all_h = self._state.hypotheses
        has_child: set[str] = set()
        for h in all_h.values():
            for parent_id in h.parent_ids:
                has_child.add(parent_id)
        return [
            h for h in all_h.values()
            if h.status == "open" and h.id not in has_child
        ]

    def get_unexplained_symptoms(self) -> list[Symptom]:
        # A symptom is "explained" iff there exists a confirmed hypothesis
        # whose predictions carry ≥1 CheckResult with ≥1 Observation that
        # cites the symptom in its ``related_symptoms``. This is a purely
        # structural read — no regex on free-text, no judgment about
        # whether the verdict "supports" or "triggers" the claim.
        #
        # Phase-2 simplification: the Phase-1 definition layered the
        # ``is_prediction_satisfied`` regex check on top of this chain.
        # That regex moved into ``rca.judge.satisfied`` along with every
        # other free-text semantic check; the store now exposes only the
        # structural skeleton. The FINALIZE coverage gate
        # (``rca_finalize``) consumes this method's output as-is — once a
        # hypothesis is confirmed and its checks carry observations
        # linking back to a symptom, that symptom is explained for
        # FINALIZE's purposes.
        explained: set[str] = set()
        for h in self._state.hypotheses.values():
            if h.status != "confirmed":
                continue
            for p in h.predictions:
                for c in p.checks:
                    for obs in c.observations:
                        explained.update(obs.related_symptoms)
        return [
            s for s in self._state.symptoms.values()
            if s.id not in explained
        ]

    def get_refuted_branches(self) -> list[Hypothesis]:
        return [
            h for h in self._state.hypotheses.values()
            if h.status == "refuted"
        ]

    def get_confirmed(self) -> list[Hypothesis]:
        # Narrow read-only helper consumed by ``rca.judge.coverage`` (and
        # by historical Phase-1 callers that have since moved into judge
        # services). Kept narrow and read-only — not on the public
        # read-API surface listed in the design doc.
        return [
            h for h in self._state.hypotheses.values()
            if h.status == "confirmed"
        ]

    def get_observation_by_signature(self, signature: str) -> Observation | None:
        return self._state.obs_by_signature.get(signature)


class _WriteHandle:
    """Write API surfaced only via :func:`claim_write_handle`.

    Phase 1 ships the minimal append surface the gate atom (commit 2) will
    build its operators on top of. Each method is the smallest possible
    mutation; the gate composes them with its precondition checks.
    """

    def __init__(self, state: _StoreState) -> None:
        self._state = state

    def add_symptom(self, symptom: Symptom) -> None:
        self._state.symptoms[symptom.id] = symptom

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        self._state.hypotheses[hypothesis.id] = hypothesis

    def set_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        h = self._state.hypotheses.get(hypothesis_id)
        if h is None:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        h.status = status

    def append_observation(self, observation: Observation) -> None:
        self._state.observations.append(observation)
        # ``tool_signature`` may be the empty string for observations that
        # come from non-tool sources (e.g. user_intake); only signed obs
        # participate in memoization.
        if observation.tool_signature:
            self._state.obs_by_signature.setdefault(
                observation.tool_signature, observation,
            )

    def attach_check(self, prediction_id: str, check: CheckResult) -> None:
        """Append ``check`` to the matching prediction (commit 2 gate hook).

        Raises ``KeyError`` if the prediction id is not present on any known
        hypothesis. The gate atom ensures the prediction is open before
        calling; the store only enforces existence.
        """

        for h in self._state.hypotheses.values():
            for p in h.predictions:
                if p.id == prediction_id:
                    p.checks.append(check)
                    return
        raise KeyError(f"unknown prediction: {prediction_id}")


# ---------------------------------------------------------------------------
# Single-writer token registry. Module-level because the gate atom does not
# import this module (§11 atom-to-atom rule); it reaches the write handle
# through ``api.get_service('rca.hgraph.write_token')`` for the token and
# ``api.get_service('rca.hgraph.claim_write')`` for the claim function below,
# which :func:`install` publishes — no sibling-atom import involved.
# ---------------------------------------------------------------------------


_pending: Final[dict[str, _WriteHandle]] = {}
_claimed: Final[set[str]] = set()


def claim_write_handle(token: str) -> _WriteHandle:
    """Return the write handle for the install whose token matches ``token``.

    The one-shot rule is scoped to the token, not the process: each
    ``install`` mints a fresh token, so multiple sessions in the same
    process each get an independent (token, handle) pair. Any subsequent
    claim of the **same** token — or of an unknown token — raises
    ``RuntimeError``. This is the structural enforcement of design §7.4
    (single-writer property) without exposing the write API as a service.

    Tests covering acceptance scenario #6 live in
    ``tests/test_store_single_writer.py``.
    """

    if token in _claimed:
        raise RuntimeError(
            f"rca.hgraph write handle for token={token!r} has already been "
            "claimed; the falsification gate is the only legal writer "
            "(design §7.4) and may only claim its own token once"
        )
    handle = _pending.pop(token, None)
    if handle is None:
        # Mark the offending token as consumed so a retry with the same
        # bogus value still raises. A misconfigured atom that calls
        # claim_write_handle before the store installs (or with a stale
        # token) must not be able to silently win the race on retry.
        _claimed.add(token)
        raise RuntimeError(
            f"no pending rca.hgraph install matches token={token!r}; "
            "token rejected"
        )
    _claimed.add(token)
    return handle


def _reset_for_tests() -> None:
    """Wipe the token registry between tests.

    Test-only — production scenarios never tear an install down within the
    same process. Marked private and undocumented in the §11 surface.
    """

    _pending.clear()
    _claimed.clear()


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    state = _StoreState()
    read_handle = _ReadHandle(state)
    write_handle = _WriteHandle(state)
    token = secrets.token_hex(16)

    _pending[token] = write_handle

    api.set_service("rca.hgraph.read", read_handle)
    api.set_service("rca.hgraph.write_token", token)
    api.set_service("rca.hgraph.claim_write", claim_write_handle)
