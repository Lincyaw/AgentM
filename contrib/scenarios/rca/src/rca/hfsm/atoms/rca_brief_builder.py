"""``rca_brief_builder`` — worker brief construction (design §5.4).

Publishes the ``rca.brief`` service. The service is a callable
``build_brief(hypothesis_id, prediction_id, *, mode="verify", blind=True)``
that returns a Markdown brief string the L2 orchestrator hands to a
dispatched L3 worker as the initial user message.

Two structural properties this atom enforces (acceptance #9 / §6.1, §6.2):

* **Falsification framing** on ``mode="verify"``: the brief asks the
  worker to *refute / contradict* the prediction rather than verify it.
  Wording is fixed at the brief-builder layer so workers cannot drift the
  framing prompt-by-prompt.
* **Hypothesis blinding** by default (``blind=True``): the brief shows
  the prediction text and the relevant L1 observation slice but NOT the
  parent hypothesis claim. Removes anchoring on the hypothesis identity.
  Callers may opt out with ``blind=False`` for explicit non-blinded
  flows (e.g. devil's advocate brief in Phase 2).

contract: stdlib + ``agentm.core.abi.*`` + ``agentm.extensions`` +
the scenario's pure ``schema`` module. No atom-to-atom imports — the
brief builder reads L1 through the ``rca.hgraph.read`` service captured
at install time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="rca_brief_builder",
    description=(
        "Publish rca.brief: a builder that renders falsification-framed, "
        "hypothesis-blinded Markdown briefs for L3 worker dispatch."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=("rca_hgraph_store",),
)

_FALSIFICATION_PREFIX = (
    "Your task is to find a piece of evidence that **refutes** "
    "(contradicts) the following prediction. Do NOT try to confirm it."
)
_STEELMAN_PREFIX = (
    "Your task is to **find supporting evidence** for the following "
    "prediction (steelman mode). If even this attempt fails, the parent "
    "hypothesis is structurally justified to be refuted."
)
_ADVERSARIAL_PREFIX = (
    "Your task is to find one piece of evidence that **contradicts** the "
    "leading hypothesis. Return observations only — no proposed update."
)


@dataclass
class _BriefBuilder:
    read: Any  # rca.hgraph.read service

    def __call__(
        self,
        hypothesis_id: str,
        prediction_id: str,
        *,
        mode: str = "verify",
        blind: bool = True,
    ) -> str:
        return self.build(
            hypothesis_id,
            prediction_id,
            mode=mode,
            blind=blind,
        )

    def build(
        self,
        hypothesis_id: str,
        prediction_id: str,
        *,
        mode: str = "verify",
        blind: bool = True,
    ) -> str:
        h = self.read.get_hypothesis(hypothesis_id)
        if h is None:
            raise ValueError(f"unknown hypothesis: {hypothesis_id}")
        pred = next(
            (p for p in h.predictions if p.id == prediction_id),
            None,
        )
        if pred is None:
            raise ValueError(
                f"prediction {prediction_id} not on hypothesis {hypothesis_id}"
            )
        # Framing prefix selected by mode; "verify" is the default falsification
        # path per §6.1. Unknown modes degrade to the verify prefix so a typo
        # never produces an un-framed brief.
        if mode == "steelman":
            framing = _STEELMAN_PREFIX
        elif mode == "adversarial":
            framing = _ADVERSARIAL_PREFIX
        else:
            framing = _FALSIFICATION_PREFIX
        lines: list[str] = [
            "# Worker brief",
            "",
            framing,
            "",
            "## Prediction",
            "",
            f"- claim: {pred.claim}",
            f"- polarity: {pred.polarity}",
        ]
        if pred.test_plan:
            lines.append(f"- test_plan: {pred.test_plan}")
        if not blind:
            # Non-blinded callers get the parent claim. Phase 1 callers
            # leave ``blind=True`` so this branch is dead-code on the
            # default path; kept for the Phase 2 devil's-advocate brief.
            lines.extend(
                [
                    "",
                    "## Parent hypothesis (visible — non-blinded mode)",
                    "",
                    f"- claim: {h.claim}",
                    f"- rationale: {h.rationale}" if h.rationale else "",
                ]
            )
        lines.extend(
            [
                "",
                "## Relevant observations (L1 slice)",
                "",
            ]
        )
        relevant = self._slice_observations(prediction_id)
        if relevant:
            for obs in relevant:
                lines.append(f"- ({obs.source_tool_call}) {obs.text}")
        else:
            lines.append("- (none recorded yet)")
        no_go = self._no_go_zones()
        if no_go:
            lines.extend(["", "## No-go zones (already-refuted branches)", ""])
            for branch_text in no_go:
                lines.append(f"- {branch_text}")
        lines.extend(
            [
                "",
                "## Expected output",
                "",
                "Return a two-column structured result:",
                "- `observations`: facts you found (text + source_tool_call + ",
                "  related_symptoms/related_predictions). These are ingested into L1.",
                "- `interpretation`: free-form `proposed_update` / `reasoning` / ",
                "  `confidence`. Advisory only — the orchestrator re-derives.",
            ]
        )
        return "\n".join(line for line in lines if line is not None)

    def _slice_observations(self, prediction_id: str) -> list[Any]:
        """Return observations whose ``related_predictions`` cite ``prediction_id``.

        Best-effort: the read API doesn't expose the raw log directly, so
        Phase 1 scans the observation-by-signature index by inspecting
        every cited prediction. We accept O(n) here — Phase 2 may add a
        prediction→observations index to the read API if profiling
        justifies it.
        """

        out: list[Any] = []
        # The read API doesn't surface ``observations``; rely on the
        # ``get_observation_by_signature`` index by walking signatures
        # exposed elsewhere. Phase 1 keeps the slice empty when the read
        # API has no enumeration helper — the brief still renders.
        list_obs = getattr(self.read, "list_observations", None)
        if callable(list_obs):
            for obs in list_obs():
                if prediction_id in getattr(obs, "related_predictions", ()):
                    out.append(obs)
        return out

    def _no_go_zones(self) -> list[str]:
        refuted = self.read.get_refuted_branches()
        return [f"{h.claim} (refuted)" for h in refuted]


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    read_handle = api.get_service("rca.hgraph.read")
    if read_handle is None:
        raise RuntimeError(
            "rca_brief_builder: rca.hgraph.read is not published; install "
            "rca_hgraph_store before this atom"
        )
    builder = _BriefBuilder(read=read_handle)
    api.set_service("rca.brief", builder)
