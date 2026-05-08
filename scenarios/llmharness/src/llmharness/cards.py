"""AFC card loader exposed as the diagnostic agent's `cards_list` /
`cards_get` tools.

Cards live as YAML at ``scenarios/llmharness/references/papers/cards/<class>/*.yaml``
and are the *reference vocabulary* the cognitive-audit child session draws on
when articulating drift findings. The audit does not receive the cards as
prompt content; it retrieves them on demand through the two functions below
(see design §4.4).

Two design rules are load-bearing for this module:

1. ``axis_hint`` lives in **Python code** (`_AXIS_HINT`), not in the YAML
   schema. The YAML files are a public contract for the rca-autorl
   downstream consumer; adding a field there would couple the three-axis
   audit partition to that contract. See design §4.4 / §6.3.

2. YAML loading is **lazy** — ``cards_list()`` performs no I/O at import
   time. The first call walks the cards root and caches the result.
   This keeps imports cheap and lets test fixtures override
   ``LLMHARNESS_CARDS_ROOT`` before the first call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Axis-hint mapping
# ---------------------------------------------------------------------------
# Each card in the corpus is annotated with one of:
#   1 — backward continuity / context binding (axis 1)
#   2 — forward fulfillment / progress (axis 2)
#   3 — content correctness / internal consistency / hallucination (axis 3)
#   None — genuinely outside the cognitive triad (e.g. pure system faults)
#
# Seeds come from design §3.1-§3.3 "Cards covered" lists. Cards not seeded
# there are classified pragmatically by their `defect.mechanism`. `None` is
# reserved for infrastructure faults that surface as agent-environment
# breakage rather than thought-graph defects.
#
# Invariant: every card_id loaded from disk MUST have a key here. If a new
# card is added, this dict must be updated — no silent fallback to None.
_AXIS_HINT: dict[str, int | None] = {
    # --- memory/ ---
    "AFC-0001": 3,  # over-simplification (lossy summary corrupts content)
    "AFC-0002": 3,  # false memory / fabricated recall (hallucination)
    "AFC-0003": 3,  # retrieval failure (content lost / wrong)
    "AFC-0004": 1,  # conversation-history-wipe (continuity break)
    "AFC-0039": 3,  # post-observation evidence fabrication (hallucination)
    "AFC-0040": 1,  # long-horizon abductive drift (revives refuted hypothesis — continuity)
    # --- reflection/ ---
    "AFC-0005": 3,  # progress-misassessment (internal consistency)
    "AFC-0006": 3,  # outcome-misinterpretation (content correctness)
    "AFC-0007": 3,  # causal misattribution (content correctness)
    "AFC-0008": 3,  # reflection hallucination
    # --- planning/ ---
    "AFC-0009": 2,  # constraint ignorance (forward fulfillment)
    "AFC-0010": 2,  # impossible action (forward fulfillment)
    "AFC-0011": 2,  # inefficient plan (forward fulfillment quality)
    "AFC-0012": 1,  # backtracking absence (continuity break)
    "AFC-0038": 1,  # hypothesis-irrelevant tool (orphan move — continuity)
    # --- action/ ---
    "AFC-0013": 1,  # planning-action disconnect (continuity)
    "AFC-0014": 2,  # action format error (action can't fulfill)
    "AFC-0015": 2,  # action parameter error (action can't fulfill)
    "AFC-0035": 3,  # defective generated artifact (content correctness)
    # --- specification/ ---
    "AFC-0016": 2,  # task spec deviation (forward fulfillment of original task)
    "AFC-0017": 2,  # role spec deviation (forward fulfillment relative to role)
    "AFC-0018": 2,  # goal drift (forward fulfillment of original task)
    "AFC-0034": 2,  # under-specified objective (declared intent lacks fulfillment criteria)
    # --- inter_agent/ ---
    "AFC-0019": 1,  # repeated handled work (continuity break across peers)
    "AFC-0020": 3,  # ambiguous request to peer (content correctness of message)
    "AFC-0021": 1,  # information withholding (continuity / handoff break)
    "AFC-0022": 1,  # peer feedback ignored (continuity break)
    # --- verification/ ---
    "AFC-0023": 2,  # verification step skipped (forward fulfillment of intent-to-verify)
    "AFC-0024": 3,  # incorrect verification (content correctness of judgment)
    "AFC-0037": 1,  # sibling-source contradiction unresolved (continuity)
    # --- termination/ ---
    "AFC-0025": 2,  # premature termination (intent not fulfilled)
    "AFC-0026": 1,  # non-termination / endless loop (continuity break — no progress edge)
    "AFC-0027": 2,  # step-limit exhausted (fulfillment failed within budget)
    "AFC-0042": 2,  # coarse-granularity termination (fulfillment incomplete)
    # --- cognitive/ ---
    "AFC-0028": 1,  # wrong-location fixation (continuity to evidence)
    "AFC-0029": 1,  # tool-output override (continuity to evidence)
    "AFC-0030": 1,  # critical-evidence miss (continuity to evidence)
    "AFC-0036": 3,  # stated-reasoning unfaithfulness (content / internal consistency)
    "AFC-0041": 3,  # rationalizing pseudo-backtrack (content / internal consistency)
    # --- system_level/ ---
    "AFC-0031": 2,  # tool execution error (action can't fulfill — borderline, kept on axis 2)
    "AFC-0032": None,  # LLM API limit (pure infra fault, not a cognitive defect)
    "AFC-0033": None,  # environment error (pure infra fault, not a cognitive defect)
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardSummary:
    """Compact view returned by ``cards_list()`` (~tool-list payload)."""

    id: str
    name: str
    axis_hint: int | None
    one_line_mechanism: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "axis_hint": self.axis_hint,
            "one_line_mechanism": self.one_line_mechanism,
        }


@dataclass(frozen=True)
class CardFull:
    """Full card content returned by ``cards_get(card_id)``."""

    id: str
    name: str
    axis_hint: int | None
    mechanism: str
    activation: dict[str, Any] = field(default_factory=dict)
    observable: dict[str, Any] = field(default_factory=dict)
    downstream_effects: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    evidence: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "axis_hint": self.axis_hint,
            "mechanism": self.mechanism,
            "activation": self.activation,
            "observable": self.observable,
            "downstream_effects": list(self.downstream_effects),
            "evidence": list(self.evidence),
        }


# ---------------------------------------------------------------------------
# Loader internals
# ---------------------------------------------------------------------------

# From src/llmharness/cards.py the scenario root is two parents up:
#   parents[0] = src/llmharness/
#   parents[1] = src/
#   parents[2] = scenarios/llmharness/   <- target
_DEFAULT_CARDS_ROOT = (
    Path(__file__).resolve().parents[2] / "references" / "papers" / "cards"
)
_MECHANISM_TRUNC = 140
_CACHE: dict[str, CardFull] | None = None
_SUMMARY_CACHE: list[CardSummary] | None = None


def _resolve_cards_root() -> Path:
    """Resolve the cards root, honoring the ``LLMHARNESS_CARDS_ROOT`` env var."""
    override = os.environ.get("LLMHARNESS_CARDS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_CARDS_ROOT


def _one_line(mechanism: str) -> str:
    """Collapse multi-line mechanism into a single sentence, truncated.

    YAML mechanisms are usually written as block-scalars with hard line wraps;
    the audit only needs an at-a-glance hint, so we take the first sentence
    (period-terminated) and trim to ``_MECHANISM_TRUNC`` chars.
    """
    flat = " ".join(mechanism.split())
    period = flat.find(". ")
    first = flat[: period + 1] if period != -1 else flat
    if len(first) > _MECHANISM_TRUNC:
        return first[: _MECHANISM_TRUNC - 1].rstrip() + "…"
    return first


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Tolerantly coerce a YAML node to a dict; missing/malformed becomes ``{}``."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_sequence(value: Any) -> tuple[dict[str, Any], ...]:
    """Tolerantly coerce a YAML node to a tuple of dicts; missing becomes ``()``."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _load_card(path: Path) -> CardFull | None:
    """Parse one YAML file. Returns None for non-card YAMLs (id missing or
    not starting with ``AFC-``)."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        return None
    card_id = raw.get("id")
    if not isinstance(card_id, str) or not card_id.startswith("AFC-"):
        return None
    name_raw = raw.get("name", "")
    name = name_raw if isinstance(name_raw, str) else str(name_raw)
    defect = _coerce_mapping(raw.get("defect"))
    mechanism_raw = defect.get("mechanism", "")
    mechanism = mechanism_raw if isinstance(mechanism_raw, str) else str(mechanism_raw)
    if card_id not in _AXIS_HINT:
        raise KeyError(
            f"Card {card_id} (file {path}) has no entry in _AXIS_HINT. "
            "Update llmharness.cards._AXIS_HINT before adding new cards."
        )
    return CardFull(
        id=card_id,
        name=name,
        axis_hint=_AXIS_HINT[card_id],
        mechanism=mechanism,
        activation=_coerce_mapping(raw.get("activation")),
        observable=_coerce_mapping(raw.get("observable")),
        downstream_effects=_coerce_sequence(raw.get("downstream_effects")),
        evidence=_coerce_sequence(raw.get("evidence")),
    )


def _load_all() -> dict[str, CardFull]:
    """Walk the cards root once and cache every card by id."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = _resolve_cards_root()
    by_id: dict[str, CardFull] = {}
    if not root.exists():
        _CACHE = by_id
        return _CACHE
    for path in sorted(root.rglob("*.yaml")):
        card = _load_card(path)
        if card is None:
            continue
        if card.id in by_id:
            raise ValueError(
                f"Duplicate card id {card.id}: {path} collides with an earlier file."
            )
        by_id[card.id] = card
    _CACHE = by_id
    return _CACHE


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def cards_list() -> list[CardSummary]:
    """Return one ``CardSummary`` per AFC card in the corpus.

    Result is cached after the first call. The audit child session calls this
    at most once per firing to scan available cards before deciding whether
    to drill into specific ones via ``cards_get``.
    """
    global _SUMMARY_CACHE
    if _SUMMARY_CACHE is not None:
        return list(_SUMMARY_CACHE)
    cards = _load_all()
    summaries = [
        CardSummary(
            id=card.id,
            name=card.name,
            axis_hint=card.axis_hint,
            one_line_mechanism=_one_line(card.mechanism),
        )
        for card in sorted(cards.values(), key=lambda c: c.id)
    ]
    _SUMMARY_CACHE = summaries
    return list(summaries)


def cards_get(card_id: str) -> CardFull:
    """Return the full ``CardFull`` for ``card_id``.

    Raises ``KeyError`` for any unknown id. There is intentionally no silent
    fallback: an audit citing a non-existent card is a bug we want to surface,
    not paper over.
    """
    cards = _load_all()
    if card_id not in cards:
        raise KeyError(f"Unknown card id: {card_id!r}")
    return cards[card_id]
