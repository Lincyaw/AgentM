# Task: cards.py module with axis_hint mapping

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-cognitive-audit-v0.md)
**Design**: [design](../designs/llmharness-cognitive-audit.md)
**Assignee**: implementer

## Objective

Create `scenarios/llmharness/src/llmharness/cards.py` exposing two
public functions used as AgentM tools by the diagnostic scenario, plus
a curated in-code `_AXIS_HINT` mapping. Cards live as YAML at
`scenarios/llmharness/references/papers/cards/<class>/*.yaml`.

## Inputs

- Read: design §4.4 "Cards as tools" (signatures, axis_hint rationale,
  ~2000 token budget for `cards_list`).
- Read: design §6.3 + §11 — `axis_hint` MUST live in Python code, NOT
  YAML schema (preserves rca-autorl public contract on cards).
- Read: directory listing of
  `scenarios/llmharness/references/papers/cards/` to enumerate every
  card_id present today (~42 cards across `memory/`, `reflection/`,
  `planning/`, `action/`, `specification/`, `inter_agent/`,
  `verification/`, `termination/`, `cognitive/`, `system_level/`).
- Read one example card YAML (e.g.
  `references/papers/cards/specification/goal-drift.yaml`) to confirm
  field shape (`id`, `name`, `defect.mechanism`, etc).

## Outputs

- New: `scenarios/llmharness/src/llmharness/cards.py` containing:
  - Frozen dataclasses `CardSummary(id, name, axis_hint, one_line_mechanism)`
    and `CardFull(id, name, axis_hint, mechanism, activation, observable,
    downstream_effects, evidence)` with `to_dict` for tool serialization.
  - `def cards_list() -> list[CardSummary]` (cached).
  - `def cards_get(card_id: str) -> CardFull` (raises `KeyError` on
    unknown id; never auto-falls-back).
  - Module-level `_AXIS_HINT: dict[str, int | None]` covering every
    AFC ID present in the cards directory. Use design §3.1–§3.3
    "Cards covered" lists as the seed; classify any remaining card
    pragmatically (axis 1 = continuity, axis 2 = fulfillment, axis 3
    = content; `None` for genuinely unclassified — but `None` should
    be rare).
  - YAML loader using `pyyaml` (already a transitive dep via existing
    code; confirm in `pyproject.toml`); discover cards by globbing
    `_CARDS_ROOT.rglob("*.yaml")` excluding non-card files (skip if
    YAML lacks `id` field starting with `AFC-`).
  - `_CARDS_ROOT` resolution: prefer
    `Path(__file__).resolve().parents[3] / "references" / "papers" / "cards"`
    (4 levels up from `src/llmharness/cards.py`); allow override via
    `LLMHARNESS_CARDS_ROOT` env var for test fixtures.
- Modified: `scenarios/llmharness/project-index.yaml` — add REQ-016
  (`title: "cards.py loader + cards_list/cards_get tools + axis_hint"`,
  `priority: P0`, `status: implemented`,
  `code: [src/llmharness/cards.py]`, `depends_on: [REQ-001]`).

## Acceptance Conditions

- [ ] `cards_list()` returns one summary per AFC YAML in the cards
      directory; result is cached after first call
- [ ] `cards_get("<known-id>")` returns a `CardFull` whose
      `mechanism` matches the YAML
- [ ] `cards_get("AFC-9999")` raises `KeyError`
- [ ] Every id returned by `cards_list()` is a key in `_AXIS_HINT`
      (no silent `None` for missing cards)
- [ ] `uv run mypy src/llmharness/cards.py` strict-clean
- [ ] `uv run ruff check src/llmharness/cards.py` clean

## Notes

- `one_line_mechanism` = first sentence of YAML's `defect.mechanism`,
  truncated to ~140 chars. The design budgets ~2000 tokens for the
  whole `cards_list` payload.
- Do NOT add `axis_hint` to YAML files — that would break the
  rca-autorl public contract on cards (design §4.4).
- Do NOT load YAMLs at import time; lazy-load on first call so import
  is cheap and tests can override `_CARDS_ROOT` first.
