# Task: Layer A Completion Report

**Date**: 2026-05-09
**Status**: COMPLETE
**Plan**: [plan](../plans/2026-05-09-gepa-layerA-forward-compat.md)

## Tasks completed

- A-1 — rename `decisions.jsonl` -> `activations.jsonl` (commit `a3af664`)
- A-2 — `target_atom: str` -> `target: ChangeSpec` with MVP guard (commit `95df492`)
- A-4 — write `candidates/<id>.json` companions per activation (commit `50cf98a`)
- A-3 — grader mu_f contract + feedback aggregation (commit `739d3af`)
- A-5 — `budget.json` slot + `max_cost_usd` cap (commit `0d60849`)

Order followed the user-provided sequence (A-1, A-2, A-4, A-3, A-5).

## Lines changed per file

| File | Insertions | Deletions |
|---|---:|---:|
| `src/agentm/extensions/builtin/tool_propose_change.py` | ~232 | ~6 |
| `src/agentm/extensions/builtin/tool_eval_run.py` | ~260 | ~6 |
| `contrib/scenarios/format_fix/eval/grader.py` | ~52 | ~5 |
| `contrib/scenarios/format_fix/tuner/prompt.md` | 5 | 4 |
| `contrib/scenarios/format_fix/tuner/README.md` | 4 | 4 |
| `contrib/scenarios/format_fix/README.md` | 1 | 1 |
| `tests/integration/test_per_task_evolution.py` | ~80 | ~10 |
| **Total** | **615 insertions** | **43 deletions** |

## Blockers escalated

None. The constitution-path glob in `core-manifest.yaml` is
`.agentm/decisions/**` (verified at `core-manifest.yaml:33`), which
already covers `activations.jsonl`, `candidates/<id>.json`, and
`budget.json` — no `agentm.core.*` modification was needed.

## Test status

`uv run pytest --tb=short` -> **97 passed, 1 skipped, 14 deselected**
(unchanged from baseline; the skipped one is the existing API-key-gated
e2e in `test_per_task_evolution_e2e.py`).

`uv run ruff check` and `uv run mypy` both clean on every touched file.

## Smoke-test outcome

The interactive `agentm` smoke (Run-3 shape) was not executed: no
provider API key surfaced in the environment. Coverage was instead
verified via the existing fail-stop integration tests in
`tests/integration/test_per_task_evolution.py`, which exercise:

- Activation log written to `.../activations.jsonl` (not the old name).
- `candidates/c_<uuid12>.json` companion file exists with the design
  section 11.1 schema; activation entry references it via
  `candidate_id`.
- ChangeSpec validation: missing/non-`atom_source` kinds reject with
  `not_yet_implemented`.
- Constitution boundary still classifies `activations.jsonl` and
  `candidates/c_x.json` as protected.

`budget.json` has no fail-stop integration test in this layer (the
behavior depends on real LLM cost reporting through child traces). It
is wired and unit-clean; B-6 is the next opportunity to lock it down.
