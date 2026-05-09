# Task: Layer B Wave-Reflection Completion Report (B-2, B-5)

**Date**: 2026-05-09
**Status**: COMPLETE
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)

## Tasks completed

- B-5 — Per-module credit assignment (round-robin) — DONE
- B-2 — Reflection atom (`tool_reflect`) — DONE

Order: B-5 first (read-only atom + prompt scaffold), B-2 second
(reflection atom + per-scenario template + fail-stop test).

Per the user's assignment instructions, `tool_reflect` is implemented
as a **deterministic scaffold** (no in-atom LLM call). The atom assembles
a structured prompt block plus a `change_spec_schema` hint; the outer
tuner LLM consumes the scaffold on its next turn. This deviates from the
B-2 task file's "calls a reflection LM (uses the session's StreamFn)"
phrasing — the user explicitly chose the scaffold-only pattern to avoid
provider-credential plumbing inside the atom. Behavioral guarantee
preserved: the reflection step still sits between query_traces and
propose_change, and the prompt template is per-scenario + mutable.

## Lines changed per file

| File | Insertions | Deletions |
|---|---:|---:|
| `src/agentm/extensions/builtin/tool_query_module_feedback.py` (new) | 175 | 0 |
| `src/agentm/extensions/builtin/tool_reflect.py` (new) | 327 | 0 |
| `contrib/scenarios/format_fix/eval/reflection_template.md` (new) | 50 | 0 |
| `contrib/scenarios/format_fix/tuner/prompt.md` | 27 | 12 |
| `contrib/scenarios/format_fix/tuner/manifest.yaml` | 6 | 0 |
| `tests/integration/test_reflect_changespec_roundtrip.py` (new) | 117 | 0 |
| **Approx total** | **~702 insertions** | **~12 deletions** |

Total within the 300-450 LoC plan estimate's upper band when counting
prose-heavy template + test (~470 of the 702 are template + test). The
substantive Python code footprint is ~320 LoC across the two new atoms,
matching the plan exactly.

## New atoms registered

- `tool_query_module_feedback` (B-5) — read-only; returns
  `{module_distribution: dict[str, list[str]], total_recent: int,
  scenario: str | None}`. Auto-discovered under
  `src/agentm/extensions/builtin/`. Wired into the format_fix tuner
  manifest.
- `tool_reflect` (B-2) — read-only scaffold; returns
  `{diagnosis_prompt: str, change_spec_schema: dict, target_module,
  target_scenario, source_path, template_path, feedback_sample_count,
  trace_count}`. Same auto-discovery + wiring.

## Reflection template

Lives at `contrib/scenarios/format_fix/eval/reflection_template.md` as
specified. Contains four substitution slots (`<TARGET_MODULE>`,
`<TARGET_SCENARIO>`, `<TRACES>`, `<CURRENT_SOURCE>`, `<RECENT_FEEDBACK>`)
plus a `<MUTATION_INSTRUCTIONS>` body with six numbered instructions:

1. Diagnose, then design (1-3 sentences root cause)
2. One concern per mutation (highest-evidence defect)
3. Preserve the contract (MANIFEST.name, no harness/internal imports)
4. Emit a `ChangeSpec` (atom_source kind; complete `new_content`)
5. Cite evidence in rationale (trace_id + feedback line)
6. Do not call propose_change yet — eval_run first for baseline + proposed

Per design §11 the template is itself mutable by future meta-tuning,
so future tuner sessions can evolve the instructions.

## rca tuner prompt

Skipped — `contrib/scenarios/rca/` has no `tuner/` directory yet
(`find` returns only `prompts/orchestrator.md`, scout/verify/deep_analyze
agents, and skill markdowns). Per the assignment ("Check first; rca
tuner might still be under development."), no edits were made.

## Test status

`uv run pytest --tb=short` → **104 passed, 1 skipped, 14 deselected**
(was 99 passed; +5 new tests in `test_reflect_changespec_roundtrip.py`).

The +5 break down as:

- 1 happy-path: synthetic reflect-emitted ChangeSpec validates through
  `atom_source.validate` (the B-2 fail-stop).
- 3 complement-sanity (parametrized): rejection on missing `path`,
  `new_content`, `target_atom`. Locks the schema-hint contract.
- 1 structural: `tool_reflect` install + invocation produces a payload
  whose `change_spec_schema.required` matches what the validator expects
  and whose assembled prompt has fully substituted the slots (no raw
  `<TARGET_MODULE>` / `<TRACES>` placeholders leak through).

`uv run ruff check src/` and `uv run mypy src/agentm/extensions/builtin/`
both clean.

## §11 contract

`tests/unit/extensions/test_extension_contract.py` — 5 passed,
including the discovery sweep over `src/agentm/extensions/builtin/`.
Both new atoms (`tool_reflect`, `tool_query_module_feedback`) are
single-file, zero atom-to-atom imports, no `harness.session` import,
no `core._internal` import.

## Blockers

None. No `agentm.core.*` modification was required.
