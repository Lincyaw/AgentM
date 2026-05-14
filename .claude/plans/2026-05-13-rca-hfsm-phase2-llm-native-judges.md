# Plan: rca_hfsm Phase 2 — LLM-native judges

**Date:** 2026-05-13
**Status:** proposed
**Design doc:** [../designs/llm-native-judges.md](../designs/llm-native-judges.md)
**Branch:** `feat/rca-hfsm-phase1` — stacks on top of Phase 1's open PR #154 (memory rule: no new PR while predecessor is open).

3 implementation commits + 1 docs commit. Each commit lands green on `uv run ruff check`, `uv run mypy contrib/scenarios/rca_hfsm/src/`, and `uv run pytest contrib/scenarios/rca_hfsm/tests/`. PR description updated post-C3 to reflect Phase 2 contents.

---

## Scope

**In:**
- Judge port + Protocol + types (`judges.py` pure module).
- 4 judge atoms — one per kind (`judge.satisfied`, `judge.coverage`, `judge.independence`, `judge.falsified_genuinely`). Each atom registers exactly one judge service; `config.mode: "llm" | "stub"` toggles backing implementation.
- Gate refactor: zero regex, zero structural rules; all preconditions move to judge calls.
- Downgrade-application semantics flip: gate returns `downgraded(suggested=..., applied_id=None)` and the refine is *not* auto-applied (Phase 1 surprise #1 from the smoke test).
- Eval integration: rca_hfsm runs against 10 cases from `contrib/scenarios/rca/eval/tasks/` with LLM-mode judges; results report at `contrib/scenarios/rca_hfsm/eval/phase2_results.md`.

**Out (Phase 3+):**
- `judge.next_to_verify` (information-gain scheduler replacement). Phase 1's overlap-counting heuristic is not on the regex-removal critical path.
- `judge.find_contradiction` (devil's advocate as judge).
- `judge.propose_alternatives` (hypothesis-generator isolation).
- `judge.what_can_be_dropped` (state-triggered compaction).
- Cross-session judge cache.
- Eval pass-rate optimization (Phase 2 ships baseline; tuning is downstream).

---

## C0 — Design + plan + index (docs only)

This commit lands the design doc and this plan. No code.

**Files:**
- `.claude/designs/llm-native-judges.md` (new)
- `.claude/plans/2026-05-13-rca-hfsm-phase2-llm-native-judges.md` (new)
- `.claude/index.yaml` — add `llm_native_judges` concept entry; cross-link to `hypothesis_driven_rca`.
- `decisions.md` — Phase 2 entry already appended in-session.

**Commit message:** `docs(rca-hfsm): LLM-native judges design + phase 2 plan`

---

## C1 — Judge port + 4 judge atoms

**Files added:**
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/judges.py` — pure module (NOT an atom):
  - `class Judge(Protocol)` — runtime-checkable, `kind: str`, `judge(JudgeContext) -> Verdict`.
  - `@dataclass(frozen=True) class JudgeContext` — `graph_slice: dict`, `operands: dict`.
  - `@dataclass(frozen=True) class Verdict` — `verdict: str` (free-text), `reason: str`, `confidence: str` (free-text).
  - Helpers: `make_unclear(reason: str) -> Verdict`, `canonical_cache_key(ctx: JudgeContext) -> str`.
  - Tool-use schema dataclass `JudgeToolSchema` for the LLM tool_use payload (verdict + reason + confidence as required string fields).
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/judge_satisfied.py` — §11 atom. Registers `api.set_service("rca.judge.satisfied", impl)`. Two impls inside the atom (private classes); `config.mode` switches. LLM impl uses `api.get_provider()` (or whatever the standard provider lookup is) and forces a single tool_use call returning JudgeToolSchema. Stub impl reads `config.scripted: list[dict]` and returns them in order, raising if exhausted.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/judge_coverage.py` — same shape as above; replaces mechanical chain-walk.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/judge_independence.py` — same shape; replaces `worker_session_id` literal compare.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/judge_falsified_genuinely.py` — same shape; replaces "at least one negative prediction checked" rule.
- `contrib/scenarios/rca_hfsm/prompts/judges/satisfied.md` / `coverage.md` / `independence.md` / `falsified_genuinely.md` — short focused prompts loaded by the LLM mode. Each ~30–80 words. The prompt asks the LLM to call the judgment tool and emit its verdict + reason inside the tool_use payload.

**Files NOT touched in C1:**
- `rca_falsification_gate.py` — refactored in C2.
- Existing tests — they keep passing (gate still uses Phase 1 rules until C2).
- Manifest — judge atoms added to manifest only in C3, when eval needs them.

**Tests (new):**
- `tests/test_judges_stub.py` — each judge in stub mode returns scripted verdicts in order; raises on exhaustion; caches identical contexts (second call hits cache, third with different context returns next scripted).
- `tests/test_judges_llm_contract.py` — each judge in llm mode, with a mocked provider that returns a tool_use payload, parses correctly into Verdict; malformed payload triggers retry; second malformed → unclear.
- `tests/test_judges_caching.py` — same context returns cached verdict; different context bypasses cache.

**Hard constraints:**
- Each judge atom = 1 MANIFEST + 1 install + private impl classes. No atom-to-atom imports. Can import from `judges.py` (pure module) and `schema.py`.
- LLM impl uses the existing provider lookup (whatever atoms like `agentm_rca.tools.duckdb_sql` or `llmharness` do). The implementing agent reads the codebase to find the right seam — does NOT add new core APIs.
- No regex parsing anywhere in this commit. Tool_use payload is structured JSON via the provider's API.

**LoC budget:** ~150 judges.py + ~200/judge atom × 4 = ~800 + ~250 tests. Hard cap 1200 atom+module code.

**Commit message:** `feat(rca_hfsm): judge port + 4 judge atoms (llm + stub modes)`

---

## C2 — Gate refactor: remove regex, consult judges

**Files modified:**
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_falsification_gate.py` — rewrite the precondition-checking branches of `_apply_confirm`, `_apply_refute`, `_apply_propose`, `_apply_attach_check` to consult judges via `api.get_service("rca.judge.<kind>")`. Delete:
  - The word-boundary regex on `verdict_proposal` / `effect` / `steelman` text.
  - The mechanical chain-walk in `_apply_confirm` for coverage.
  - The `worker_session_id` literal-equality independence check.
  - The structural "at least one negative prediction checked" rule.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/updates.py` — IF any precondition helpers there embed regex/structural rules, replace with calls into the same judges. Most logic should already be inlined in gate after C2; updates.py becomes pure UpdateProposal/UpdateResult types.
- `rca_falsification_gate.py` — downgrade semantics: return `UpdateResult.downgraded(suggested=<refine proposal>, applied_id=None, reason=judge.reason)`. The refine is *not* applied. The orchestrator decides.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_hgraph_store.py` — IF `get_unexplained_symptoms()` still embeds the §7.1 chain-walk, it's now a coverage-judge consumer; this method may shrink to a candidate-list (raw symptom IDs without coverage filter) and coverage moves entirely into `judge.coverage`. Implementing agent decides the cleanest split.

**Files NOT touched:**
- The 4 judge atoms (C1).
- `schema.py`, `evidence_tools.py`, `fsm_policy.py`, `finalize.py`, `worker_return.py`, `brief_builder.py` — public surfaces unchanged.

**Tests:**
- Existing tests must pass with stub-mode judges mounted via a test fixture in `_gate_fixtures.py`. The fixture scripts stub judges to mirror Phase 1's structural verdicts (e.g., for the existing `test_gate_confirm.py` cases: script `judge.falsified_genuinely → "genuine_attempt"` when the fixture set up a checked negative prediction; `→ "no_attempt"` otherwise). This is the **behavior-preservation gate**.
- New test `tests/test_gate_no_regex.py` — opens `rca_falsification_gate.py` source, asserts that none of: `import re`, `re.`, `\\b`, `.match(`, `.search(`, `"triggered"`, `"supports"`, `"steelman"` (literal strings used by Phase 1's rules) appear in the file. This is the **regex-removal gate**.
- New test `tests/test_gate_downgrade_no_apply.py` — assert that a downgraded confirm result has `applied_id=None` and the original hypothesis remains in `open` status (not flipped to `refined→...`). Verifies Phase 1 surprise #1 is fixed.

**Hard constraints:**
- §11 still passes (no new core imports).
- All Phase 1 tests still pass; no test deletions; only fixture additions allowed.
- mypy clean; ruff clean.

**LoC budget:** gate likely shrinks by 100–200 lines. Tests add ~150.

**Commit message:** `refactor(rca_hfsm): gate consults judges (zero regex)`

---

## C3 — Eval integration + 10-case run + results report

**Files added:**
- `contrib/scenarios/rca_hfsm/eval/run_phase2_eval.py` — script that:
  1. Reads 10 representative tasks from `contrib/scenarios/rca/eval/tasks/`. Implementing agent inspects the suite and picks cases that exercise variety (different `task_class` values if present, different `root_cause_keywords` shapes, different fixture sizes). If `tasks/` has fewer than 10, runs all available.
  2. For each task, runs `agentm` (or the equivalent in-process session bootstrap) against `contrib/scenarios/rca_hfsm/manifest.yaml` with LLM-mode judges and the same provider/model that the existing rca scenario uses by default (whatever `.env` / scenario config provides — no override).
  3. Captures the trajectory JSONL under `.agentm/observability/<trace_id>.jsonl`.
  4. Runs the existing rca grader (`contrib/scenarios/rca/eval/grader.md` or `grader.py` — whichever applies) on the final report.
  5. Tabulates per-case metrics: pass/fail (grader verdict), turn count, judge call count by kind, downgrade count, hypothesis count, observation count.
- `contrib/scenarios/rca_hfsm/manifest.yaml` — updated to mount the 4 judge atoms in `llm` mode by default. Stub mode remains available via override for tests.
- `contrib/scenarios/rca_hfsm/eval/phase2_results.md` — the report. Format:
  - **Summary** (top): N/10 pass, total cost (if extractable from trajectory), broad observations.
  - **Per-case** sections: case ID, prompt summary, verdict (pass/fail), notable trajectory events (downgrades, refines, hypothesis count), one-line qualitative note ("the falsified_genuinely judge correctly flagged a perfunctory negative check"), 3–6 lines per case.
  - **Cross-case patterns** (bottom): things to feed into Phase 3 tuning — judges that frequently say "unclear", downgrade patterns, FSM states where the orchestrator got stuck.

**Files NOT touched:**
- Anything in `contrib/scenarios/rca/` — that scenario is the eval source, never modified.

**Hard constraints:**
- The 10-case run consumes real LLM tokens. Implementing agent runs with whatever model the user has configured; does NOT override with a cheaper/different one without a logged decision.
- Trajectory inspection follows the CLAUDE.md E2E methodology: drive through `agentm` CLI (or equivalent), inspect JSONL, never assert on `session._tools` internals.
- Report is honest. If pass rate is low (e.g., 2/10), the report says so and the cross-case patterns section gets longer. Hiding bad results violates the "Surface problems early" principle.

**Tests:**
- The eval script is itself a test; no separate pytest test is required for this commit. If the implementing agent finds a way to unit-test case-selection logic cheaply, they may; otherwise skip.

**LoC budget:** eval script ~250, report length depends on results.

**Commit message:** `eval(rca_hfsm): phase 2 results on 10 rca eval cases`

---

## Acceptance gate for Phase 2

1. **C1 + C2 + C3 land in order**, each green on ruff + mypy + pytest.
2. **`test_gate_no_regex.py` passes** — gate file contains no regex / lemma-matching strings.
3. **All Phase 1 fail-stop tests still pass** with stub judges mounted (behavior preservation).
4. **All 4 judges have both `stub` and `llm` modes**, each tested.
5. **Eval report exists** at `contrib/scenarios/rca_hfsm/eval/phase2_results.md` with all 10 cases reported on. The pass rate is reported truthfully; this plan does NOT specify a minimum target.
6. **PR #154 description updated** post-C3 to describe Phase 2 inclusions. No new PR.
7. **`.claude/index.yaml`** has the `llm_native_judges` concept entry.
8. **`decisions.md`** has the Phase 2 entry (already appended in-session before C0).
9. **Existing `contrib/scenarios/rca/`** untouched.

If any acceptance fails, the implementing agent stops and reports. Phase 2 is correct iff it's a clean refactor + a clean measurement.

---

## What this enables for Phase 3

After Phase 2, Phase 3 work is incremental:

- Add more judges (`next_to_verify`, `propose_alternatives`, `find_contradiction`, `what_can_be_dropped`) — same Protocol, same pattern.
- Run the tuner from `per_task_evolution_loop` against each judge's prompt — its eval set per judge is whichever cases stress that judge.
- A/B individual judges across `llm` modes by varying just one judge's prompt; the per-judge isolation makes the credit-assignment problem tractable.

None of these need framework changes. Phase 2 lays the substrate; Phase 3 fills it in with data.
