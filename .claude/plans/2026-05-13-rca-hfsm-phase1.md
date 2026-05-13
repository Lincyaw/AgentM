# Plan: rca_hfsm Phase 1 — Falsification-gated RCA scenario MVP

**Date:** 2026-05-13
**Status:** proposed
**Design doc:** [../designs/hypothesis-driven-rca.md](../designs/hypothesis-driven-rca.md)
**Implementation target:** `contrib/scenarios/rca_hfsm/` (new workspace member, sibling to existing `contrib/scenarios/rca/` — does not replace it).

5 commits on a new branch `feat/rca-hfsm-phase1`. Each commit ends green on
`uv run ruff check`, `uv run mypy src/ contrib/scenarios/rca_hfsm/src/`, and
`uv run pytest --tb=short` (scenario-local tests run from inside
`contrib/scenarios/rca_hfsm/`; root pytest config already excludes nested
workspaces).

Commits are sequenced strictly: commit 1 lands the schema + store with the
single-writer property; commits 2–5 stack additively. Do not open new
branches before commit 5 lands.

---

## Scope of Phase 1 (and what is deliberately out)

**In (design §12 Phase 1 MVP):**

- §3 HypothesisGraph schema + store atom
- §6 two-column `WorkerReturn` contract (`observations` vs `interpretation`)
- §6.2 negative-prediction-required precondition on `propose(H)`
- §7.1 confirm gate (falsification + independent positive + coverage)
- §7.2 refute gate (steelman attempt OR triggered negative prediction)
- §4 FSM transitions with the default `informational` verification scheduler (§8)
- §5.1–§5.3 layered context (L1 store + L2 orchestrator + L3 workers)
- Acceptance scenarios 1–7 from design §11

**Out (Phase 2+ — separate plans):**

- §5.5 state-triggered compaction
- §9 independent-corroboration mechanisms (twin verifier, devil's advocate, tool-heterogeneous workers, isolated hypothesis_generator)
- §10 bias telemetry
- L2 sub-linear growth measurement (acceptance #8) — needs trace data from Phase 1 first
- Brief-disjointness and generator-blinding property tests (acceptance #9, #10)
- Cross-trace skill learning, multi-incident memory, concurrent verification

The L2-growth claim is a load-bearing premise of the design. Phase 1 lands the structure that should make it true; Phase 2 measures whether it actually does on the eval suite. If L2 grows linearly in tool calls anyway, the design needs revisiting before Phase 2 ships compaction.

---

## Commit 1 — Scenario skeleton + L1 schema + hgraph store

**Scope.** Create the new workspace member with the data model and storage atom. No FSM behavior yet; the store is unused but importable and tested.

**Files added.**

- `contrib/scenarios/rca_hfsm/pyproject.toml` — workspace member declaration; depends on `agentm` from the workspace root.
- `contrib/scenarios/rca_hfsm/README.md` — one-screen orientation, points to the design doc.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/__init__.py`
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/schema.py` — dataclasses for `Symptom`, `Prediction`, `Hypothesis`, `CheckResult`, `Observation`, `Interpretation`, `WorkerReturn` (per design §3.1, §3.2, §6). Free-text fields where the design specifies (`status`, `polarity` is the one fixed enum; `verdict_proposal`/`confidence`/`proposed_update` are free-text per CLAUDE.md "no preset enums for subjective dimensions").
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/__init__.py`
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_hgraph_store.py` — single-file §11 atom. `MANIFEST` + `install(api, config)`. Publishes two services:
  - `api.set_service('rca.hgraph.read', read_handle)` — public read API: `get_symptoms()`, `get_hypothesis(id)`, `get_open_leaves()`, `get_unexplained_symptoms()`, `get_refuted_branches()`, `get_observation_by_signature(sig)`.
  - The write handle is **not** published as a service. Instead, the atom exposes a module-level `claim_write_handle(token: str)` function. The gate atom (commit 2) acquires it once at install time using a shared-secret token published by this atom via `api.set_service('rca.hgraph.write_token', token)`. Any other atom calling `claim_write_handle` after the gate is rejected with `RuntimeError`.
- Root `pyproject.toml` — add `contrib/scenarios/rca_hfsm/` to workspace members.

**Files NOT touched.** No changes to `src/agentm/core/**`. No new ExtensionAPI surface in core (per design §2).

**Test strategy (fail-stop only).**

- `tests/test_store_single_writer.py` — acceptance #6: after a first `claim_write_handle` call succeeds, a second call from any caller raises. Read API remains available to anyone via `api.get_service`. (Covers the "graph writes are mediated by exactly one atom" property.)
- `tests/test_schema_roundtrip.py` — schema dataclasses serialize/deserialize through `dataclasses.asdict` cleanly (needed for observability JSONL). Single fail-stop test, not exhaustive field coverage.

**DoD.**

- [ ] `uv sync` resolves the new workspace member.
- [ ] `uv run ruff check contrib/scenarios/rca_hfsm/src/` clean.
- [ ] `uv run mypy contrib/scenarios/rca_hfsm/src/` clean (the scenario must mypy-clean from day one — store types are load-bearing).
- [ ] `uv run pytest contrib/scenarios/rca_hfsm/tests/` green; both fail-stop tests present.

---

## Commit 2 — Falsification gate + update operators

**Scope.** Land the gate atom with all update operators from §3.3 and the confirm/refute preconditions from §7.1/§7.2. Downgrade-to-refine semantics (a failing confirm becomes `refine(H, "needs <missing piece>")` rather than raising) per design §7.1.

**Files added.**

- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_falsification_gate.py` — single-file §11 atom. Acquires the write handle from the store at install time. Exposes one service: `api.set_service('rca.gate', gate)` with method `gate.apply(update: UpdateProposal) -> UpdateResult`. `UpdateResult` carries either `applied` with the resulting node ID, `downgraded` with the new operator that ran instead, or `rejected` with reason text.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/updates.py` — `UpdateProposal`, `UpdateResult` types and the structural precondition functions (pure functions on the graph, no I/O). Kept separate from the gate atom so they're unit-testable without going through the atom install path.

**Files NOT touched.** Store atom from commit 1 is read-only from outside this atom; no changes there.

**Test strategy (fail-stop only).**

- `tests/test_gate_propose.py` — acceptance #1: `propose(H)` with zero negative predictions is rejected. The rejection carries a precise reason string (`"hypothesis must declare at least one negative prediction"`) the LLM can act on.
- `tests/test_gate_confirm.py` — acceptance #2: `confirm(H)` is downgraded to `refine` when (a) no negative prediction has been checked, (b) only one worker has produced positive checks (independence requirement fails), or (c) unexplained symptoms remain. Three sub-cases; each asserts the downgrade-with-reason result rather than a raise.
- `tests/test_gate_refute.py` — acceptance #3: `refute(H)` is downgraded when neither a triggered negative prediction nor a `mode="steelman"` check exists.
- `tests/test_gate_independence.py` — the "independent worker" check (§7.1) compares `worker_session_id`s and rejects when both supporting checks share an id. Brief-slice disjointness is NOT enforced here (that's acceptance #9, Phase 2).

**Public-contract impact.** None — gate is the first writer of the graph, no prior consumers exist.

**DoD.**

- [ ] mypy/ruff clean on the new files.
- [ ] Four new fail-stop tests green.
- [ ] No atom-to-atom imports (§11): gate imports types from `schema.py` and `updates.py` (pure modules, allowed) but does not import the store atom module — it reaches the store only through the `api.get_service('rca.hgraph.write_token')` channel.

---

## Commit 3 — Evidence tools + ObservationLog memoization

**Scope.** Register the user-facing tool surface (`record_symptom`, `record_observation`, `propose_hypothesis`, `attach_check`, `propose_update`) routing through the gate. Add the idempotency-keyed memoization wrapper for read-only tools.

**Files added.**

- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_evidence_tools.py` — registers the five tools above. Each tool builds an `UpdateProposal`, calls `gate.apply`, and surfaces the `UpdateResult` back to the LLM as `ToolResult.text`. Reasons from downgrades/rejections are passed through verbatim so the LLM gets actionable feedback.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_observation_cache.py` — single-file atom that wraps idempotent tool registrations with an ObservationLog-lookup shim. Reads `idempotent: bool` from the tool's registration metadata; for `idempotent=True` tools it computes `tool_signature = sha256(tool_name + canonical_json(args))` and checks the store before executing. On hit, emits `tool_call_cached` diagnostic to the bus and returns the cached observation as the tool result.

**Files NOT touched.** The existing `duckdb_sql` atom from `contrib/scenarios/rca/` will need an `idempotent=True` flag at registration to opt into memoization. That change is **not** in this commit (touching another scenario's atoms is out of scope for Phase 1); it's a follow-up after Phase 1 lands. For commit 3 tests, use a stub tool registered with `idempotent=True`.

**Test strategy (fail-stop only).**

- `tests/test_evidence_tool_routing.py` — every evidence tool produces an `UpdateProposal` that flows through the gate; a downgrade from the gate is visible in the tool result. One end-to-end test per tool (5 sub-cases).
- `tests/test_observation_memoization.py` — acceptance #5: register a stub `idempotent=True` tool that increments a counter on each real call; invoke it twice with identical args; assert the second call returns the cached observation and the counter is 1. Then call with different args and assert the counter advances to 2.
- `tests/test_observation_no_cache_for_side_effects.py` — same setup but `idempotent=False`; counter advances on every call.

**DoD.**

- [ ] mypy/ruff clean.
- [ ] Three new fail-stop tests green.
- [ ] `tool_call_cached` event appears in the observability JSONL when the cache hits (verified via a `EventBusObserver` in the test).

---

## Commit 4 — FSM policy + brief builder + L2/L3 wiring + FINALIZE guard

**Scope.** The behavioral spine. Adds state-aware system prompt assembly, the information-gain verification scheduler (§8 default), worker brief construction (§5.4), worker-return ingestion that separates observations from interpretation (§6), and the FINALIZE coverage guard.

**Files added.**

- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_fsm_policy.py` — single-file atom. Subscribes to graph mutation events (emitted by the gate atom from commit 2 — gate emits `rca.graph.mutated` after each successful apply); maintains an FSM state attached to the trace; injects state-specific prompt fragments via `api.prompt_templates`; filters the visible tool set per state via `ToolFilterEvent`.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_brief_builder.py` — pure-function brief construction from a `(hypothesis_id, prediction_id, mode)` triple. Falsification-framed wording for `mode="verify"`; hypothesis blinding (strips parent claim from the brief) is on by default; non-overlap of slices is NOT yet enforced (Phase 2).
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/atoms/rca_finalize.py` — registers `submit_final_report`; the tool is the only legal way to leave `FINALIZE` state. Coverage check runs before accepting: every symptom must be linked through `Observation.related_symptoms` to a satisfied prediction of a `confirmed` hypothesis.
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/scheduler.py` — pure functions implementing the `informational` scheduler (overlap-counting approximation per §8).
- `contrib/scenarios/rca_hfsm/src/agentm_rca_hfsm/worker_return.py` — pure helpers: parse a worker session's structured output into `WorkerReturn`; ingest a `WorkerReturn` into L1 by appending `observations` to the log via the gate's `record_observation` operator. `interpretation` is recorded as a trace event but does NOT enter the graph.
- `contrib/scenarios/rca_hfsm/prompts/{intake,observe,hypothesize,verify,judge,finalize}.md` — one prompt fragment per state. Each fragment is the additive content appended by `rca_fsm_policy` to the system prompt when the trace is in that state.

**Files NOT touched.** Sub-agent dispatch lives in the existing `agentm.extensions.builtin.sub_agent` atom — reused as-is. The new scenario's manifest (commit 5) wires the investigator persona into it.

**Test strategy (fail-stop only).**

- `tests/test_fsm_transitions.py` — drive a minimal trace through INTAKE → OBSERVE → HYPOTHESIZE → VERIFY → JUDGE → FINALIZE using direct tool calls; assert each state's visible tool set matches the policy's filter declaration.
- `tests/test_brief_builder.py` — `mode="verify"` brief contains the falsification phrase and omits the parent hypothesis claim. Property-style assertion, no example strings hardcoded beyond the falsification keyword.
- `tests/test_worker_return_separation.py` — acceptance #4: ingest a `WorkerReturn` twice — once with full `interpretation`, once with `interpretation` blanked — and assert the resulting `HypothesisGraph` is equal. This is the structural test that the orchestrator's re-derivation only depends on `observations`.
- `tests/test_finalize_coverage.py` — acceptance #7: `submit_final_report` is rejected when at least one symptom is unexplained; the rejection reason lists the unexplained symptom IDs.
- `tests/test_scheduler_information_gain.py` — given three open hypotheses with overlapping predictions, the scheduler picks the prediction with the highest discrimination potential. One fail-stop test; the scheduler is otherwise an approximation, so exact behavior isn't pinned.

**Public-contract impact.** First commit where the scenario actually does something end-to-end at the atom level.

**DoD.**

- [ ] mypy/ruff clean.
- [ ] Five new fail-stop tests green.
- [ ] No imports of `agentm.core.runtime.*` from any atom file (the §11 validator catches this; CI must run the validator on the new scenario).

---

## Commit 5 — Manifest + investigator persona + stub-provider smoke test

**Scope.** Compose the atoms into a runnable scenario, add the orchestrator persona, and add one end-to-end smoke test using `agentm`'s stub provider so the wiring is exercised without LLM API calls.

**Files added.**

- `contrib/scenarios/rca_hfsm/manifest.yaml` — full atom stack:
  - `agentm.extensions.builtin.operations_local`
  - `agentm_rca_hfsm.atoms.rca_hgraph_store`
  - `agentm_rca_hfsm.atoms.rca_falsification_gate`
  - `agentm_rca_hfsm.atoms.rca_evidence_tools`
  - `agentm_rca_hfsm.atoms.rca_observation_cache`
  - `agentm_rca_hfsm.atoms.rca_fsm_policy`
  - `agentm_rca_hfsm.atoms.rca_brief_builder`
  - `agentm_rca_hfsm.atoms.rca_finalize`
  - `agentm.extensions.builtin.observability`
  - `agentm.extensions.builtin.otel_tracing`
  - `agentm.extensions.builtin.sub_agent` (configured for `investigator` persona)
  - Reused from `rca/`: `agentm_rca.tools.duckdb_sql` (registered with `idempotent=true` once a follow-up issue patches the registration; for now, registered with default `idempotent=false` and the cache simply doesn't kick in)
- `contrib/scenarios/rca_hfsm/agents/investigator/SKILL.md` — persona file. System prompt explains the scientific-method framing in two paragraphs; the per-state prompt fragments do the operational work.
- `contrib/scenarios/rca_hfsm/agents/investigator/agent.yaml` — frontmatter declaring inherited extensions (mirrors the existing `rca/` critic config).
- `contrib/scenarios/rca_hfsm/tests/test_smoke_stub_provider.py` — integration test using `agentm`'s stub provider with a scripted response sequence that:
  1. Records a symptom via `record_symptom`.
  2. Proposes one hypothesis with one positive + one negative prediction.
  3. Verifies the negative prediction (with no observations triggering it).
  4. Verifies the positive prediction (with a confirming observation).
  5. Attempts `confirm` — gate downgrades because only one worker session has produced checks (independence fails).
  6. A second scripted worker session verifies the positive prediction independently.
  7. `confirm` accepted; coverage check passes; `submit_final_report` succeeds.

  Asserts on the JSONL trajectory: presence of `rca.graph.mutated` events, presence of one downgrade event, and a clean FINALIZE.

**Files NOT touched.** No changes to the existing `rca/` scenario. Eval sharing with `rca/eval/` is **not** wired in commit 5 (no scenario-aware eval-suite-reference mechanism exists yet); that's a follow-up plan once Phase 1 is stable.

**Test strategy (fail-stop only).**

- The smoke test is the one new test. It is not "an extra E2E test" — it is the wiring-correctness fail-stop. The structure (gate → store → events → trace) is the AgentM-value-prop position for this scenario.

**DoD.**

- [ ] `uv run agentm --cwd <sandbox> --scenario rca_hfsm "<NL prompt>"` runs to termination on at least one trivial prompt against a stub provider (manual check before merge; documented in PR description).
- [ ] mypy/ruff clean across the full scenario.
- [ ] All Phase 1 fail-stop tests (acceptance #1, #2, #3, #4, #5, #6, #7 plus the smoke test) green.
- [ ] `.claude/index.yaml` entry for `hypothesis_driven_rca` updated: `status: proposed → status: phase_1_landed`, and add `plans: ["plans/2026-05-13-rca-hfsm-phase1.md"]`.

---

## Out of scope (forever or Phase 2+)

- §5.5 state-triggered compaction.
- §9 twin verifier / devil's advocate / tool-heterogeneous workers / isolated hypothesis_generator.
- §10 bias telemetry.
- Brief-slice disjointness enforcement (acceptance #9).
- Generator-blinding property test (acceptance #10).
- L2-growth sub-linearity test (acceptance #8) — requires trace data from Phase 1 on the eval suite.
- Patching `agentm_rca.tools.duckdb_sql` to declare `idempotent=true`. Touches another scenario; do separately.
- Shared eval-suite reference mechanism. Each scenario currently carries its own `eval/`. A common-eval primitive is its own design question, not blocking Phase 1.
- Renaming `rca_hfsm` if a better name surfaces. Cheap to do before merge; not part of the plan.

---

## Phase 1 acceptance (gate to merge)

1. **Five commits land in order**, each green on ruff + mypy + pytest, stacked on `feat/rca-hfsm-phase1`.
2. **All seven acceptance scenarios from design §11 (#1–#7) have a corresponding fail-stop test**, all green.
3. **Smoke test in commit 5 runs end-to-end via the stub provider** with a scripted falsification flow that exercises one downgrade and one successful confirm.
4. **§11 atom contract validator passes** on every new atom file (no `core.runtime.*` imports, no atom-to-atom imports, one MANIFEST + one install per file).
5. **No changes to `src/agentm/core/**`**. No new ExtensionAPI surface in core. No new bus event types in core (gate emits a scenario-scoped `rca.graph.mutated` event but registers it via the existing typed-event registration path — does not modify core's event registry).
6. **Index updated**: `hypothesis_driven_rca` status set to `phase_1_landed`, this plan referenced.
7. **The existing `contrib/scenarios/rca/` scenario still passes its existing tests** (sanity check that nothing leaked across).

If a commit needs more than ~400 LoC of atom code (excluding tests/schema), stop and check — that's a smell that the atom is doing too much and should split. The estimated per-commit weight is:

| Commit | LoC (atom code) | LoC (tests) |
|---|---|---|
| 1 | ~150 | ~80 |
| 2 | ~250 | ~200 |
| 3 | ~200 | ~150 |
| 4 | ~400 | ~250 |
| 5 | ~120 (manifest + persona) | ~150 (smoke) |

Numbers are budget, not target. Under-shooting is fine.
