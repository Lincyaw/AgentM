# Design: Hypothesis-Driven RCA

**Status**: PROPOSED
**Created**: 2026-05-13
**Builds on**: [pluggable-architecture.md](pluggable-architecture.md), [extension-as-scenario.md](extension-as-scenario.md), [sub-agent-lifecycle.md](sub-agent-lifecycle.md), [observability.md](observability.md), [llmharness-cognitive-audit.md](llmharness-cognitive-audit.md), [per-task-evolution-loop.md](per-task-evolution-loop.md)
**Implementation target**: a new scenario under `contrib/scenarios/rca_hfsm/` (provisional name). Does **not** replace the existing `contrib/scenarios/rca/` — coexists as a methodologically more disciplined sibling so the two can be A/B'd on the same eval set.

---

## 1. First Principle

> **Root-cause analysis is process-of-elimination: propose hypotheses, predict observables, gather evidence, eliminate the ones contradicted by evidence, accept the one that explains every recorded symptom and has at least one credible refutation attempt that failed.**

Three commitments follow, and they are structural — not prompt-level guidance:

1. **Hypothesis lifecycle is a DAG of update operators, not a confirm/refute binary.** A hypothesis can be refined, split, merged, or superseded; verifications produce update operations on the graph rather than terminal verdicts.
2. **Falsification discipline is enforced by the FSM, not asked for by the prompt.** A hypothesis cannot transition to `confirmed` without ≥1 satisfied negative prediction and ≥1 independent positive verification; cannot transition to `refuted` without a steelman attempt that failed.
3. **Semantic structure (FSM) is orthogonal to context structure (sessions).** The FSM says what kind of step this is; session topology decides which LLM conversation does the work with what context. The same FSM runs under monolithic-session, per-state-session, or layered-session topologies — the scenario picks one as a separate decision.

Everything below is the concrete shape these three commitments take in AgentM.

---

## 2. Layer Position

Pure autonomy layer. No constitution changes, no new ExtensionAPI surface beyond what `pluggable_architecture` and `sub_agent_lifecycle` already provide.

```
constitution    (unchanged)
                core + reload primitive + ResourceWriter + observability sink
                + sub_agent dispatch infrastructure
   ↑
autonomy        atoms (NEW):
                  rca_hgraph_store        — Session-state axis
                  rca_evidence_tools      — Tool-environment axis
                  rca_fsm_policy          — Policy axis
                  rca_brief_builder       — Policy axis
                  rca_falsification_gate  — Policy axis
                  rca_bias_audit          — Policy + observability
                scenario (NEW):
                  contrib/scenarios/rca_hfsm/manifest.yaml
                  contrib/scenarios/rca_hfsm/agents/{investigator,critic,devils_advocate}/
                  contrib/scenarios/rca_hfsm/prompts/{intake,observe,hypothesize,verify,judge,finalize}.md
                  contrib/scenarios/rca_hfsm/eval/ — see per_task_evolution_loop §3
   ↑
discovery       fs scan + reload (unchanged)
```

All atoms are single-file §11-compliant. No atom-to-atom imports; cross-atom communication exclusively through `ExtensionAPI` services and bus events.

---

## 3. The Hypothesis Graph (L1 state)

The persistent, trace-scoped data structure that every session — orchestrator, worker, critic — reads and (controlled by the FSM gate) mutates. Owned by atom `rca_hgraph_store`, exposed as `api.set_service('rca.hgraph', store)`.

### 3.1 Node types

```python
# Conceptual schema — actual types live in the atom; no preset enums on
# subjective fields (per CLAUDE.md feedback rule).

@dataclass
class Symptom:
    id: str
    text: str                   # free-form: "disk fills at 14:32 UTC"
    source: str                 # tool_call_id | "user_intake"
    ts: float

@dataclass
class Prediction:
    id: str
    hypothesis_id: str
    claim: str                  # "if H true, we expect to observe X"
    polarity: Literal["positive", "negative"]
    test_plan: str | None       # free-form sketch; filled by Hypothesize step
    checks: list[CheckResult]   # appended by Verify step

@dataclass
class CheckResult:
    id: str
    prediction_id: str
    worker_session_id: str      # which L3 worker produced this
    observations: list[Observation]   # raw facts, see §3.2
    interpretation: Interpretation    # advisory, see §6
    verdict_proposal: str       # free-text from worker; advisory only
    ts: float

@dataclass
class Hypothesis:
    id: str
    parent_ids: list[str]       # DAG: refine/split/merge ancestry
    claim: str                  # free-form
    predictions: list[Prediction]
    status: str                 # free-form: "open" | "confirmed" | "refuted"
                                # | "refined→<id>" | "split→[ids]" | "merged→<id>"
                                # | "suspended" | "superseded"
    generation: int             # depth in the DAG
    rationale: str              # why we proposed it
```

### 3.2 The ObservationLog

```python
@dataclass
class Observation:
    id: str
    text: str                   # rendered evidence (sql row excerpt, log snippet)
    source_tool_call: str       # the tool_call_id that produced it
    tool_signature: str         # canonical (tool_name, normalized_args) hash
    related_symptoms: list[str] # which symptoms it speaks to
    related_predictions: list[str]
    ts: float
```

The log serves two roles:

- **Evidence**: a primary store of what we've observed, citable by check results.
- **Memoization**: keyed by `tool_signature`, idempotent tool calls hit the log first and return historical observations rather than re-executing. Non-idempotent tools (declared `idempotent: false` at registration) bypass the cache.

This is what shrinks redundant exploration when worker sessions are independent — the worker brief (§5.4) injects log slices relevant to the assigned prediction, and the tool wrapper short-circuits already-answered queries.

### 3.3 Update operators

The graph mutates only through these operators. Each operator is a function `(graph, update) → graph` plus a structural precondition check.

| Operator | Precondition | Effect |
|---|---|---|
| `propose(H)` | ≥1 negative prediction declared | append H to graph as `open` |
| `confirm(H)` | falsification gate (§7) | H.status := `confirmed` |
| `refute(H)` | steelman gate (§7) | H.status := `refuted`; close open predictions |
| `refine(H, condition)` | check observed partial match | create H' with H' parent=H, H.status := `refined→H'` |
| `split(H, [H1, H2, ...])` | check observed multi-mechanism evidence | create children, H.status := `split→[ids]` |
| `merge([H1, H2], H')` | predictions of H1, H2 converge | create H', mark sources `merged→H'` |
| `supersede(H, H')` | a more precise sibling exists | H.status := `superseded` |
| `suspend(H, reason)` | external information needed | H.status := `suspended` |
| `record_observation(o)` | (none) | append to ObservationLog |
| `attach_check(p, c)` | p is open | append c to p.checks |

Operators are **proposed** by sessions (orchestrator or worker) and **applied** by the FSM gate atom (§7). Single-writer-via-gate prevents concurrent graph corruption and centralizes falsification enforcement.

---

## 4. The FSM (semantic structure)

```
       ┌─────────┐
       │ INTAKE  │  symptoms recorded
       └────┬────┘
            ▼
       ┌─────────┐
   ┌──>│ OBSERVE │  gather facts; no hypothesis yet OR existing ones all closed
   │   └────┬────┘
   │        ▼
   │   ┌──────────────┐
   │   │ HYPOTHESIZE  │  propose ≥1 H with ≥1 negative prediction each
   │   └────┬─────────┘
   │        ▼
   │   ┌─────────┐
   │   │ VERIFY  │  pick next prediction (§8 scheduler) → dispatch worker → record check
   │   └────┬────┘
   │        ▼
   │   ┌─────────┐
   │   │ JUDGE   │  apply update operator(s) via gate (§7)
   │   └────┬────┘
   │        │
   │  ┌─────┴────┐
   │  │          │
   │  ▼          ▼
   │ all open    confirmed H covers all symptoms?
   │ predictions       │
   │ exhausted &       └─yes──> ┌──────────┐
   └─no candidates             │ FINALIZE │
              │                 └──────────┘
              ▼
        ┌──────────┐
        │ BLOCKED  │  external help needed
        └──────────┘
```

**The FSM does not gate LLM thinking.** Free-form text is unconstrained. The FSM gates **graph mutations** and **worker dispatch**. The LLM in any session may reason, plan, query — but the graph only changes through evidence tools, and evidence tools route through the gate.

This is the same separation as `agent-loop.md`'s decide_turn_action: the policy layer disciplines the structured side-effects, the LLM owns its own thinking.

---

## 5. Layered Context (orthogonal axis)

The FSM is invariant under three different session topologies. This scenario picks **L1+L2+L3 layered context**, because alternatives have known failure modes for trajectories that exceed ~30 verifications:

| Topology | Trajectory length | Knowledge sharing | Failure mode |
|---|---|---|---|
| Monolithic single session | Linear in tool calls | Free (one context) | Context dilution; cost; attention drift |
| Per-state new session | Short per session | Brief required | Cold-start re-exploration |
| **L1 + L2 + L3 (this design)** | L2 linear in *hypothesis events*; L3 ephemeral | L1 persistent store + brief | Brief construction is non-trivial; worker bias (§6) |

### 5.1 L1 — Persistent Structured Layer

Owned by atom `rca_hgraph_store`. Survives across all sessions in the trace. Contents:

- `SymptomSet`
- `HypothesisGraph` (§3)
- `ObservationLog` (§3.2)
- Derived views, materialized on read:
  - `open_leaves`: hypotheses with no descendants and status `open`
  - `unexplained_symptoms`: symptoms not yet linked to a confirmed prediction
  - `refuted_branches`: closed subtrees, used as "no-go zones" in worker briefs
  - `contested_predictions`: predictions where two independent workers disagreed

### 5.2 L2 — Orchestrator Session

One long-lived session per trace. The investigator persona. Its context contains:

- The current derived views of L1 (refreshed via prompt template at each turn)
- The last K hypothesis events (default K=20)
- Compaction summaries of older events (triggered at state transitions — see §5.5)

The L2 context **does not contain raw tool outputs**. Raw outputs live in ObservationLog (L1) and in L3 worker transcripts (which are discarded). This is the structural decision that keeps L2 growth proportional to cognitive operations rather than tool operations.

### 5.3 L3 — Worker Sessions (per verification)

One short-lived session per `plan_check + submit_check_result` cycle. Spawned by `sub_agent.dispatch` with a structured brief (§5.4). On return, only the structured WorkerReturn (§6.1) is captured back to L1; the worker's full transcript is written to observability JSONL but does not enter L2.

Persona variants:

- `investigator` — the default verifier, briefed in **falsification framing** (§6.2)
- `critic` — adversarial review of a confirmed-candidate hypothesis (existing rca scenario already has this)
- `devils_advocate` — fixed role: find one piece of evidence that contradicts the leading hypothesis (§9)

### 5.4 Worker Brief Schema

```yaml
# Constructed by rca_brief_builder atom, injected as the worker's
# first user message.
task_id: <prediction_id>
mode: verify | steelman | adversarial
prediction:
  claim: "<text>"
  polarity: positive | negative
hypothesis_blinded: true | false   # see §6.2
relevant_observations:             # L1 slice
  - {text, source_tool_call, ts}
no_go_zones:                       # refuted-branch summaries
  - "Service X DB connection was checked at 14:34 — not the cause"
expected_output:
  observations: [...]              # facts only
  interpretation:                  # advisory
    proposed_update: <update op text>
    reasoning: <text>
    confidence: <text>             # free-form, no preset scale
budget:
  max_turns: 8
  max_cost_usd: 0.10
```

The brief is **the contract**. Worker can read L1 derived views via `api.get_service('rca.hgraph')` for richer context, but the brief is what the FSM holds the worker accountable to.

### 5.5 Compaction at State Transitions

`api.compaction` is invoked from `rca_fsm_policy` at three transitions:

- `JUDGE → OBSERVE` when a hypothesis was refuted: compact the refuted branch into a one-line "no-go" record.
- `JUDGE → VERIFY` when a hypothesis advanced (refined/split): compact pre-refinement checks into a parent-attached summary, leaving descendants to carry forward.
- `* → FINALIZE`: full L2 compaction into the final report scaffold.

State-triggered compaction is **semantically justified compression** rather than token-count-triggered truncation — by definition the compressed material is no longer load-bearing for the live investigation.

---

## 6. The Worker → Orchestrator Contract (anti-bias core)

Worker returns are **two-column**:

```python
@dataclass
class WorkerReturn:
    observations: list[Observation]   # facts — go into ObservationLog
    interpretation: Interpretation    # advisory — stays in trace, optional input

@dataclass
class Interpretation:
    proposed_update: str              # free-form: "confirm H", "refine H with condition X"
    reasoning: str
    confidence: str                   # free-form
```

The orchestrator (L2) by default **consumes only `observations`** and re-derives the update decision. `interpretation` lives in the trace for audit but does not enter the graph automatically.

This is the highest-leverage anti-bias move in the entire design. It enforces: **a worker cannot replace evidence with rhetoric**. If `observations` cannot support `interpretation.proposed_update`, the orchestrator's re-derivation exposes the gap.

### 6.1 Falsification-framed briefs

Default brief mode for `verify` is "find a piece of evidence that *refutes* this prediction." Phrasing matters and is enforced at the brief-builder level:

- ✓ "Look for evidence that contradicts: <prediction>"
- ✗ "Verify the following claim: <prediction>"

Hypothesis identity is blinded by default (`hypothesis_blinded: true`) — the worker sees only the prediction text and L1 facts, not the parent hypothesis claim. This removes anchoring on the hypothesis identity.

### 6.2 Negative prediction requirement

`propose(H)` is rejected if H has zero negative predictions. The reasoning: a hypothesis with only positive predictions is unfalsifiable — every confirming observation is consistent with infinitely many alternative hypotheses. Negative predictions ("if H, then NOT Z") are the only structural mechanism for refutation.

The check sits inside the gate atom (§7); enforcement is at graph-mutation time, not prompt time.

---

## 7. The Falsification Gate

A single atom (`rca_falsification_gate`) intermediates every graph mutation. Its job is to enforce structural preconditions on the update operators (§3.3).

### 7.1 Confirm gate

```
confirm(H) accepted only if:
  ∃ ≥1 negative prediction p ∈ H.predictions where
      p has ≥1 CheckResult AND no check's observations triggered p.claim
  AND ∃ ≥1 positive prediction p' ∈ H.predictions where
      p' has ≥1 CheckResult from an *independent* worker session
      (independent = different worker_session_id and brief constructed without
       reusing the same `relevant_observations` slice)
  AND H covers all currently-unexplained symptoms
      (each symptom is linked via Observation.related_symptoms to some
       satisfied prediction of H)
```

A failing precondition does not error — it **downgrades the proposed update** to `refine(H, "needs <missing piece>")` so the orchestrator gets a structured nudge instead of a hard wall.

### 7.2 Refute gate

```
refute(H) accepted only if:
  ∃ ≥1 CheckResult on some prediction of H with worker mode = "steelman"
      (steelman mode brief asks the worker to FIND supporting evidence;
       if even the steelman fails to find any, refutation is structurally
       justified)
  OR ∃ ≥1 CheckResult whose observations directly triggered a negative
       prediction of H
```

This is asymmetric on purpose: a single triggered negative prediction is sufficient (Popperian falsifiability), but the "no supporting evidence found" path requires a steelman attempt to avoid lazy refutation.

### 7.3 Refine/Split/Merge gates

Lighter checks — these don't terminate hypotheses, they restructure the graph:

- `refine`: precondition is a non-empty CheckResult whose observations are not in `no_go_zones`.
- `split`: precondition is observations citing two or more distinct mechanisms (worker must enumerate them in `interpretation.proposed_update`).
- `merge`: precondition is overlap in satisfied predictions across two hypotheses (computed by the gate, not the worker).

### 7.4 The gate as the only writer

`rca_hgraph_store` exposes read APIs publicly but write APIs only to the gate atom. Other atoms cannot write the graph directly. This is enforced by `api.set_service` namespacing — the store publishes `rca.hgraph.read` to all atoms but `rca.hgraph.write` only to the gate. Pattern parallels `audit_check_registry` (llmharness-cognitive-audit) for write-isolation.

---

## 8. Verification Scheduler

When in `VERIFY` state, which open prediction is verified next? Three policies, picked by manifest config:

- `informational` (default): pick the prediction with the highest expected information gain — the one whose outcome would most reduce the count of open hypotheses. Information gain is computed structurally: for each open prediction, count how many open hypotheses' predictions overlap with it; higher overlap = higher discrimination potential. Approximation, not Bayesian-correct, but cheap and bias-free.
- `breadth_first`: round-robin across open hypotheses, one prediction each, to prevent orchestrator from "favoring the main suspect" before alternatives are tested.
- `orchestrator_choice`: LLM picks. Highest flexibility, highest bias — kept available for experimentation but not the default.

The `informational` default is the structural reply to "orchestrator chooses verification order, which is itself a bias source" (§5 in the conversation that produced this doc).

---

## 9. Independent Corroboration (selective)

Heavyweight anti-bias mechanisms applied only at **graph-impact-critical** mutations to control cost.

### 9.1 Critical mutation definition

A mutation is critical if it would:
- Trigger `FINALIZE` (a `confirm` on a hypothesis covering all unexplained symptoms), OR
- Close the last open branch in a major subtree (a `refute` removing the only remaining alternative).

### 9.2 Mechanisms at critical mutations

- **Twin verifier**: a second L3 worker briefed independently runs the same prediction. Briefs must not share `relevant_observations` slices verbatim (the brief-builder regenerates from L1 with a different selection seed). Disagreement marks the prediction `contested` and forces a third worker before the mutation is applied.
- **Devil's advocate**: a fixed worker persona with brief "find one piece of evidence that contradicts the leading hypothesis." Produces observations only (no proposed update). If non-empty, the orchestrator must address them in the rationale of the confirming mutation.
- **Tool-heterogeneous workers**: the second verifier is constrained to a different tool subset (e.g., primary uses SQL, twin uses log file read). Hypothesis confirmation that holds across tool surfaces is more robust than one that holds in a single surface.

### 9.3 Hypothesis generation isolation

To counter orchestrator-level anchoring on its own initial proposals: at `OBSERVE → HYPOTHESIZE`, in addition to the orchestrator's self-proposed hypotheses, spawn a `hypothesis_generator` L3 session briefed only with `SymptomSet` + `ObservationLog`, **not** with the current `HypothesisGraph`. The orchestrator must merge external proposals into its candidate set; it cannot quietly drop them.

This is the structural reply to "orchestrator's own framing biases the entire investigation."

---

## 10. Bias Telemetry

Atom `rca_bias_audit` subscribes to graph mutation events and the bus. Emits to observability JSONL:

| Metric | Definition | Surface |
|---|---|---|
| `worker_confirm_rate` | Per-worker-persona ratio of `confirm` proposals to total CheckResults | session.end summary |
| `evidence_asymmetry` | Per-hypothesis: count(supporting obs) / count(refuting obs) | hypothesis node |
| `prediction_coverage` | Per-prediction: fraction of declared test_plan steps actually executed | check_result event |
| `unexplained_symptom_count` | Count at FINALIZE attempt | finalize event |
| `gate_downgrade_count` | Number of times a proposed update was downgraded by §7 gates | trace summary |

These do not gate FSM transitions; they ride along in the trajectory for post-hoc auditing and for tuner consumption (`per_task_evolution_loop` reads JSONL traces and computes scenario-specific metrics).

---

## 11. Acceptance Scenarios

Fail-stop tests this scenario must satisfy (per CLAUDE.md "Testing philosophy"). These are the positions where the value proposition fails if broken.

1. **No-negative-prediction rejection**: `propose(H)` with zero negative predictions is rejected by the gate, and the rejection is visible as an `emit:diagnostic` event in JSONL.
2. **Confirm-without-falsification rejection**: `confirm(H)` proposed without any negative prediction having been checked is downgraded to `refine`, not applied.
3. **Refute-without-steelman rejection**: `refute(H)` proposed with no triggered negative prediction AND no steelman check is downgraded.
4. **Worker observation/interpretation separation**: the orchestrator can be configured to ignore `interpretation`; graph state is reproducible from `observations` alone. Verified by replaying a trace with `interpretation` zeroed out and checking graph equality.
5. **ObservationLog memoization**: a worker that issues a `tool_signature`-equivalent call to an earlier observation receives the cached result; observability JSONL records a `tool_call_cached` event.
6. **Single-writer property**: any atom other than `rca_falsification_gate` attempting to write `rca.hgraph` raises and is caught; mutation is not applied.
7. **FINALIZE coverage check**: `submit_final_report` is rejected if `unexplained_symptoms` is non-empty.
8. **L2 growth bound**: trajectory test on the eval set — L2 token count at trace end grows sub-linearly in total tool calls (linear in hypothesis events instead). Concretely: regression test that L2 tokens / total_tool_calls < 0.3 across the eval suite.
9. **Independent-worker independence**: at a critical mutation, the two briefs constructed must not share verbatim `relevant_observations` text. Property-test the brief-builder.
10. **Hypothesis-generator blinding**: the external proposer atom's brief must not contain any text from the current `HypothesisGraph`. Property-tested.

These run via `uv run pytest` against the new scenario; they are scenario-local tests under `contrib/scenarios/rca_hfsm/tests/` (do not pollute the core fail-stop suite).

---

## 12. Open Questions / Phasing

### Phase 1 (MVP — first slice of value)

- §3 hypothesis graph schema + store atom
- §6 two-column WorkerReturn contract
- §7.1, §7.2 falsification + refute gates
- §6.2 negative prediction requirement
- §4 FSM transitions (informational scheduler default)
- §5.1, §5.2, §5.3 layered context (no compaction yet — verify L2 stays small enough on representative eval cases first)
- Acceptance scenarios 1–7

### Phase 2 (anti-bias depth)

- §5.5 state-triggered compaction
- §9 independent corroboration mechanisms
- §10 bias telemetry
- Acceptance scenarios 8–10

### Phase 3 (open, requires data)

- **Cross-trace skill learning**: when a worker discovers a methodology that succeeded (e.g., "for disk-fill RCA, check logrotate state before inode usage"), should it be persisted as a skill? If yes, where — trace-local L1, or cross-trace via the existing `skills` resource axis? This connects to `evolution_substrate.md` and `per_task_evolution_loop.md`. Currently parked: implement when there's enough trace data to know what's worth distilling.
- **Multi-incident memory**: an RCA case may produce facts useful to a *future* case (a known buggy library, a known noisy metric). Same question as above — too early to commit to a shape.
- **Concurrent verification**: §3.3 single-writer-via-gate works for serial; parallel workers on independent predictions need either a queue at the gate or per-prediction optimistic locking. Defer until L1/L2 cost data shows serial verification is the bottleneck.

### Resolved decisions

- Hypothesis lifecycle is DAG-shaped with rich operators (§3.3), not binary confirm/refute.
- Session topology is layered (§5), not monolithic.
- Worker → Orchestrator contract is two-column (§6), not single-verdict.
- Falsification is structurally enforced (§7), not prompt-asked.
- Verification order is information-gain-scheduled (§8) by default, not orchestrator-chosen.
- Hypothesis generation is split across orchestrator + isolated generator (§9.3), not orchestrator-only.

---

## 13. Relationship to Existing rca Scenario

`contrib/scenarios/rca/` (in-tree) is a single-orchestrator + single-critic flow with `hypothesis_tools` exposing graph manipulation as free tool calls. It works; it ships an eval suite; and several of its atoms (`duckdb_sql`, `worker_skills`, `worker_finalize`, `rcabench_contract`) are independently reusable.

This design does **not** deprecate it. The new scenario reuses:

- `duckdb_sql` and any data-source-specific tool atoms
- `worker_skills` + `skill_loader` for SKILL.md-shaped methodology resources
- `rcabench_contract` for the eval harness contract
- `observability` builtin (OTLP spans + logs to per-session ndjson)
- `sub_agent` for worker dispatch

And **replaces**:

- `hypothesis_tools` — the new graph store + gate is the structural successor
- `finalize` (the `submit_final_report` atom) — replaced by FSM `FINALIZE` state guarded by coverage check
- the free-form orchestrator → critic dispatch — replaced by the layered context + falsification gate flow

The two scenarios share an eval set (`contrib/scenarios/rca/eval/` is the obvious shared resource; the new scenario references it rather than forking). This enables a direct A/B: same tasks, same fixtures, two methodologies, observable in JSONL traces.

`per_task_evolution_loop` then has both scenarios available for evidence-driven evolution. Whether the FSM scenario actually beats the free-form one is an empirical question this design takes no position on — it commits only to making the comparison possible.
