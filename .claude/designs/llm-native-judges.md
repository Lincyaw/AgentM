# Design: LLM-Native Judges

**Status**: PROPOSED
**Created**: 2026-05-13
**Builds on**: [hypothesis-driven-rca.md](hypothesis-driven-rca.md), [pluggable-architecture.md](pluggable-architecture.md), [extension-as-scenario.md](extension-as-scenario.md)
**Relationship**: vertical to `hypothesis-driven-rca.md`, not a version of it. That doc describes *what* the system is (graph + FSM + layered context). This doc describes *how* the system makes decisions (judgments, not structural rules). Both coexist.

---

## 1. First Principle

> **Structural enforcement is the wrong primitive for semantic judgments.** The gate's job is to be the only writer of the HypothesisGraph; its decision logic is LLM judgment, not regex or structural rules.

Phase 1 of `hypothesis-driven-rca` shipped a falsification gate whose preconditions were enforced by word-boundary regex on free-text fields, mechanical chain-walking through observation/prediction/hypothesis links, and literal `worker_session_id` comparison. Each of these works for the cases it was designed for and fails on cases that mean the same thing but spell it differently — two workers reading the same brief and writing identical evidence but with different session IDs look "independent" structurally; a worker writing "the data corroborates H" instead of "supports H" looks "unclear" to the regex; a hypothesis that explains 95% of symptoms but leaves one unlinked observation in a corner fails the coverage check despite being the right answer.

The honest fix is not better regex. The honest fix is to take every "this looks like a rule but is actually a judgment" point in the system and route it through the LLM.

---

## 2. The Three True Invariants

After this refactor, only three things in rca_hfsm remain non-negotiable rules:

1. **Typed graph state** — the shape of `Hypothesis`/`Prediction`/`CheckResult`/`Observation` does not change. Data is structured because data shape and decision logic are different concerns.
2. **Single-writer property** — the gate is the only atom that mutates the graph. This is enforcement, not judgment; it has no semantic content.
3. **Two-column WorkerReturn contract** — `observations[]` is the fact channel (enters the graph), `interpretation` is the advisory channel (audit only). This is interface, not judgment.

Everything else — *what counts as a satisfied prediction, what counts as coverage, what counts as independence, whether falsification was genuinely attempted, which prediction to verify next, whether evidence explains a symptom* — is judgment. All judgment moves behind the **Judge port**.

---

## 3. The Judge Port

A new pluggability axis. Each kind of judgment is one Protocol; each implementation is one §11 atom. Atoms register via `api.set_service('rca.judge.<kind>', impl)`. The gate retrieves and calls; it does not embed decision logic.

### 3.1 Protocol shape

```python
@runtime_checkable
class Judge(Protocol):
    kind: str                              # "satisfied" | "coverage" | "independence" | "falsified_genuinely"

    def judge(self, context: JudgeContext) -> Verdict: ...
```

```python
@dataclass(frozen=True)
class JudgeContext:
    # Structured slice of L1 graph state relevant to this judgment.
    # Concrete fields depend on the judge kind (see §4).
    graph_slice: dict[str, Any]
    operands: dict[str, Any]               # the specific inputs to this judgment

@dataclass(frozen=True)
class Verdict:
    verdict: str                           # free-text: "satisfied", "refuted", "unclear", "independent", "redundant", ...
                                           # Schema is free-text per CLAUDE.md "no preset enums for subjective dimensions";
                                           # callers may pattern-match on canonical strings the LLM is asked to emit,
                                           # but the field is not Literal-constrained.
    reason: str                            # free-text rationale, surfaced to LLM on downgrade
    confidence: str                        # free-text, advisory
```

### 3.2 Two modes per judge

Each judge atom supports two backing implementations toggled by `config.mode`:

- `mode: "llm"` (default for production) — calls the LLM with a small focused prompt and **requires structured output via tool_use**. The LLM is forced to call a single judgment tool whose schema matches `Verdict`. There is no string parsing on the judge's output.
- `mode: "stub"` (default for tests) — reads a scripted verdict from `config.scripted: list[Verdict]` and returns them in order. Deterministic, fast, free.

The `mode` flag is the only configuration. No knobs for "how strict" or "what threshold" — the prompt is the policy.

### 3.3 Caching

Per-session in-memory LRU keyed by `sha256(canonical_json(context))`. Same judge + same context returns the same verdict without a second LLM call. Cache is cleared at session start; cross-session caching is a Phase 3 optimization.

### 3.4 Failure modes

- LLM call fails (provider error, timeout): retry once with the same prompt; on second failure, return `Verdict(verdict="unclear", reason="judge LLM unreachable: <err>", confidence="none")`. The gate treats `unclear` exactly like a precondition miss — surfaces the reason to the calling LLM, which decides next steps.
- LLM returns malformed structured output (tool_use schema mismatch): retry once; on second failure, also `unclear`.

There is no fallback to structural rules. `unclear` is a first-class verdict the system reasons about.

---

## 4. The Four Default Judges

The minimum set to remove every regex and structural-rule branch from the Phase 1 gate. Each replaces a specific piece of structural logic.

### 4.1 `judge.satisfied`

**Replaces**: word-boundary regex on `verdict_proposal` (`triggered` / `supports` / `steelman` lemma matching).

**Asks**: Given this prediction (claim + polarity) and these check results (observations + interpretation), is the prediction satisfied?

**Returns**: `verdict ∈ {"satisfied", "refuted", "unclear", "partial"}` (canonical strings the prompt requests; field is still free-text). `partial` is a new outcome that opens the natural path to refine (see §5.2).

### 4.2 `judge.coverage`

**Replaces**: mechanical chain-walk through `Observation.related_symptoms` → satisfied prediction → confirmed hypothesis.

**Asks**: Given this candidate-confirmed hypothesis, the symptom set, and the observation log, does the hypothesis explain every symptom?

**Returns**: `verdict ∈ {"covers", "gaps", "unclear"}`. If `gaps`, `reason` enumerates which symptoms are not addressed.

### 4.3 `judge.independence`

**Replaces**: `worker_session_id` literal compare.

**Asks**: Given two check results, are they drawn from genuinely independent investigations, or do they share enough source/reasoning that one corroborates the other only superficially?

**Returns**: `verdict ∈ {"independent", "redundant", "unclear"}`. Catches: same source data, brief copy-paste, identical reasoning chain with different session IDs, two passes by the same worker mode under different IDs.

### 4.4 `judge.falsified_genuinely`

**Replaces**: structural "at least one negative prediction has been checked" rule.

**Asks**: Given the full check history of a hypothesis, was a genuine falsification attempt made — i.e., did the verification process actually look for evidence that would refute the hypothesis, not just gather supporting evidence?

**Returns**: `verdict ∈ {"genuine_attempt", "no_attempt", "unclear"}`. This captures cases the structural rule misses: a worker that "checked" a negative prediction by writing one cursory observation and concluding "not triggered" without real investigation.

---

## 5. The Gate as Dispatcher

After this refactor, the gate atom's code shape is:

```python
def apply(self, update: UpdateProposal) -> UpdateResult:
    if update.op == "propose":
        return self._apply_propose(update)
    if update.op == "confirm":
        return self._apply_confirm(update)
    if update.op == "refute":
        return self._apply_refute(update)
    if update.op == "attach_check":
        return self._apply_attach_check(update)
    # ... light-precondition operators apply directly ...
```

Each `_apply_*` does **one thing**: consult the relevant judge(s), and either write or downgrade based on the verdict.

### 5.1 `_apply_confirm` (post-refactor)

```python
def _apply_confirm(self, update):
    h = self.read.get_hypothesis(update.hypothesis_id)

    falsified = self.judges.falsified_genuinely.judge(JudgeContext(
        graph_slice={"hypothesis": h, "checks": self.read.checks_for(h.id)},
        operands={},
    ))
    if falsified.verdict != "genuine_attempt":
        return UpdateResult.downgraded(reason=falsified.reason, ...)

    # Independence: pick two supporting checks; ask the judge.
    indep = self.judges.independence.judge(JudgeContext(...))
    if indep.verdict != "independent":
        return UpdateResult.downgraded(reason=indep.reason, ...)

    cov = self.judges.coverage.judge(JudgeContext(...))
    if cov.verdict != "covers":
        return UpdateResult.downgraded(reason=cov.reason, ...)

    return self._write(h, "confirmed")
```

There is **no regex**. There is **no enumeration of which lemma triggers what**. The LLM judge reads the structured graph slice and the natural-language `verdict_proposal` / `interpretation.reasoning` text and answers.

### 5.2 Downgrade-application semantics change

A second consequence of the refactor, surfaced by Phase 1 implementation:

In Phase 1, a failed `confirm` was downgraded to `refine(H → H')` and the refine was **applied** by the gate. This stranded the parent hypothesis as `refined→H'` and broke "re-attach a second worker, retry confirm" flows.

Post-refactor: gate returns `UpdateResult.downgraded(suggested=..., applied_id=None, reason=judge.reason)`. The refine is **not applied**. The orchestrator LLM reads the judge's reason and decides next steps — gather more evidence, propose refine explicitly, propose split, propose merge. The gate stops deciding on the LLM's behalf.

This is the same principle as the rest of the refactor: where Phase 1 had a structural rule ("downgrade means apply refine"), Phase 2 has a judgment ("here's what the judge said, you decide").

---

## 6. Why N Small Judges, Not One Big Judge

The alternative is one judge per gate-decision (or even one judge per session). It's tempting — fewer LLM calls per turn, one prompt to tune.

We choose N small judges for three reasons all aligned with the existing `Simple and pluggable` memory:

1. **Independent eval**: each judge has its own input/output schema. Eval data per judge tells us which is the weak link without re-running the whole system.
2. **Independent swap**: a scenario can replace `judge.independence` with a custom implementation (e.g., one tuned for a specific domain's sense of "independent evidence") without touching the other judges. This is the §11 atom contract paying off.
3. **Independent caching**: the same `(prediction, checks)` slice may come up multiple times in a long trace. Per-judge cache hit rates are much higher than whole-decision cache hit rates.

Cost concern is real but small: a typical Phase 1 trace had ~30 gate `apply` calls. Each `confirm` triggers 3 judges; each `attach_check` triggers 1 (satisfied). With caching, ~50–80 judge calls per trace, each using ~200 input tokens + ~50 output. Negligible relative to the orchestrator + worker token costs.

---

## 7. How Original §9 / §10 Features Collapse into Judges

The Phase-2-or-later features the prior design enumerated mostly stop being separate machinery once the Judge port exists. They become *configurations* of existing judges or *new judges of the same shape*:

| Original feature | Post-refactor form |
|---|---|
| Twin verifier (§9.2) | `judge.independence` is *required* to pass on `confirm`; the orchestrator decides whether to spawn a second worker based on the verdict. The judge does the calling — no new machinery. |
| Devil's advocate (§9.2) | A `judge.find_contradiction` (Phase 3, not Phase 2): asks the LLM to find one observation/argument that would refute the leading hypothesis. Same Protocol shape. |
| Hypothesis generator isolation (§9.3) | A `judge.propose_alternatives` (Phase 3): same shape, prompt receives only `SymptomSet` + `ObservationLog` (no current `HypothesisGraph`), returns candidate hypotheses. |
| Bias telemetry (§10) | Every judge call writes its inputs + verdict to the observability JSONL. Telemetry is *queries over that stream*, not a separate subsystem. The bias signals the original §10 listed (worker confirm rate, evidence asymmetry, etc.) all become trace queries. |
| State-triggered compaction (§5.5) | A `judge.what_can_be_dropped` (Phase 3): at JUDGE→OBSERVE/VERIFY transitions, asks the LLM to summarize refuted branches into a one-line trail. The FSM transition is when, the judge decides what. |

The lesson: once the Judge port exists, "Phase 2 features" mostly become "more judges of the same shape." This is the elegance the simple-and-pluggable principle was pointing toward.

---

## 8. Acceptance Properties

Phase 2's refactor is correct iff:

1. **Zero regex / string-matching in gate source.** A grep over `rca_falsification_gate.py` for `re\.`, `\\b`, `.match(`, `.search(`, `"triggered"`, `"supports"`, `"steelman"` returns empty. Asserted by a test.
2. **Phase 1's 50 fail-stop tests still pass** when scenario mounts stub-mode judges scripted to mirror Phase 1's structural decisions. This proves behavior preservation under the refactor.
3. **Gate function signatures unchanged.** External callers (evidence tools, FSM policy, finalize) see the same `gate.apply(update) -> UpdateResult` API. The refactor is internal.
4. **Each judge has both stub and llm modes; both modes covered by tests.** Stub tests are unit-style; llm tests are contract-style (mocked provider returns scripted tool_use payloads; judge atom passes them through to `Verdict`).
5. **Eval run on 10 cases against `contrib/scenarios/rca/eval/tasks/`** completes without infrastructure errors, produces per-case trajectory under `.agentm/observability/`, and yields a results report. Pass-rate is not a strict acceptance — the question is whether the refactor *behaves reasonably* on real cases, not whether it matches a target score on the first run.

The pass-rate question is intentionally not in the acceptance gate. Phase 2's purpose is removing hardcode and enabling LLM-driven judgment; whether that yields better answers is a downstream measurement (Phase 3+ would tune prompts and possibly the judge set based on what the 10-case run reveals).

---

## 9. What Stays Structural

For clarity, the things that remain non-LLM in Phase 2:

- **Graph topology** — DAG with operators; mutation type taxonomy; node/edge shapes. Data is structured because data is data.
- **Single-writer enforcement** — gate has exclusive write access via the commit-1 token mechanism. Token check is a struct comparison, not a judgment.
- **Two-column return contract** — `WorkerReturn` schema. Worker writes observations and interpretation in separate fields. The split is interface, not interpretation.
- **FSM states** — `INTAKE | OBSERVE | HYPOTHESIZE | VERIFY | JUDGE | FINALIZE | BLOCKED` is a fixed enum, because state is a structural concept, not a subjective one.
- **Evidence-tool routing** — tool calls dispatch UpdateProposals to the gate; that wiring is plumbing.
- **Idempotency-keyed memoization in `rca_observation_cache`** — the tool_signature hash is data, not a judgment about evidence equivalence. (Whether two *observations* are equivalent is a judgment; whether two *tool calls* with identical signatures should return the same value is not.)

Everything that left the gate's branching logic went to judges. Everything that's still in code is data flow, type shapes, and enforcement.

---

## 10. Relationship to `hypothesis-driven-rca`

Same scenario. Same `rca_hfsm` directory. Same graph schema. Same FSM. Same atoms 1–4 from Phase 1 (store, gate-as-atom-shape, evidence_tools, observation_cache, fsm_policy, brief_builder, finalize, scheduler).

What changes: the **gate atom's internals**, the **scheduler.py** (optional, deferred — see §4 deferral note), and **scenario manifest** (mounts judge atoms in addition to the rest).

What doesn't change: the public surface every other atom interacts with. `gate.apply(update) -> UpdateResult` is invariant. `api.get_service("rca.hgraph.read")` is invariant. The HypothesisGraph schema is invariant.

Phase 2 is in spirit a *deletion*: it removes the structural rules and replaces them with judge-Protocol indirection. Lines of code go down; pluggability and adaptability go up; correctness on cases that "say it differently" goes up.

The hypothesis-driven-rca design document doesn't change. This design supplements it, capturing the decision-substrate refactor as its own concept so future readers don't have to chase the elegant version through Phase 1's regex-laden first cut.
