# Constraint Satisfaction Analysis — Design

Extends the trajectory grounding analysis (SCHEMA.md) with constraint-level
verification, on an explicit theoretical footing borrowed from program
analysis. The current system catches entity grounding errors (~32% of
TELBench gold); this extension targets the ~40% value/reasoning errors it
misses.

A first draft of this design mapped constraints onto per-symbol def-use
edges. Review found that structurally unsound (evidence for a constraint
routinely lives on satellite entities unreachable from the answer entity's
chain; per-edge verdicts cannot express relational constraints; "no mapping
found" was misread as "agent never verified"). This version replaces it.

## Theoretical framing

What we analyze, and what each layer owes:

1. **A trajectory is one concrete trace, not a program.** The right frame is
   runtime verification / trace monitoring, not all-paths static analysis:
   correct agent behavior is the spec; the question is at which prefix the
   trace violates it.

2. **The goal is proving bugs present, not proving them absent.** This is
   the under-approximation regime of Incorrectness Logic (O'Hearn 2019): a
   reported error must come with a witness from the trace. Missed errors are
   tolerable; false accusations are not. TELBench span-F1 punishes false
   positives, so the regime matches the metric.

3. **May/Must layering — the two-layer contract.**
   - The *index* is a may-analysis: it over-approximates candidate issues
     (attention hints). Its obligation is recall — real errors should be in
     the candidate set. It never asserts an error.
   - The *auditor* is a must-check: it confirms a candidate only with
     concrete trace evidence (the incorrectness witness). Its obligation is
     precision.
   This is CEGAR's shape: a cheap abstraction proposes counterexample
   candidates; an expensive concrete check confirms or refutes them. The
   index→auditor pipeline already had this shape implicitly; this contract
   makes the approximation direction of each layer explicit, and forbids the
   index from emitting must-level claims (the old `constraint_unverified`
   was exactly that violation).

4. **Abstraction with a soundiness ledger.** The index is an abstract
   domain; extraction is the abstraction function α. Ours is *not* a sound
   over-approximation: extraction misses entities, truncation drops text,
   alias merges err. Per the soundiness stance (Livshits et al., CACM 2015),
   each unsoundness source is declared and, where possible, code-detectable
   (e.g. "this step's text was truncated"), so downstream passes can
   discount conclusions that stand on it.

5. **Constraints are relational queries, not dataflow properties.** Def-use
   chains are per-symbol (unary); constraint satisfaction joins evidence
   across entities — the evidence for "parent was an auctioneer" lives on
   the parent entity, not on the answer entity. The right formalism is
   Datalog-style analysis (Doop): base facts (EDB) plus derivation rules
   (IDB joins). The LLM's role is a **tuple oracle** for uncertain
   relations: it fills in individual tuples; all joins and propagation are
   code. This preserves "the model gives a point, code propagates" at the
   correct granularity.

6. **Three-valued verdicts.** Every oracle judgment is {true, false,
   unknown}; joins propagate by Kleene semantics; **unknown never escalates
   into a warning**. A missing or unparseable model verdict defaults to
   unknown — the failure polarity stays conservative (contrast: the old
   design turned a missing mapping into an `unverified` alarm).

7. **Safety vs liveness decides localization.** (Alpern–Schneider)
   - Fabrication / contradicted use are *safety* properties: a unique
     earliest bad prefix exists — the first step that uses the bad value.
   - Constraint omission is *liveness-like*: on a finished trace its
     violation point is the **commitment step** (the agent commits to an
     answer never having verified C), not any intermediate step.
   TELBench gold labels are commitment-centric — they mark liveness
   violation points. Mixing the two localization semantics is the root of
   the observed off-by-one errors. Localization policy is per property
   class, in code, zero LLM cost.

## Architecture

```
Pass 0    Constraint extraction + normalization   [LLM + code]
Pass 1–3  PARSE / RESOLVE / DATAFLOW              [existing, unchanged]
Pass E    Fact extraction (tuple oracle)          [LLM local judgments]
Pass J    Relational join                         [code, Kleene 3-valued]
Pass L    Localization                            [code, per property class]
Auditor   Must-check with witness                 [LLM]
```

### Pass 0 — constraint extraction + normalization

From the question text only, one LLM call:

```json
{"id": "c1", "subject": "answer",
 "desc": "born in the 1960s",
 "normalized": {"kind": "year_range", "lo": 1960, "hi": 1969}}
```

- `subject` binds the constraint to a variable (usually the answer
  candidate); relational constraints name both sides.
- `normalized` is emitted only when the constraint is machine-checkable
  (dates, quantities, counts). For those, satisfaction is decided **by
  code** (interval / number comparison over values located in step text);
  the LLM only locates candidate numbers, never does date arithmetic —
  the auditing model must not be able to repeat the audited agent's own
  temporal-reasoning mistake. Semantic constraints (occupation,
  relationship) stay free-text and go to the oracle.
- No preset type enum: `normalized.kind` exists only where it is
  load-bearing (it dispatches to a specific code checker).

### Pass E — fact extraction (LLM as tuple oracle)

Uncertain relations, each filled by batched local judgments:

- `Commit(step, candidate)` — the agent commits to this candidate as its
  answer. Detection policy: the last assistant step asserting an
  answer-class entity; if none exists (aborted / failed trace), `Commit` is
  empty — then no `constraint_violated` can fire and omission findings
  degrade to low confidence.
- `About(step, candidate)` — this grounded step's content is evidence about
  this candidate, directly or via a satellite entity (the parent, the book,
  the year). This relation is what makes cross-entity evidence reachable —
  it does not depend on per-symbol def-use connectivity.
- `Entails(steps, c, candidate)` / `Contradicts(steps, c, candidate)` —
  the grounded content of this step *set* establishes / refutes constraint
  c for this candidate. Judged per (constraint, candidate) over all mapped
  steps jointly, not per edge — relational constraints ("worked as X before
  acting") need joint evidence that no single edge carries.

Candidate scoping: candidates are answer-kind entities with a `Commit`
tuple or late-span references. Evidence steps are blocked per candidate by
code (entity-link and lexical overlap **on step text**, not on entity
names) before any oracle call; oracle calls are batched.

### Pass J — relational join (pure code)

```
Verified(c, cand)   :- ∃ steps: About(steps, cand) ∧ Entails(steps, c, cand)=true
Violated(c, cand)   :- ∃ steps: About(steps, cand) ∧ Contradicts(steps, c, cand)=true
CommitError(c, s)   :- Commit(s, cand) ∧ Violated(c, cand)
Omitted(c, s)       :- Commit(s, cand)
                        ∧ no Verified/Violated/unknown tuple for (c, cand)
                        ∧ code-negative: no tool-output step in the whole
                          trace shares c's content tokens or normalized values
```

- Violations by **non-committed** candidates are exploration, not errors:
  the join keys on the committed candidate. (A grounded "Kipling born 1865"
  that violates "born in the 1860s" is the agent correctly *rejecting*
  Kipling — it must not fire.)
- `Omitted` requires the code-checkable negative, not merely "the mapping
  pipeline returned nothing": **unmapped ≠ unverified**. Any unknown tuple
  for (c, cand) suppresses `Omitted`.

### Pass L — localization (pure code)

| error class | property class | violation point |
|---|---|---|
| ungrounded / contradicted use | safety | earliest use step |
| CommitError | safety at commit | commit step |
| Omitted | liveness | commit step |

### Auditor contract

The index layer ships *candidates with suggested spans*, never verdicts.
The auditor confirms each candidate with a witness (quoted trace content)
or drops it. New warning kinds are gated per kind, so a noisy kind can be
disabled without losing the others.

## Data model

```python
@dataclass(frozen=True, slots=True)
class Constraint:
    id: str
    subject: str                     # variable the predicate is about
    description: str                 # free text
    normalized: Mapping[str, Any] | None = None  # only when machine-checkable

type FindingStatus = Literal["verified", "violated", "omitted", "unknown"]

@dataclass(frozen=True, slots=True)
class ConstraintFinding:
    constraint_id: str
    candidate: str                   # committed candidate it was judged for
    status: FindingStatus
    evidence_step_ids: tuple[str, ...]
    commit_step_id: str | None       # localization anchor for commit-class errors
    confidence: float = 1.0
```

`TrajectoryIndex` gains `constraints` and `constraint_findings`; **both are
serialized in `dump()`/`load()`** (the auditor consumes a loaded index).
Pass E/J/L output is derived state under the same idempotence contract as
Pass 3: any alias merge or dependency rebuild invalidates it wholesale; the
passes rebuild from facts. Pass E/J/L run strictly after all Pass 2/3
mutations.

`warnings()` stays "pure code, no LLM" as documented. Constraint findings
surface through a parallel accessor (`constraint_attention()`), merged into
the auditor's attention-hint feed by the caller — the documented
recomputability invariant of `warnings()` is untouched.

## Review findings addressed

| review finding | resolution |
|---|---|
| per-symbol chains can't reach satellite-entity evidence | `About` relation maps steps→candidate independent of def-use connectivity |
| "unverified" unsound by construction | three-valued verdicts; `Omitted` needs code-checkable negative + Commit |
| per-edge verdicts can't express relational constraints | `Entails`/`Contradicts` judged per (constraint, candidate) over the step set |
| rejected candidates trigger false violations | joins key on the committed candidate only |
| lexical filter matches entity names, not evidence | blocking runs on step text; placeholder symbols excluded |
| def_value/use_value often None; compare_values sends step text | oracle inputs are step texts, cost accounted accordingly |
| adjudicator repeats the agent's own date arithmetic errors | machine-checkable constraints normalized in Pass 0, decided by code |
| "misinterpreted" status had no producer | dropped; `CommitError` covers commit-time misjudgment |
| final-answer-span assumption unreliable | explicit `Commit` detection policy with an empty-commit degradation path |

## Validation plan (ordered)

1. **One-call baseline first.** Question + final answer + the grounded
   tool-output snippets already in the index → a single LLM call listing
   verified / violated / unaddressed constraints with cited steps. This is
   Pass E+J collapsed into one call, with the whole-evidence view. If the
   pass structure cannot beat it on a dev slice, the decomposition is
   losing information — stop and rethink.
2. **Dev slice** (~30 hand-labeled (constraint, evidence-steps, verdict)
   triples) to attribute failures to mapping vs judgment vs aggregation vs
   localization. Offline script, not pytest.
3. **Per-kind precision gates** measured on the dev slice before enabling
   any new warning kind by default.
4. TELBench full run only after 1–3.

The impact estimate from the taxonomy (~21% of gold errors touch constraint
handling) is a **taxonomy-derived ceiling**, not an expected yield —
detection, mapping, and localization losses all discount it.

## Cost

- Pass 0: 1 call (question only).
- Pass E: ~2 batched calls (About blocking; Entails/Contradicts per
  committed candidate). Machine-checkable constraints cost 0 oracle calls.
- Pass J/L: code.
- Upper bound +3 calls/trajectory, comparable to the existing 4
  (extraction, alias, coreference, compare_values). Oracle inputs are step
  texts, so token cost is dominated by evidence-step length, not call count.

## Out of scope

- AFTraj: its "constraints" are preconditions / invariants / success
  criteria, not entity attributes; Pass 0's prompt is TELBench-shaped.
  Pass E/J stay gated off for AFTraj until a domain prompt exists — the
  62.5% F1 there must not be exposed to downside.
- Multi-hop constraints that emerge mid-trajectory (from tool results, not
  the question).
- Pure reasoning errors with no constraint anchor (unsupported inference
  over fully grounded facts).

## References

- SCHEMA.md — the grounding analysis this extends.
- O'Hearn, *Incorrectness Logic*, POPL 2020 — under-approximate bug-finding.
- Cousot & Cousot — abstract interpretation; soundness as over-approximation.
- Livshits et al., *In Defense of Soundiness*, CACM 2015 — declared
  unsoundness sources.
- Doop / Datalog-based pointer analysis — analysis as relations + joins.
- Bruns & Godefroid — three-valued model checking.
- Alpern & Schneider — safety/liveness decomposition; bad prefixes.
- Clarke et al. — CEGAR.
