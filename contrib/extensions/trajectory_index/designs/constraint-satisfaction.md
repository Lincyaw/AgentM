# Constraint Satisfaction Analysis — Design

Extends the trajectory grounding analysis (SCHEMA.md) with constraint-level
verification. The current system catches entity grounding errors (~32% of
TELBench gold); this extension targets the ~40% value/reasoning errors it
misses.

Two review rounds shaped this design. The first killed a per-edge mapping
draft (evidence for a constraint routinely lives on satellite entities
unreachable from the answer entity's def-use chain; per-edge verdicts cannot
express relational constraints; "no mapping found" was misread as "agent
never verified"). The second stress-tested the theoretical framing and
stripped it to the load-bearing parts recorded below.

## Principles

These are the contracts the implementation must honor. Each is checkable;
none is aspirational. Related formal framings are noted at the end, as
orientation rather than as claimed guarantees.

### P1 — Two-layer contract: the index proposes, the auditor asserts

The index layer ships *candidates with suggested spans*; it never asserts an
error. The auditor confirms a candidate only with a witness (quoted trace
content) or drops it. New warning kinds are gated per kind, so a noisy kind
can be disabled without losing the others.

Precision-first is a **metric-contingent bet**, not a structural fact: F1 is
symmetric in false positives and false negatives; this regime pays while the
system is precision-limited (which failure analysis currently shows), and
must be re-validated whenever the operating point shifts.

### P2 — No silent false negatives downstream of extraction

Extraction itself has an unbounded, undetectable miss rate (an LLM silently
failing to extract an entity leaves no trace). Downstream of it, the
enforceable invariant is: **every pruned candidate is pruned by a recorded,
deterministic code decision** — per-stage recall loss is measurable and
attributable. Injection test: an error injected into an index is either in
the candidate set or in a named stage's prune log, never simply gone.
Detectable unsoundness sources (step-text truncation, failed oracle calls)
are recorded so downstream consumers can discount conclusions standing on
them; the undetectable extraction miss is acknowledged as the one hole the
ledger cannot cover.

### P3 — Transcript determinism

Every model influence on a verdict flows through a **recorded oracle tuple**
(this includes the Omitted coverage sweep — an oracle tuple consumed by Pass
J's code, not a Pass J model call). Therefore the derived layer is a
deterministic, idempotent function of (base facts, oracle transcript):
rebuild wholesale, same answers → same verdicts. Ablating a tuple and
re-deriving yields the verdict's **dependence lineage** — which recorded
answers it stands on — enabling eval attribution. (It does not yield oracle
counterfactuals: judgments are batched, so one item's presence conditions
another's answer at generation time. Replay-based re-derivation from a
persisted transcript is not implemented yet; today the transcript serves
attribution and audit.)

### P4 — Bounded-context judgments, positive polarity

The old "local judgments only" slogan is retired — evidence aggregation for
relational constraints is inherently a judgment over a step *set*, and
coverage checking is inherently global. The honest model/code line:

- (i) every oracle call reads a **code-selected window of whole steps** —
  selection never cuts content mid-step (a partial view silently poisons
  the judgment); deselection is a logged prune; an oversized window fails
  the call and degrades to unknown rather than being trimmed;
- (ii) code owns selection, aggregation over unbounded scope, arithmetic,
  and everything decidable;
- (iii) an oracle judgment may assert only **positive facts about presented
  content**. It never asserts absence or any claim quantified beyond its
  window. Global-negative claims exist only where an explicit
  attested-coverage mechanism licenses them (see the Omitted rule).

(iii) is the load-bearing clause: it retroactively classifies the killed
`constraint_unverified` design as the violation it was, and keeps joint
`Entails` judgments legitimate (positive, in-window).

### P5 — Three-valued verdicts; unknown never escalates

Every oracle judgment is {true, false, unknown}; joins propagate by Kleene
semantics; **unknown never becomes a warning**. A missing or unparseable
model verdict defaults to unknown — failure polarity stays conservative.

### P6 — Code owns the decidable

Machine-checkable constraints (dates, quantities, counts) are normalized at
extraction and decided by **code** (interval/number comparison); the oracle
only locates candidate values in text, never does date arithmetic. The
auditing model must not be able to repeat the audited agent's own
temporal-reasoning mistake.

### P7 — Localization = end of the minimal bad prefix, anchors validated per kind

All error classes here are *safety* properties over the finite trace: each
has a minimal bad prefix, and the violation point is that prefix's end.
"Verify C before committing" is a precedence property whose bad prefix ends
at the **commit step**; fabricated-use's bad prefix ends at the earliest
reliance. Which anchor matches gold is a fact about the benchmark's **label
convention** (TELBench gold is commitment-centric), not derivable from first
principles — anchors are set per error kind and validated empirically on the
dev slice.

## Architecture

```
Pass 0    Constraint extraction + normalization   [LLM + code]
Pass 1–3  PARSE / RESOLVE / DATAFLOW              [existing, unchanged]
Pass E    Fact extraction (oracle tuples)         [LLM, bounded windows]
Pass J    Relational join                         [code, Kleene 3-valued]
Pass L    Localization                            [code, per-kind anchors]
Auditor   Must-check with witness                 [LLM]
```

Relations are the model/code interface: the oracle fills tuples of uncertain
relations; code does all joins and propagation over them (P3, P4). The rules
are non-recursive two-level joins — no fixpoint machinery needed or implied.

### Pass 0 — constraint extraction + normalization

From the question text only, one LLM call:

```json
{"id": "c1", "subject": "answer",
 "desc": "born in the 1960s",
 "normalized": {"kind": "year_range", "lo": 1960, "hi": 1969}}
```

- `subject` binds the constraint to a variable (usually the answer
  candidate); relational constraints name both sides.
- `normalized` is emitted only when the constraint is machine-checkable;
  those are decided by code per P6. No preset type enum: `normalized.kind`
  exists only where it is load-bearing (dispatches to a code checker).

### Pass E — fact extraction (oracle tuples)

Uncertain relations, filled by batched bounded-window judgments (P4):

- `Commit(step, candidate)` — the agent commits to this candidate as its
  answer. Detection policy: the last assistant step asserting an
  answer-class entity; if none exists (aborted / failed trace), `Commit` is
  empty — then no violation finding can fire and omission findings degrade
  to low confidence.
- `About(step, candidate)` — this grounded step's content is evidence about
  this candidate, directly or via a satellite entity (the parent, the book,
  the year). Per-step tuples; this is what makes cross-entity evidence
  reachable independent of def-use connectivity.
- `Entails(S, c, cand)` / `Contradicts(S, c, cand)` — the grounded content
  of step set `S = {s : About(s, cand) = true}` (code-computed lifting,
  capped per P4-i) establishes / refutes constraint c for this candidate.
  Judged per (constraint, candidate) over the set jointly — relational
  constraints ("worked as X before acting") need joint evidence no single
  step carries. Positive-polarity: the oracle reports what the window
  establishes, never what "nothing establishes".

Candidate scoping: candidates are answer-kind entities with a `Commit`
tuple or late-span references. Evidence steps are blocked per candidate by
code (entity-link and lexical overlap **on step text**, not entity names);
every prune is logged (P2).

### Pass J — relational join (code over recorded tuples)

```
Verified(c, cand)   :- Entails(S, c, cand) = true
Violated(c, cand)   :- Contradicts(S, c, cand) = true
CommitError(c, s)   :- Commit(s, cand) ∧ Violated(c, cand)
Omitted(c, s)       :- Commit(s, cand)
                        ∧ no true/unknown Verified- or Violated-support for c
                        ∧ lexical-negative ∧ sweep-negative      (see below)
```

- Violations by **non-committed** candidates are exploration, not errors:
  the join keys on the committed candidate. (A grounded "Kipling born 1865"
  violating "born in the 1860s" is the agent correctly *rejecting* Kipling.)
- Any unknown tuple for (c, cand) suppresses `Omitted` (P5).

**The Omitted rule** asserts a global negative, which no bounded-window
judgment may license (P4-iii). It fires only when **two independent absence
checks conjoin**:

1. *Lexical code-negative*: no tool-output step in the whole trace shares
   c's content tokens or normalized values.
2. *Attested coverage sweep*: one oracle call over all grounded snippets —
   "is there evidence bearing on c anywhere here?" Candidate-unbound
   (deliberately more conservative: suppresses more). A "yes" must cite a
   step (the citation suppresses Omitted checkably); a "no" is necessary
   but never sufficient alone. If grounded text exceeds the window cap, the
   sweep **abstains** and Omitted stays unknown — a negative over partial
   coverage is the vacuous-closed-world bug in miniature.

`Omitted` ships only if the sweep's "no evidence" precision clears the
per-kind gate on the dev slice; fallback is restricting Omitted to
normalized machine-checkable constraints.

### Pass L — localization (pure code)

| error class | minimal bad prefix ends at | anchor |
|---|---|---|
| ungrounded / contradicted use | first reliance on the bad value | earliest use step |
| CommitError | commit without the constraint holding | commit step |
| Omitted | commit without prior verification | commit step |

Anchors are per-kind policy validated against gold label conventions on the
dev slice (P7) — if fabrication gold also proves commitment-centric, the
first row's anchor moves; nothing else changes.

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
    confidence: float = 1.0          # ordinal ranking prior, NOT a probability
    confidence_source: str = ""      # what attained the min: "code" | "oracle:<relation>"
```

Confidence is min() over contributing tuples — documented as **ordinal**
(a ranking prior for auditor attention), never probabilistic: oracle
confidences are uncalibrated self-reports and code checks are ~1.0, so the
scalar has no unit. `confidence_source` carries the lineage the scalar
cannot: the auditor weighs a code-attained min differently from an
oracle-attained one. (min() cannot express corroboration — three independent
0.6 supports score 0.6 — accepted as bounded harm since the auditor re-reads
evidence; exact confidence algebra was considered and rejected as
disproportionate at 10–30 tuples per trajectory.)

`TrajectoryIndex` gains `constraints` and `constraint_findings`; **both are
serialized in `dump()`/`load()`** (the auditor consumes a loaded index).
The oracle transcript (all tuples with their raw verdicts) is persisted
alongside, per P3. Pass E/J/L output is derived state under the same
idempotence contract as Pass 3: any alias merge or dependency rebuild
invalidates it wholesale; the passes rebuild from facts + transcript.
Pass E/J/L run strictly after all Pass 2/3 mutations.

`warnings()` stays "pure code, no LLM" as documented. Constraint findings
surface through a parallel accessor (`constraint_attention()`), merged into
the auditor's attention-hint feed by the caller.

## Review findings addressed

| finding (round 1–2) | resolution |
|---|---|
| per-symbol chains can't reach satellite-entity evidence | `About` maps steps→candidate independent of def-use connectivity |
| "unverified" unsound by construction | P5 + Omitted's conjoined double-negative with attested coverage |
| per-edge verdicts can't express relational constraints | `Entails` judged per (constraint, candidate) over the step set |
| rejected candidates trigger false violations | joins key on the committed candidate only |
| lexical filter matches entity names, not evidence | blocking on step text; every prune logged (P2) |
| adjudicator repeats the agent's date arithmetic errors | P6: normalized constraints decided by code |
| final-answer-span assumption unreliable | explicit `Commit` policy with empty-commit degradation |
| "may-analysis" label self-contradicted by lossy extraction | replaced by P2's recorded-prune invariant + dev-slice recall SLO |
| omission mislabeled liveness; localization "derived" | P7: all safety, minimal-bad-prefix anchors, validated per kind |
| Omitted incoherent under blocking (closed world via lexical heuristic) | conjoined lexical + sweep negatives; abstention on truncation |
| "LLM fills tuples, code joins" false for joint Entails | P4: bounded-context + positive-polarity replaces locality slogan |
| Datalog/IL/CEGAR semantics claimed but not held | demoted to related framings; P3 states the actual guarantee |

## Validation plan (ordered)

1. **One-call baseline first.** Question + final answer + the grounded
   tool-output snippets already in the index → a single LLM call listing
   verified / violated / unaddressed constraints with cited steps. This is
   Pass E+J collapsed into one call with the whole-evidence view — the null
   hypothesis with real prior weight given the 1.7pp margin over DRIFT. If
   the pass structure cannot beat it on a dev slice, the decomposition is
   losing information — stop. The pass structure's only standing argument
   is small-model feasibility (bounded judgments easier for an 8B model
   than a global audit), and that is an empirical claim this step tests.
2. **Dev slice** (~30 hand-labeled (constraint, evidence-steps, verdict)
   triples) to attribute failures to mapping vs judgment vs aggregation vs
   localization (P3's lineage makes the attribution mechanical). Offline
   script, not pytest.
3. **Per-kind precision gates** on the dev slice before enabling any new
   warning kind by default; the Omitted sweep gate per Pass J.
4. TELBench full run only after 1–3. Re-check the precision-limited
   assumption (P1) at the new operating point.

The impact estimate from the taxonomy (~21% of gold errors touch constraint
handling) is a **taxonomy-derived ceiling**, not an expected yield.

## Cost

- Pass 0: 1 call (question only).
- Pass E: ~2 batched calls (About; Entails/Contradicts per committed
  candidate). Machine-checkable constraints cost 0 oracle calls.
- Omitted coverage sweep: ≤1 call, only when the lexical negative already
  holds.
- Pass J/L: code.
- Upper bound +4 calls/trajectory, comparable to the existing 4
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

## Related framings (orientation, not claimed guarantees)

The design borrows orientation from program analysis; review established
which parts are load-bearing versus vocabulary, and this section keeps only
honest attributions:

- **Runtime verification** — the category fact that a finished trajectory
  is one finite trace, so the question is "which prefix violates the spec,"
  not "what may happen on any path." This correction killed the per-edge
  draft. No RV machinery (spec formalism, monitor construction) is imported.
- **Incorrectness logic** (O'Hearn) — inspiration for witness-backed
  reporting (P1). The actual guarantee here is statistical (per-kind
  precision gates), not proof-theoretic; the one region with a checkable
  entailment is P6's code-decided constraints.
- **Datalog-style analysis** (Doop) — the architectural residue survives:
  relations as the model/code interface, non-recursive rules, derived facts
  rebuilt wholesale (P3). Least-fixpoint semantics and termination are
  trivial here and not claimed.
- **Soundiness** (Livshits et al.) — the declared-unsoundness discipline
  behind P2, with the honest caveat that the dominant source (silent
  extraction miss) is undetectable by construction.
- **Safety/liveness** (Alpern–Schneider) — all error classes here are
  safety; P7's minimal-bad-prefix anchoring is the useful residue.
- **Three-valued model checking** (Bruns–Godefroid) — Kleene propagation
  with conservative unknown (P5).

## References

- SCHEMA.md — the grounding analysis this extends.
- Review transcripts: round 1 (per-edge design killed), round 2 (theory
  framing stress-tested) — in session notes.
