# Feedback Dimensions for Runtime Critique

Status: design note, agreed direction as of 2026-07-23.

Relationship to other documents:

- `docs/policy-anomaly-detection.md` keeps the two-stage mechanics (cheap
  triggers, heavyweight critic, suppression). This note supplies the model
  that decides *what a critic checks* and *why those checks exist*.
- The earlier observability note (`policy-trajectory-failure-observability.md`,
  removed 2026-07-23) is fully absorbed: its causal taxonomy by the dimension
  model here, its structural signal catalog by the feature set in
  `docs/policy-anomaly-detection.md`.
- Grounding evidence: the seven root-cause analyses of the failed
  `jobs/2026-07-21__22-27-02` calibration tasks (electric, gitea, turborepo,
  paperless, firezone, harbor, plausible), each correlated against oracle
  patches and per-test verifier results.

## 1. Problem

The policy layer sits between a coding agent and its environment. It watches
the live trajectory and, at the right moment, either recommends that the agent
reflect on a specific aspect of its own work, or spawns a critic subagent to
review that aspect. It never decides for the agent and it never reveals
answers. Offline, with full `(task, trajectory, eval result)` triples
available, a strong model mines failure patterns; online, with no ground
truth, a routing layer matches the live `(task, trajectory)` against those
patterns and triggers the corresponding critique. The overall shape is a
semi-automatic recommendation system: "given what you are doing right now, it
is worth re-checking X".

That design needs an answer to one question before anything else: **what are
the dimensions along which a critic can give feedback?** An enumerated list of
observed failures does not answer it; every new batch would grow the list. We
need a generator: a principle that produces the dimension set and tells us
when it is complete.

## 2. The generator: the solving process as a chain of representations

An agent working a task carries the problem through a series of intermediate
representations:

```text
task description
  -> understanding        what it believes the task asks
  -> diagnosis            what it believes about the code / world
  -> plan                 what it decides to do
  -> change               what it actually edits
  -> evidence             what its validations actually observed
  -> belief               what it concludes about the current state
  -> claim / stop         what it reports and when it stops
```

Two observations turn this chain into the dimension generator:

1. **Every failure is a mismatch between two stages of the chain.** A wrong
   solution that was implemented faithfully, validated thoroughly within its
   own frame, and reported honestly still failed somewhere: the mismatch is
   between the task and the understanding. A correct understanding with dead
   code is a mismatch between plan and change. Green tests that could not
   have caught the bug are a mismatch between change and evidence.
2. **Every feedback dimension is a checkable pair of stages.** A critic picks
   one pair, measures their consistency, and feeds the error back before the
   loop closes. This is what makes the system a negative-feedback loop rather
   than an anomaly detector: the router does not need to detect failure, it
   only needs to recognize a situation in which checking a particular pair is
   worth the cost. The semantic judgment belongs to the critic, which works
   from the task text, the trajectory, and optionally its own probing
   actions. None of that requires ground truth.

## 3. The dimensions

Six chain dimensions, one time-domain axis, and one gate that precedes all of
them.

### D1 Interpretation fidelity (task <-> understanding)

Did the agent's reading of the task preserve its constraints, directions, and
literal details? The critic re-derives the requirements from the task text
alone, blind to the agent's reading, and diffs the two.

Evidence. Gitea: the task requires `added-code` to wrap the syntax span from
the outside; the agent implemented and tested the nested-inside direction,
consistently and confidently. Harbor (the genuine secondary bug, not the
recorded score): the task states exact retained-filename patterns; the agent
substituted a broader glob that prunes orphan `trajectory.cont-N.json` files.

### D2 Requirement coverage (understanding -> plan/change)

Does every requirement clause map to some part of the change, and were all
sibling sites of a multi-site change addressed? The critic builds a checklist
from the task text and maps clauses to hunks; for sibling completeness it
sweeps for analogous call sites of the pattern the change touches.

Evidence. Paperless: the optimization applies to three count sites; the agent
changed one and left the tag-hierarchy and default-mixin paths on the old
JOIN.

### D3 Implementation fidelity (claimed behavior <-> actual code semantics)

Does the code actually do what the agent says it does? New branches must be
reachable; guards must be satisfiable in the scenario they target. The critic
traces reachability and control flow. This is pure code reasoning against the
agent's own stated intent; no oracle involved.

Evidence. Electric: cache-buster emission and bounded-retry error were
implemented behind `is_nil(state.shape_handle)`, which is always false after
a 409 refetch, making the entire mechanism dead code in exactly the scenario
it targets. The agent's summary asserted the behavior works; its own test
asserted `cache-buster == nil`, green-lighting the dead path. The rubric
judge, reading statically, scored the behavior as implemented. A reachability
critic catches this without any ground truth.

### D4 Diagnosis validity (claimed cause <-> collected evidence)

Does the evidence gathered in the trajectory actually entail the agent's
causal diagnosis, and were alternative explanations excluded? The critic
re-derives the diagnosis from the evidence already in the trajectory and
checks the exclusion structure.

No clean sample in the current seven cases. Retained because it satisfies the
admission criterion: a distinct comparand pair and a critic procedure that
needs no ground truth.

### D5 Evidence adequacy (change <-> executed validation)

If the change were wrong, would any executed validation have failed? This is
the falsification-pressure question, with three sub-forms observed so far:
breadth (was the mutated scope ever exercised unfiltered), independence (did
anything not authored this session gate the conclusion), and dimension match
(was the graded quality, such as latency, ever measured at all).

Evidence. Turborepo: five `cargo test -p turborepo-scm` runs, every one
narrowed by a positional selector; the unfiltered `--lib` suite containing
the three pre-existing regression tests never ran. Paperless: latency was
never measured once despite a perf task; the two failing verifier checks are
stable perf budgets (1.80s vs 1.5s, 0.288s vs 0.25s). Electric: the one
discriminating suite was only ever green through a self-authored line
selector (`client_test.exs:917`).

### D6 Belief-evidence consistency (evidence <-> belief/claim)

Were observed outcomes absorbed into the agent's conclusions? Red results
must not be ignored; claims must not exceed evidence. Largely checkable by
structural rules alone.

Evidence. Firezone: the final `mix test` at turn 118 exited 1; the agent then
ran only formatting and git inspection and submitted. The measurement was
correct; the judgment discarded it.

### D7 Loop dynamics (the chain over time)

The six dimensions above are consistency checks on the chain at a moment.
D7 is a different kind of axis: convergence of the loop itself. Stagnation,
A -> B -> A oscillation, repeated identical failures without an intervening
change, effort concentration far past diminishing returns, and premature
stopping all live here. Any chain state, healthy or not, can exhibit bad
dynamics, which is why this axis is orthogonal to all six in the strongest
sense: it measures the time domain, they measure the state domain.

### G0 Measurement validity gate (precedes everything)

Not a dimension of agent behavior: a validity check on our own scoring
instrument. If the verifier environment or judge is broken, the recorded
label carries no information about the agent and no dimension may be scored
from it.

Evidence. Three of the seven recorded zeros in the calibration batch were
artifacts: harbor (verifier image missing the `daytona` dependency, test
collection crashed; the re-run passes 2/2), firezone and plausible (judge
model emitted malformed forced-tool-call JSON, `reward.txt` written empty,
reward defaulted to 0). Plausible's true outcome was 2 of 3 stories passing.
Labels must pass this gate before any offline mining consumes them.

## 4. Boundary definitions

The two pairs most likely to blur, pinned down:

- **D2 vs D5.** Coverage maps requirement clauses to *changes* (is there code
  implementing the clause). Adequacy maps changes to *evidence* (is there a
  validation that could catch the change being wrong). Paperless separates
  cleanly under this rule: the one-of-three count sites is D2; the
  never-measured latency is D5.
- **D1 vs D4.** Both are "the agent's model is wrong", but the comparand
  differs: D1 compares against the task text, D4 against the codebase and
  the collected runtime evidence.

## 5. Cascades, shadows, and root attribution

The dimensions are not statistically independent, and are not meant to be. An
upstream fault propagates: gitea's D1 misreading produced self-authored tests
encoding the same misreading, which lights D5. The chain ordering supplies
the attribution rule:

> Attribute a failure to the most upstream dimension that fails; downstream
> firings are shadows of it.

A shadow is still useful online (gitea's "every gating test was authored this
session" is observable and is the correct trigger for a D1 critique), but the
mined pattern and the critique content should name the root, not the shadow.

Orthogonality therefore holds in the sense that matters and fails in the
sense that does not:

- **As check targets, the dimensions are pairwise distinct.** Different
  comparand pairs, different critic inputs, different procedures. Each has a
  demonstrated or constructible single-point failure: gitea fails only D1
  (its implementation was faithful to its misreading, its validation rigorous
  within it), electric's dead guard fails D3 regardless of understanding,
  turborepo fails only D5, firezone only D6.
- **As failure events, they cascade by construction**, and the root
  attribution rule absorbs that.

## 6. Relationship to AdaMAST (arXiv 2607.16387)

AdaMAST induces a compact vocabulary of named failure codes and reuses it as
a shared interface for improvement procedures. Its three fixed axes partition
codes **by intervention point**: A system-level (repair the harness), B
role-specific (rewire a role), C domain-specific (inject task knowledge). The
consumer is a system developer.

Our dimensions partition **by error locus in the solving chain**, because our
consumer is a runtime critic that must know *what to re-check*, not who owns
the fix. The two partitions are complementary coordinates, not competitors: a
concrete failure code needs both (what to check x how to intervene).

Mapping: their A axis corresponds to our G0 gate; their B axis collapses for
a single-agent system; their C axis is not a dimension here but a *reference
source* several dimensions draw on (domain practices such as "user-controlled
input reflected into a redirect must be exercised adversarially"). The chain
itself has no counterpart in their taxonomy; it exists because our unit of
improvement is an in-flight solve rather than a system configuration.

Directly borrowed ideas:

- **Codes are induced inside fixed axes.** Offline mining produces named
  codes as model-and-task-type-specific instantiations of a dimension, e.g.
  "on invariant-rendering tasks this model tends to invert directions" is a
  D1 code with its own trigger signature. Our offline miner may use
  ground-truth eval results, which their induction deliberately avoids; our
  setting is a strict superset.
- **Agreement as the acceptance test.** A mined code, and this dimension cut
  itself, is real only if independent blind annotators (or per-dimension
  critics) re-detect it consistently. This is the future empirical test of
  the claimed separability.
- **The nudge names the failure mode, never the fix**, with a bounded repair
  loop and honest reporting of unresolved issues.

## 7. Runtime architecture: recommendation, not detection

Per dimension, three sensor tiers with increasing cost:

1. **Structural rules** (deterministic, trajectory-only): D6 shipped-on-red;
   D5 breadth (a mutated named build target whose unfiltered suite never
   ran); D7 hash cycles and stagnation windows. These are the stage-1
   triggers of `docs/policy-anomaly-detection.md`.
2. **LLM read** (task text plus a trajectory slice, no tools): D1
   re-derivation of requirements, D2 checklist mapping, D3 reachability
   reasoning, D4 evidence-entailment checks.
3. **Active probing critic** (tools, may act on the environment): re-derive
   the invariant direction from the task and check the actual output; run
   the unfiltered suite; send the adversarial request. The critic is itself
   a sensor that creates new, independent measurements. This dissolves most
   of the "indistinguishability boundary" argued in the observability note:
   a wrong-direction implementation is indistinguishable from a correct one
   *in the passive trace*, but not to a critic that can probe.

The DSL is the routing layer between them: trigger conditions over the live
`(task, trajectory)`, learned offline, that map a situation to a dimension
and a sensor tier, then either recommend reflection (naming the dimension and
the structural facts that triggered it) or spawn the critic. Injection
follows the context-injection cache discipline (append to the tail of the
last message).

Hard constraints carried over from the two-stage design:

- No ground truth online. Ever.
- The nudge carries the pattern (name plus structural evidence), never the
  answer. Practice-class content ("changes on a read-hot path usually have a
  latency budget; measure it") is legitimate; task-specific criteria (the
  1.5s threshold, "strip return_to") are leaks.
- Budgeted and suppressible: critics cost tokens and injections cost cache;
  false triggers are not free.

## 8. Honest limits

- **Requirements that never surface anywhere the agent can see** (plausible's
  hidden third story: reject client-supplied `return_to`) produce no
  chain mismatch at all: the trajectory is healthy, the tests self-consistent,
  the stop clean. Only domain-practice priors (a C-style reference source,
  offline-mined and keyed by requirement class of the task type, not by model
  identity) can even warn here, and only in practice-class terms.
- **D4 is currently unexercised** by the seven cases; it stands on the
  admission criterion only.
- **D7 detectors exist but their discrimination is unproven**; the earlier
  rule audit showed complexity-correlated signals (repeated-region-reading)
  firing on passes and failures alike. Dynamics signals must clear the same
  bar as everything else: hit failed sessions, stay quiet on passes.

## 9. Acceptance criteria for this cut

- **Admission:** a dimension exists only if it names a comparand pair and a
  critic procedure that needs no ground truth. All seven pass; G0 is
  deliberately outside (it validates the label, not the agent).
- **Separability (future, empirical):** blind per-dimension critics over the
  calibration corpus; a dimension survives if its critic fires on the cases
  attributed to it and stays quiet on confirmed passes, with
  inter-annotator agreement at the level AdaMAST uses to accept codes.
- **Actionability:** a triggered critique must name a concrete re-check the
  agent can perform. "Re-derive the required direction from the task text
  and compare with your implementation" is actionable; "be careful" is not.
