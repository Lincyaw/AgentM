# Generic Trajectory Analysis — Layered Architecture

Design for generalizing the trajectory analysis stack beyond QA-shaped
trajectories, produced by an architecture-design review and grounded in
SCHEMA.md, designs/constraint-satisfaction.md (contracts P1–P7), and the
current implementation. Status: proposal — Step 1 is actionable now; later
steps gate on dev-slice validation.

## Framing correction

We are not analyzing a program; we are analyzing **one finite trace of a
program we cannot see** (the agent's policy). The stack is therefore not
compiler-shaped (source → CFG → dataflow over all paths) but **decompiler +
dynamic-analysis shaped**: recover structure from a linear trace, then run
trace analyses over the recovered structure. Two consequences:

- Every recovered structure (roles, episodes, edges) is itself a *may-layer
  interpretation* under P1, never ground truth. Structural findings need
  auditor confirmation like everything else.
- There are no unexecuted branches. "What the agent could have done" is
  counterfactual and unfalsifiable from the trace; nothing in this design
  quantifies over hypothetical paths.

The stack:

```
L0  NORMALIZE   raw input → Event IR + provenance lattice     [attested roles + Pass 1 labels (code-verified) + abstention]
L1  STRUCTURE   Event IR → episode tree + typed edges         [code seeds + bounded oracle labels]
L2  DATAFLOW    (existing PARSE/RESOLVE/DATAFLOW, authority-generalized)
L3  SEMANTICS   spec sources → obligations × commitments      [generalized Pass 0/E/J/L]
L4  CROSS       dataflow ⋈ structure ⋈ commitments            [pure code joins]
```

L2 and most of L3 exist. L0 is the urgent gap (the TELBench provenance
incident proved it), L1/L4 are the genuinely new analysis class, and each
must clear a per-kind precision gate before shipping (P1).

## L0 — Front end: the Trajectory IR

### Pass layering: nodes → edges → judgments

Decided 2026-07-13. The whole stack splits by what each pass is allowed
to produce, mirroring a compiler front end:

- **Pass 1 — nodes only.** The extractor visits the trajectory once
  (incrementally, per chunk) and emits every per-message fact as a
  first-class node: symbols, claims, provenance labels — and, planned,
  commits and constraints. All nodes are verbatim-verified by code at
  populate time. O(1) per consumer: downstream passes read the index,
  never re-scan the trajectory.
- **Pass 2 — edges.** Relations between nodes (grounding claim→
  observation, about constraint→evidence, alias/anaphora). Model
  proposes, code verifies endpoint existence and direction. Never in
  Pass 1: an edge needs both endpoints to exist as nodes first, and
  cross-chunk endpoints are invisible to an incremental extractor.
- **Pass 3 — judgments.** Three-valued verdicts over code-assembled
  sets (entailment, source consistency, omission). Code contributes only
  set assembly, coverage accounting, and the Kleene join.

The division of labor everywhere: **the model does all recognition and
judgment; code does only verbatim checks, exact table lookups,
deterministic parsing, and set/lattice algebra.** No fact is ever
"extracted" by a regex or keyword table.

### Event vocabulary

Six event kinds plus an orthogonal **provenance record** on every event:

| kind | meaning | maps from current StepRole |
|---|---|---|
| `utterance` | free text produced by an agent (reasoning, claims, plans) | assistant |
| `action` | a request to the environment: tool call, command, message send | tool_call |
| `observation` | environment response bound to an action | tool_result |
| `injection` | content pushed at the agent unrequested: user message, system reminder, interrupt | user / system |
| `spawn` | delegation of a task to another agent | (multi-agent) |
| `report` | a sub-agent's return — an observation whose responder is another *agent* | (multi-agent) |

`spawn`/`report` are separate from `action`/`observation` because the
responder is non-deterministic and fallible; that difference is exactly what
the authority lattice encodes, and collapsing them is how "sub-agent said
so" gets laundered into "tool said so."

Deliberately **not** in the vocabulary: `commitment`, `verification`,
`backtrack` — semantic/structural facts *derived* by L1/L3, not lexed.

### Provenance: the load-bearing part

```python
type Actor       = Literal["agent", "subagent", "tool", "user", "harness", "unknown"]
type Authority   = Literal["observation", "attestation", "assertion", "unknown"]
type ProvQuality = Literal["native", "recovered", "unknown"]

@dataclass(frozen=True, slots=True)
class Event:
    id: str
    seq: int
    kind: EventKind
    actor: Actor
    content: str
    binds_to: str | None       # observation/report → its action/spawn
    authority: Authority
    prov_quality: ProvQuality  # how we know the provenance
    prov_basis: str | None     # oracle tuple id when recovered (lineage, P3)
```

`Authority` is a lattice ordering *usability as grounding source*:
`observation` (deterministic environment feedback) > `attestation` (a
fallible party vouched: sub-agent report, user statement) > `assertion`
(the agent's own claim) > `unknown`. The existing boolean
`Reference.grounded` is the two-point projection of this lattice —
migration is a field widening, not a rewrite.

`ProvQuality` is orthogonal: it records *how we know* the authority.
`native` = the input format carried roles; `recovered` = reconstructed;
`unknown` = abstained.

### Ingestion of provenance-degraded input

Decided 2026-07-13 (supersedes the earlier code-sniffer tier): **code
never extracts — code only verifies.** The keyword sniffer that lived in
the TELBench adapter (`classify_span`) silently missed whole trajectory
families (56% of the hard-232 set had zero recognized observations, which
starved every evidence-consuming pass downstream); that is the structural
failure mode of code doing recognition, and it applies to any future
sniffer too. The tiers are now:

1. **Attested roles win.** A structurally attested role (a real
   `tool_result` message from the substrate) is never overridden.
2. **Pass 1 provenance labels.** For unattested messages the extractor —
   the one model that visits the whole trajectory — labels
   `observation` / `mixed` messages as first-class extraction nodes,
   alongside symbols and claims. `mixed` carries a verbatim boundary;
   code verifies it by exact (prefix-laddered) substring search and
   rejects unverifiable labels with a log line (P2). A label can only
   ADD observation status to an assistant step, never remove attested
   status. (`Step.observation_segment` / `Step.action_segment` are the
   accessors every downstream pass reads; nothing reads `role` for
   evidence selection anymore.)
3. **Abstention.** Unlabeled messages stay agent action — a recorded
   default, not a preprocessing failure.

Measured (2-4-message incremental chunks, 10 TELBench cases): model
labels are a strict superset of the deleted keyword classifier's
positives on 9/10 cases (1 genuine miss), recover 0→20-40-step evidence
universes on dead cases, reject ≈0 boundaries after the
no-separator-boundary prompt rule, and are MORE precise than single-chunk
labeling on query-chain spans (single-chunk over-labeled query chains as
observations).

Position: **recovered observation-authority is usable for grounding**, but
its lineage must survive to the finding (via `confidence_source`), so the
auditor can discount a warning standing entirely on recovered provenance.
Capping recovered provenance below `observation` silently re-creates the
TELBench degeneracy. Adversarial mimicry (agent text formatted like tool
output) is mitigated by lineage + auditor discount — a declared unsoundness
in the P2-ledger sense, not prevented.

### What `unknown` propagates

- **Pointwise:** a def whose event has `authority=unknown` yields grounding
  risk `unknown` — not `ungrounded`. ("Can't tell it was tool-backed" ≠
  "fabricated".) Kleene, unknown never escalates (P5).
- **Universe-level:** every analysis quantifying over the evidence universe
  (Omitted sweep, coverage claims) reads a code-computed **provenance
  profile** — `{authority level → event fraction}` per trajectory — and
  abstains below a declared floor. The hand-added empty-universe guard in
  constraints.py is the special case; the profile makes it a uniform gate
  every global-negative rule must pass.

## L1 — Structure: reasoning "control flow"

### Not a CFG — an episode tree

A CFG describes all paths; a trace has one path. The correct object is a
**hierarchical episode tree with typed edges over the linear trace** — the
analogue of reconstructing a call tree plus loop structure from an
instruction trace. Nodes are *episodes*: sets of events (usually
contiguous) serving one recoverable intent.

```python
@dataclass(slots=True)
class Episode:
    id: str
    events: tuple[str, ...]
    parent: str | None
    intent: str          # FREE TEXT, oracle-labeled, recorded — no preset intent enum
    outcome: Literal["completed", "abandoned", "superseded", "unknown"]
    outcome_basis: str   # code rule name or oracle tuple id

@dataclass(frozen=True, slots=True)
class EpisodeEdge:
    src: str
    dst: str
    kind: Literal["decomposes", "explores", "abandons", "resumes", "verifies", "commits", "delegates", "retries"]
```

(The edge-kind Literal is analysis vocabulary like `Risk`, not a subjective
content label — the no-preset-enums rule applies to `intent`, which stays
free text.)

### Code-recoverable vs oracle-judged

| structure | recoverable by | mechanism |
|---|---|---|
| action↔observation pairing | code | call-id / `binds_to` |
| retry loops | code | same tool + high-arg-similarity + prior failure observation |
| delegation boundaries | code | spawn/report events |
| turn/phase boundaries | code | injection events segment the trace |
| error observations | mostly code | structured error shapes; oracle for ambiguous text |
| episode intent | oracle | bounded window over the episode's events, positive polarity |
| abandonment | oracle | code proposes the boundary from lexical/entity signals, oracle confirms locally |
| backtrack target | oracle | coreference-style: code blocks candidate episodes, oracle picks |

Same division of labor as Pass 2 alias/coref: code blocks candidates, the
oracle decides each locally with recorded tuples, code assembles the global
tree. The tree is derived state under the same wholesale-rebuild idempotence
contract as Pass 3.

### What becomes checkable (real error classes current layers miss)

1. **Commit with no verify edge** — structural generalization of Omitted;
   needs only the tree, no content sweep. Weaker (verification may leave no
   recognizable episode), so a may-candidate feeding the auditor, never a
   standalone alarm.
2. **Use-after-abandon** — see L4.
3. **Retry without repair** — a `retries` cycle with unchanged args and
   state (dataflow supplies "unchanged"). The classic agent insanity loop;
   pure code once the loop is recovered.
4. **Silent readoption** — the committed candidate's exploration episode
   ended `abandoned` with no `resumes` edge: the agent talked itself out of
   the answer and answered it anyway. Invisible to grounding and to
   constraints.
5. **Trust laundering** — a `report`'s content used as if
   observation-authority with no grounding inside the delegation. Falls out
   of the authority lattice plus the delegation edge.

## L3 — Semantics: generalizing the QA constraint layer

### Domain-generic residue (survives unchanged)

- Positive-polarity establishment over code-selected whole-step windows (P4).
- **Commitment as join key and localization anchor** — "violations by
  non-committed candidates are exploration" generalizes verbatim:
  violations inside abandoned episodes are exploration; only what reaches a
  commitment point can be an error.
- The Omitted double-negative, with the universe re-based on
  `authority ≥ observation` events and gated on the provenance profile.
- Kleene joins, unknown never escalates, transcript determinism.

### QA-specific parts needing replacement

Pass 0 reads a *question* (spec source is domain-bound); `Commit` = "last
assistant step asserting an answer-class entity" (commitment kind is
domain-bound); `About` keyed on entity mention (evidence-relevance key is
domain-bound).

### The generic abstraction

```python
@dataclass(frozen=True, slots=True)
class Obligation:
    id: str
    source: Literal["task", "tool_schema", "invariant", "acquired"]
    scope: str            # binding rule: "termination" | "action:<tool>" | "global" | "commitment"
    description: str      # free text
    normalized: Mapping[str, Any] | None   # only when code-decidable (P6)

@dataclass(frozen=True, slots=True)
class CommitmentPoint:
    event_id: str
    kind: str             # "answer" | "irreversible_action" | "external_message" | "delegation" | "termination"
    binding: str          # what is committed: the candidate entity, the action instance, the sub-task
    basis: str            # code rule (tool-schema flag) or oracle tuple id
```

**Spec sources**, decreasing extraction reliability: tool schemas
(machine-readable preconditions and irreversibility flags — *free*
obligations, zero oracle calls); declared environment invariants; task text
(existing Pass 0, per-domain prompt); **acquired obligations** — constraints
arriving mid-trajectory inside observations. The last is the genuinely new
agentic requirement and the hardest; explicitly deferred.

**Commitment kinds**: QA has one commitment at trace end; agentic tasks
have **many interleaved commitments** — every irreversible action is a
localization anchor. Obligations carry `scope` saying which commitment
points they bind to.

**Localization generalizes cleanly** (P7): each (obligation,
commitment-instance) pair is a safety property on the finite trace; the
minimal bad prefix ends at that commitment instance. On a finite completed
trace, "eventually achieve G" is decidable at the terminal event — even
success criteria collapse to safety-at-termination. P7's all-safety framing
is a property of analyzing completed traces, not an accident of QA.

Findings keep the `ConstraintFinding` shape with `(constraint_id,
candidate)` → `(obligation_id, commitment.binding)`. `About` becomes
`Relevant(step, binding)` — same blocking-plus-oracle machinery.

## L4 — Dataflow × control flow: the missing analysis class

The existing def-use taint is **scope-blind**: it knows a value was
tool-backed but not *in what deliberative context*. Fix: **tag every def
with its episode id, check the episode's status at each use** — dynamic
taint with scope tags; a table join, not a new engine.

1. **Grounded-in-abandoned-context** → new risk `orphaned`: value grounded
   inside a later-abandoned episode, used after the `abandons` edge.
   Flagship L4 check.
2. **Stale-after-interference**: grounded read → later action writes the
   same resource → use of the old value. Needs per-tool **declared effects**
   (`reads`/`writes`, `irreversible`) — declaration-only, no inference.
   Domain-gated.
3. **Verification-scope mismatch**: evidence judged inside
   episode-of-candidate-B credited toward a commitment binding candidate A.
4. **Retry-without-repair**: loop from structure, "nothing changed" from
   dataflow.

All pure code over recorded facts; weakness inherited entirely from episode
recovery accuracy — hence L1 must be precision-gated before L4 ships.

## What NOT to build

- No fixpoint / abstract-interpretation machinery (one finite trace;
  non-recursive joins).
- No branch/path analysis over hypothetical paths (unfalsifiable from the
  trace) — kills counterfactual "should have explored X" checks.
- No grammar/parser formalism for L1: episode boundaries are a labeling
  problem, not a parsing problem; a grammar fabricates crisp structure where
  input is soft.
- No type system over reasoning steps (preset enum over subjective content).
- No general heap/alias model of the environment — declared per-tool effects
  only; inferring interference is the frame problem.
- No liveness monitors / LTL: completed traces make everything
  safety-or-decidable-at-termination.
- No trajectory rewriting/optimization. Analysis only.

Where the PL analogy actively misleads: (i) "CFG" implies branches are in
the artifact — here they are lossy reconstructions of deliberation; (ii)
"taint" implies binary trust — the four-point authority lattice with
first-class `unknown` is the actual model, and the binary version caused the
TELBench degeneracy; (iii) a compiler *rejects* malformed input — this front
end must never reject, only degrade to unknown; (iv) SSA implies
deterministic memory — the environment is external; versions are heuristic
ordering, not semantics.

## Incremental path (by value/cost)

1. **L0 provenance normalization + authority lattice.** Ingestion module
   (sniffers → oracle role labels → abstention); widen `Reference.grounded`
   to the lattice (boolean stays as projection); provenance-profile gate
   replaces the hand-added empty-universe guard. *Fix measured:* TELBench
   tool_result recovery 0 → measurable; "every entity fabricated" becomes
   per-event unknown; the false-omitted class prevented structurally.
2. **Commitment generalization (L3-lite).** `CommitmentPoint` kinds +
   per-tool irreversibility declarations (name-heuristics as declared-
   unsound stopgap). Lifts the AFTraj gate on the existing constraint
   machinery with a domain Pass 0 prompt.
3. **L1 minimal episode recovery** — code-recoverable structure first
   (pairing, retry loops, delegation), then one oracle pass for
   exploration/abandonment labels. Build a ~30-item labeled dev slice for
   episode boundaries BEFORE anything consumes the tree.
4. **L4 joins** — ship exactly two checks first: use-after-abandon
   (`orphaned`) and retry-without-repair. Pure code, gated per kind.
5. **Interference/effects extension** — needs per-domain effect curation.

Each step follows the QA layer's validation discipline including the
one-call-baseline null hypothesis: before building L1's pass structure,
check whether a single whole-trace LLM call labels abandonment/commitment
well enough — if the decomposition cannot beat it on the dev slice, stop.

## Open problems

1. **Episode recovery is the load-bearing unknown.** Its silent-miss profile
   is the same P2 hole as entity extraction. Mitigation: dev slice + gates.
   L4's value is strictly bounded by it.
2. **Acquired obligations** (mid-trajectory, from observations): unbounded
   extraction surface; deferred explicitly.
3. **Effect declarations need an owner**: tool schemas rarely state
   irreversibility; heuristics are declared unsoundness until scenarios
   curate them.
4. **Multi-agent recursion.** V1 treats `report`s as attestation-authority
   atoms; recursing the analyzer into child trajectories is deferred — the
   composition semantics (does a child's grounded fact become the parent's
   observation or attestation?) is genuinely open.
5. **Adversarial provenance recovery**: mimicry is mitigated by lineage +
   auditor discount, not prevented.
6. **Anchor conventions stay empirical** (P7): each benchmark's label
   convention validated per kind before its anchors are trusted.

## References

- SCHEMA.md — existing grounding analysis (L2).
- designs/constraint-satisfaction.md — contracts P1–P7; QA instance of L3.
- src/trajectory_index/index.py — StepRole/Step; `Reference.grounded`
  (the field L0 widens).
- src/trajectory_index/constraints.py — empty-universe guard (the special
  case the provenance profile generalizes).
- agentm_eval/benchmarks/telbench/adapter.py — deterministic format
  conversion only; its keyword `classify_span` sniffer (the motivating
  provenance incident) was deleted in favor of Pass 1 labels.
