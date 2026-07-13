# Trajectory index — formal model

The index is a static-analysis stack over one recorded agent trajectory.
This file defines the objects each pass produces and consumes; the
governing contracts (P1–P7) live in `designs/constraint-satisfaction.md`,
the layered-architecture rationale in
`designs/trajectory-analysis-architecture.md`.

The division of labor everywhere: **the model does all recognition and
local pairwise judgment; code does only the decidable** — verbatim
checks, exact lookups, deterministic parsing and partitioning, set and
lattice algebra. No fact is ever extracted by a regex or keyword table.

## 0 · The trajectory

A trajectory is a finite sequence of steps

$$T = \langle s_1, \dots, s_N \rangle, \qquad s_i = (i, r_i, x_i)$$

where $i$ is the position, $x_i$ a text (the step content, rebuilt
deterministically from the recorded message blocks), and the role
$r_i \in \{\mathrm{user}, \mathrm{assistant}, \mathrm{tool\_call},
\mathrm{tool\_result}, \mathrm{system}\}$. A role is **attested** when
the record's structure carries it (a real harness message); degraded
serializations (benchmark record dumps) arrive with uninformative roles.

**Authorship labeling.** Every character of a step's content has exactly
one author — the agent wrote it, or the environment produced it
(retrieved page, tool output). Formally this is a labeling function over
character positions (the sequence-labeling view familiar from NLP text
annotation):

$$\mathrm{auth}_i : \{0, \dots, |x_i|-1\} \to \{\mathrm{agent}, \mathrm{env}\}$$

The **observation regions** $O_i$ are the maximal runs of env-labeled
positions, written as half-open character intervals $[a, b)$ (the
standard convention of Python slices and NLP standoff annotation); the agent-authored remainder is $A_i$. Because
$\mathrm{auth}_i$ is a total function, the two sides partition the step
by construction: no character has two authors, none has zero.

Three shapes are the degenerate and general cases of the same
definition: a step that is wholly environment output ($O_i$ covers
everything, e.g. an attested $\mathrm{tool\_result}$), a pure agent step
($O_i = \varnothing$), and an interleaved query/content/summary sandwich
(several disjoint intervals).

The derived **evidence space** is

$$E = \{\, e_i = (i, O_i) \;:\; O_i \neq \varnothing \,\}$$

and everything outside it is agent action text.

## 1 · Pass 1 — nodes (one trajectory visit)

**Purpose.** Every downstream question (*does the evidence contradict
the agent's conclusion? which entities lack a source?*) presupposes
facts the raw text does not carry: what is evidence ($\mathrm{auth}_i$,
hence $E$), what was asserted ($C$), what the names are ($\Sigma$).
Pass 1 establishes these ground facts **once**:

- it is the only pass that reads the trajectory text — everything later
  queries the node tables instead of re-reading it;
- recognition is a model's job exactly here and nowhere else —
  authorship/assertion/naming are semantic (keyword recognizers failed
  silently at scale); after Pass 1, exact lookups suffice;
- it fills only what the record lacks: an attested
  $\mathrm{tool\_result}$ step has $O_i = \{[0, |x_i|)\}$ by structure;
  recognition can only ADD observation status, never override an
  attested role.

**The annotation atom.** Every Pass 1 fact is one triple

$$\text{atom} = (\text{region},\ \text{dimension},\ \text{label})$$

a text region $[a,b)$ of a step, the dimension of the question being
answered about it, and that question's answer. There are three
orthogonal dimensions — the same region can carry an atom in each, but
at most one label per dimension:

| dimension | question | label space | today's node |
|---|---|---|---|
| source | who produced this text? | agent, env (generalizes to the authority lattice: subagents, users, harness) | `obs` = the env value; unlabeled = agent |
| stance | what is this utterance doing? | assert (claim; `role=commit` marks the final answer), require (constraint), and later: assume, intend, retract, … | `claim`, `constraint` |
| reference | what does this text point at? | entities (name × kind × class) | `sym` |

The source dimension is a total labeling (every character has exactly
one author); stance and reference are sparse region annotations. The
current three node kinds are the minimal configuration of this schema —
one value per dimension, each with a downstream consumer. The extension
rule follows: a new node kind must be a new label on one of these
dimensions (constraint and commit landed exactly this way — one new tag / one
attribute; premise is next), and anything that does not fit — "these five
steps form an episode", "this claim retracts that one" — is a relation
BETWEEN regions and belongs to Pass 2 edges or the structure layer,
never to inline annotation.

**Mechanism.** The extractor visits $T$ once, in chunks of a few steps,
re-emitting each annotated step body **verbatim** with
`⟦tag attrs|content⟧` annotations inserted. Recognition is not taken on faith
— verification is strip-and-compare: removing every annotation must reproduce
the exact text the extractor was shown (whitespace-tolerant), making
every annotation offset exact; a diverging re-emission rejects that step's
annotations whole, into a recorded prune log. Three node kinds:

| node | tag | definition | verification |
|---|---|---|---|
| provenance | `⟦obs\|…⟧` | the intervals of $O_i$ | strip-and-compare → exact offsets; overlaps merged; attested roles never overridden |
| claim | `⟦claim\|…⟧` | $c = (i, [a,b))$ — a sentence the agent asserts **as settled fact**: verification statements, conclusions/identifications, settled negative findings; plans, questions, hedges excluded (the test is stance, not wording or polarity) | text is the content slice $x_i[a{:}b]$ — verbatim by construction |
| constraint | `⟦constraint …\|…⟧` | a task requirement, in task text only (stance: require); machine-checkable comparisons carry typed attrs (`kind=year_range lo hi`, `kind=number op value`) | text verbatim by construction; attrs validated by code, unparseable attrs degrade to a semantic constraint, logged |
| symbol | `⟦sym …\|…⟧` | $\sigma = (\mathrm{name}, \mathrm{kind}, \mathrm{class})$, declared at first mention; other marked surfaces of the same name become aliases automatically | occurrences located by exact name/alias matching; an occurrence inside $O$ is a grounded (tool-backed) def |

The claim set is $C = \{c_1, \dots, c_K\}$; the symbol table $\Sigma$
carries the occurrence relation
$\mathrm{occ} \subseteq \Sigma \times \mathrm{locations}$. The class is
the name/value axis, independent of kind: *could a tool report different
content for this while it stays the same thing?* yes → value, no →
identifier, vague/anaphoric → unknown. Nodes only — no relations, no
verdicts: a chunked extractor cannot see both endpoints of a cross-step
relation.

Cost note: gap elision (below, §"the annotation atom" table aside)
keeps re-emission output proportional to annotation density rather than
text size, which is what lets the extractor see steps IN FULL — there is
no prompt-window truncation, and every character is annotatable.

## 2 · Pass 2 — edges (relations between nodes)

Pass 2 consumes only the Pass 1 node tables — it never re-reads the
trajectory text. This pins its ceiling up front: every conclusion below
is relative to the **indexed** evidence space ($E$ always means the
indexed one), and a retrieved region Pass 1 failed to label is invisible
to every sweep — no negative can be stronger than Pass 1's recall of
$O$.

### 2.1 Object: typed relations over nodes

Pass 2 populates one labeled relation table over Pass 1 nodes:

$$R \subseteq V \times V \times L, \qquad V = C \cup E \cup \Sigma$$

Evidence edges are one label family; identity edges (`same_as`, an
equivalence over $\Sigma$'s surface forms, §2.7) are another; `about`
(relevance) is a reserved label — named, not yet specified. The
extension rule mirrors Pass 1's: anything that relates two regions is a
new label here, never a new inline annotation.

**Evidence edges**:

$$\mathrm{evd} \subseteq C \times E \times \{\mathrm{supports}, \mathrm{conflicts}\}$$

Scope: an evidence edge relates nodes of ONE trajectory — $c$ and $e_j$
always share the run. The symbol table $\Sigma$ is currently global
across runs (a `same_as` edge may therefore cross trajectories); until
$\Sigma$ is run-scoped this is a stated consequence, not a feature —
cross-run coreference is out of scope.

### 2.2 Judgment: pairwise textual entailment

The unit judgment is textual entailment over one pair — hypothesis the
claim text, label $\in \{\mathrm{entail}, \mathrm{contradict},
\mathrm{neutral}\}$; entail/contradict become an edge, neutral becomes
nothing. The premise needs care: $O_j$ is the retrieval POOL
(document-scale, unbounded); the judged premise is the excerpt
verification narrows to. The pass as a whole is FEVER-shaped
(retrieval / verification / aggregation), in two model stages with code
around them:

- **retrieval** (high recall): code partitions $E$ deterministically
  into whole-step groups $P_1, \dots, P_J$ under a character budget;
  per partition, the model nominates candidate evidence steps for every
  claim — no polarity, no quotes, an easy listing task. Recall lives at
  this stage, which is why sampling does too: samples are unioned.
- **verification** (high precision): per claim, one focused call over
  its nominated excerpts only, judging polarity and copying the decisive
  passage; small context, adversarially phrased ("try to refute this
  pairing").

A step whose observation content alone exceeds the budget is split at
its observation-REGION boundaries (a sandwich step's $O_i$ intervals are
the natural sub-units — no mid-content cut). If a single region still
exceeds what one call can faithfully see, that partition is marked
coverage-degraded rather than swept: a "successful" call over content
the model may not actually have attended to must not entitle a negative.

Single-shot batch judgment (all claims × a 60K-char partition in one
generation) is the degenerate one-stage form; implicit $|C| \times |P|$
matching inside one generation underreports systematically, which is why
the pipeline splits — the same reason FEVER separates sentence selection
from claim verification.

### 2.3 Certificates: the model proposes, code verifies

Every proposed edge carries a witness — a quote $q$ claimed to occur in
$O_j$. The trust boundary first: **a certificate attests that the quote
exists, not that it entails or contradicts the claim** — relevance and
polarity remain model judgments. This is exactly where precision can
leak (a verbatim but non-probative quote), and why verification is a
focused per-pair judgment rather than a side effect of retrieval.

What code checks is the decidable fragment: endpoints exist, label
valid, FULL-quote verbatim containment within a single observation
region (whitespace-normalized; not a prefix match, not spliced across
region seams), and $q$ is not the claim's own words. Stored edges
$\subseteq$ certificate-valid proposals; duplicate proposals (same
claim, step, label) collapse to the first verified one. Proposals may be
sampled freely — a sample can only ADD candidates; assertions pass only
through the verifier (proof-carrying proposals).

A stated limitation of the self-quote gate: it is verbatim containment
only. Paraphrase circularity (an observation restating the agent's
words) and env-laundered agent text (the agent writes a file and reads
it back — authorship correctly labels the read as env) pass the
decidable gate and remain model-side precision leaks.

### 2.4 Coverage: what entitles a negative

"No evidence for $c$" is negation as failure — sound only under a
closed-world condition, and the condition is PER CLAIM. Three parts, all
required for claim $c$:

1. **content coverage** — the partitions exhaust the space:
   $\bigcup_{j} P_j = E$ and no partition is coverage-degraded (§2.2);
   ($E = \varnothing$ makes this vacuously true — the
   $\mathrm{evidence\_empty}$ flag of §3 marks that degenerate sweep);
2. **attention coverage** — every partition's retrieval carries an
   explicit row for $c$, attested by at least one successful sample of
   that partition; an empty candidate list is a decision, not an
   omission;
3. **judgment coverage** — every pair retrieval nominated for $c$ was
   decided by a verification call. A failed verification demotes $c$ —
   and only $c$ — to unknown, never to unsourced; nominated-but-
   unverified candidates are never stored as edges (retrieval carries no
   polarity).

Content coverage alone (the excerpts were in the prompt) attests nothing
about attention; the retrieval row is the attestation of attention; the
verification decision is the attestation of judgment. Even with all
three, `unsourced` is sound only RELATIVE TO RETRIEVAL RECALL — it is an
attested-attention negative, not an attested-judgment one: a nomination
retrieval missed converts directly into a false unsourced. This mirrors
the Pass 1 recall ceiling stated at the top of §2, one layer down; it is
also why sampling lives at the retrieval stage.

### 2.5 Time is a fact, not a filter

Each kept edge records the timeline fact

$$\mathrm{position}(c, e_j) = \mathrm{sign}(j - i) \in \{\mathrm{before}, \mathrm{same}, \mathrm{after}\}, \qquad c = (i, [a,b))$$

(negative sign = the evidence step precedes the claim's = before; when
the claim's step cannot be located, position is recorded as unknown,
never guessed). Never a filter: consistency is time-agnostic, diagnosis
is time-sensitive — before-supports reads as grounded progress,
after-conflicts as "committed early, refuted later, never retracted".
Both are signal. One accepted loss: `same` collapses within-step order
(the stored claim carries its step, not its character interval) —
downstream must read co-location, nothing more, into it.

### 2.6 Failure monotonicity

The four statuses are totally ordered by information:

$$\mathrm{unknown} \;<\; \mathrm{unsourced} \;<\; \mathrm{supported} \;<\; \mathrm{conflicted}$$

Each step up requires strictly more attested input — coverage, then a
verified supports certificate, then a verified conflicts certificate;
conflicted dominating is "more information", not "worse news". The
monotonicity claim is then precise: $\mathrm{status}(c)$ (the §3 fold)
is a monotone function of $(\text{verified edges at } c,\ \text{coverage
bits of } c)$, and every failure SHRINKS those inputs — a failed oracle
call removes coverage (→ unknown), a rejected certificate removes one
proposal, a retrieval miss removes a nomination. So every failure moves
status down this order, never up. Sampling is monotone on the candidate
side and gated on the assertion side, so repetition is safe.

The order is informational, not evaluative. Read as alarms: a failure
can UNDER-alarm (a missed conflicts edge leaves a wrong claim looking
supported) but never OVER-alarm (no failure path fabricates a conflict).
The fail-stop guarantee is the second direction only.

This safety story is scoped to evidence edges. It does NOT extend to
identity edges, whose failure direction is inverted (§2.7).

### 2.7 Identity edges

`same_as`: alias/coreference resolution — an equivalence over $\Sigma$'s
surface forms; code blocks candidate pairs, the model judges each
locally, code merges and rewrites anaphors.

Identity sits OUTSIDE the certificate discipline: no decidable witness
exists for "these two surfaces name one thing" beyond endpoint
existence. And the merge is a union-find closure — monotone toward MORE
identification: repetition adds merges rather than being safe, and one
false `same_as` contaminates its whole equivalence class through
transitivity. The gate is therefore conservatism instead of
certificates: block candidate pairs aggressively, judge each pair
independently, and prefer a missed merge (two nodes for one entity —
recoverable) over a false one (one node for two entities —
contaminating).

## 3 · Pass 3 — judgments (folds over nodes + edges)

**Claim status** (pure code, zero model calls):

$$\mathrm{status}(c) =
\begin{cases}
\mathrm{conflicted} & \exists\, e : (c, e, \mathrm{conflicts}) \in \mathrm{evd} \quad \text{(dominates)}\\
\mathrm{supported} & \text{else } \exists\, e : (c, e, \mathrm{supports}) \in \mathrm{evd}\\
\mathrm{unsourced} & \text{else if coverage complete} \quad \text{(attested negative)}\\
\mathrm{unknown} & \text{otherwise} \quad \text{(never escalates)}
\end{cases}$$

Coverage is complete in the per-claim §2.4 sense — content, attention,
and judgment coverage all holding for that claim; a verification failure
demotes exactly the claims it touched to unknown. The flag
$\mathrm{evidence\_empty} \iff E = \varnothing$ distinguishes "swept
$E$ and found nothing" from "the record carries no observation content
at all" — itself a strong trajectory-level fact.

The vocabulary deliberately tracks the established fact-verification
and NLI conventions — supported/conflicted/unsourced corresponds to
FEVER's SUPPORTED / REFUTED / NOT-ENOUGH-INFO and to NLI's
entail / contradict / neutral — with one strengthening: `unsourced` is
only issued under attested coverage, where NEI/neutral carry no such
guarantee.

**Def-use grounding** (code, global — "the model gives a point, code
propagates it"): each symbol use links to its reaching def (SSA-style
versions, code-assigned); grounding propagates from occurrences in $O$;
each def-use edge gets a risk:

- `grounded` — reaching def was tool-backed; safe.
- `premature` — used before grounding, but grounded later and consistent.
- `ungrounded` — never grounded anywhere; fabricated (name or value).
- `contradicted` — used a value a later grounded binding differs from.
- `stale` — used an older grounded version while a newer one exists
  *(defined, not yet emitted — needs coreference to an older version)*.

Value fidelity: for value edges with a grounded binding, the model
judges confirm/contradict — a local pairwise call in the Pass 2 mold.

**Constraint layer**: a third consumer folding question-derived
constraints against the same $E$, swept in the same partitioned way,
joined with the same Kleene discipline (unknown never escalates; an
omission verdict requires both a lexical code-negative and the attested
coverage sweep). Committing to an answer is the agent's implicit claim
that every requirement holds, so a requirement with no independent
evidence (`omitted`) is an unverified commitment, and one the evidence
refutes (`violated`) is a committed-against-evidence error — both are
source-verification signals the auditor localizes at the FIRST assertion
of the answer. No self-verification: the evidence set excludes the
commit step itself (a decidable step-id exclusion) — when Pass 1 labels
the final report's answer synthesis as an observation region, the agent
restating "I used method X" must not count as tool confirmation of X.
This layer detects unverified/refuted commitments; it does not judge
whether the agent gathered the RIGHT source (that needs a reference
outside the trajectory) — see out-of-scope.

## 4 · What leaves the index

Everything surfaced downstream is a **fact with provenance** — a node,
an edge with its witness quote and position, a status with its coverage
record — rendered as ADVISORY context for the auditor. The index never
designates an error step and never issues a global verdict: it is the
LSP, the auditor is the analyst.

## Concept → implementation

| concept | where |
|---|---|
| step content $x_i$ + extractor view (one shared walk) | `data.message_parts`, `data.view_body_with_map` |
| annotation grammar, strip-and-compare, alignment | `markup.py` |
| Pass 1 populate + verification | `index.TrajectoryIndex.populate_from_extraction` |
| $O_i$, segments | `index.Step.obs_regions`, `observation_segment` / `action_segment` |
| evidence edges, partitioned sweep, coverage | `edges.build_claim_edges` |
| identity edges (alias/coreference) | `adjudicate.py` |
| claim status fold | `verification.fold_claim_statuses` |
| def-use grounding, risks | `index.build_dependencies`, `data._build_references` |
| constraint layer | `constraints.analyze_constraints` |
| extractor prompt (annotation contract) | `agents/entity_extractor/prompts/default.md` |

```python
Step:         run_id, step_id, index, role, content, obs_regions
Claim:        id, run_id, step_id, text                                 # verbatim content slice
Symbol:       id, canonical_name, kind, aliases, entity_class
Reference:    symbol_id, location, kind, grounded, form, value          # an occurrence
Edge:         kind ∈ {supports, conflicts}, src=claim, dst=step,
              quote, evidence_position ∈ {before, same, after}
ClaimFinding: claim_id, status, edge_ids, evidence_empty                 # Pass 3 fold output
Dependency:   def_ref/use_ref, version, risk, def_value, use_value      # def-use edge
```

Every pass is best-effort and idempotent: a model failure degrades the
affected tuples (never fabricates), and derived layers are rebuilt
wholesale on rerun.

## Out of scope (current)

- Premises embedded in tool-call arguments (assumption smuggling) — a
  claim-adjacent node kind not yet extracted.
- Wrong-source errors (the agent verified against a source that does not
  actually establish the answer, or searched the wrong thing). Partially
  reachable: the constraint layer catches when a committed answer's
  requirements are unverified or refuted BY the trajectory's own evidence
  (a closed check). It cannot catch when the gathered source is itself
  wrong or irrelevant to the requirement — judging a source's validity
  needs a reference outside the trajectory (task constraints reach some
  of this; world knowledge, which the index never imports, the rest).
- Cross-run coreference; non-textual grounding.
