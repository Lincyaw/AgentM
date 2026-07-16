# Trajectory index — formal model

The index is a static-analysis stack over one recorded agent trajectory.
This file defines the objects each pass produces and consumes, and is the
authority for the model. The governing contracts referenced below as P1–P7
are the P-numbered invariants those passes uphold — among them P2 (no silent
false negative: every prune is logged), P4 (positive-polarity establishment
over whole-step windows), P5 (unknown never escalates), P6 (code owns only the
decidable), P7 (localization anchors are per-benchmark empirical).

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
The same degradation can strip two substrates several later extensions
presume: the VALUE WORLD (per-reference values — absent when the extractor
emits none) and the ACTION SUBSTRATE (tool calls folded into assistant prose
— no `tool_call` role, no call id). Tool OUTPUTS still survive as `⟦obs⟧`
regions, but tool CALLS and values may not; an extension that needs them is
meaningful in general yet unexercisable on such a record (see Designed
extensions).

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
| source | who produced this text? | agent, env (generalizes to the authority lattice: subagents, users, harness) | `obs` = env; `given` = user/harness authority (built, §Designed extensions); unlabeled = agent |
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
| value | `⟦val sym=…\|…⟧` | a concrete value the agent reads from a tool result or a key metric produced by a command; `sym` names the symbol this value belongs to | located in the source step by word-level matching; creates a `Reference` with `value` populated and `entity_class=value` |

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

Evidence edges (claim ↔ observation) are one label family; identity
edges (`same_as`, an equivalence over $\Sigma$'s surface forms, §2.7)
are another; agent-internal edges (claim ↔ claim: `contradicts` /
`retracts` / `supersedes`, §2.8) a third — designed, the conspicuously
missing member; `about` (relevance) is a reserved label — named, not
yet specified. The extension rule mirrors Pass 1's: anything that
relates two regions is a new label here, never a new inline annotation.

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

The root cause is that authorship is labeled by POSITION (which slot the
bytes occupy) rather than by ORIGIN (which authority produced the value).
The sound fix tracks each value's producing authority and taint-matches
an env value against the agent's own prior output: a value whose origin
chain passes through the agent is not tool-backed, however it is later
re-served. This is the single-agent case of the authority lattice named
in §1 — a sub-agent relaying a value is the general case, env-laundering
the degenerate one, and both are one check over authored origin.
Recording the producing authority at extraction is cheap and structural;
the taint-match that would consume it is not yet built, so grounding
stays position-labeled and this remains a known leak.

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

The order is informational, not evaluative. Read as alarms: a
MECHANICAL failure of the fold (a failed oracle call, a rejected
certificate, a retrieval miss) can UNDER-alarm (a missed conflicts edge
leaves a wrong claim looking supported) but never OVER-alarm. This
fail-stop guarantee is precise but PRECONDITIONED, and the preconditions
are load-bearing — three paths outside mechanical failure DO fabricate
an alarm:

- **model polarity** — the certificate proves the quote exists, not that
  it conflicts (§2.3); a wrong `conflicts` judgment fabricates a
  `conflicted` status. The guarantee covers mechanical failure, not a
  false model polarity call.
- **identity contamination** — a false `same_as` (§2.7) or coreference
  bind fabricates a `contradicted` value verdict downstream.
- **grounding recall** — the def-use fold is NOT symmetric with claim
  status: a Pass 1 obs miss makes a real grounded value look
  `ungrounded`/fabricated (§3 gates this on coverage to restore symmetry).

The honest statement is therefore: the evidence-edge fold never
over-alarms under CORRECT model polarity, CORRECT identity, and COMPLETE
Pass 1 obs-recall; where those fail, over-alarm is possible — which is
why the model-judged tiers surface as advisory, not fact (§4). The
safety story does NOT extend to identity edges, whose failure direction
is inverted (§2.7).

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
contaminating). The current merge is a destructive union-find pop that
leaves no trace in $R$; the sound form persists `same_as` as a
retractable edge (Designed extensions), making a merge auditable and a
false one reversible rather than permanently contaminating.

### 2.8 Agent-internal edges (designed)

Evidence edges compare the agent to the ENVIRONMENT. The symmetric half
— the agent against ITSELF — is a claim↔claim relation the current model
does not build (not even reserved), so self-contradiction and silent
belief-revision are invisible unless a claim also happens to conflict
with an observation. It decomposes into the same three tiers as the
evidence edge:

- **value contradiction** (`contradicts`): two claims bind the same
  symbol to unequal values. This is NOT the near-free code check it first
  looks like. The def-use comparison it would reuse presumes
  `Reference.value` is populated, and the extractor emits no values today —
  the value world is structurally absent (every reference's `value` is ⊥),
  so this tier has no operands until Pass 1 is extended to emit them. Even
  with values, "same symbol" is model-decided (identity contamination,
  §2.6) and value inequality over-alarms on comparability ($5M vs 5000000,
  a point release vs a major, two values answering different sub-questions
  of one symbol). So the honest form is an ADVISORY self-contradiction lead
  — the genuinely new capability is catching an agent that contradicts
  ITSELF with no environment evidence — not a code-decidable high-signal
  edge.
- **retraction / supersession** (`retracts`): a claim carrying a
  `retract` stance (Pass 1) that shares a symbol with an earlier claim —
  DECIDABLE-modulo-coreference (the stance is model-marked, the target is
  pinned by the shared symbol). A fold reading it can mark the earlier
  claim dead, which also removes the cross-branch over-alarm of the
  lineage extension.
- **propositional contradiction** — *BUILT*: two claims that conflict with
  no shared value ("X is safe" vs "X is risky"; an answer identified then
  disowned) — irreducible model NLI, surfaced as a MONOTONE advisory
  witnessed by both claim texts, never a hard verdict. Same discipline as
  the evidence edge's polarity call. Code groups candidate claim pairs by
  shared symbol; the model judges each pair; a `self_contradicts` edge is
  emitted only on `contradict`
  (`pass2_edges.self_contradiction.build_self_contradiction_edges`, surfaced
  in `get_insights`).

Status: the **propositional tier is BUILT** (above) — the agent-vs-itself
advisory that needs no value world. The **value-contradiction tier** is NOT
near-free (correction from review): the value world it would reuse is empty
(`Reference.value` = ⊥ across the corpus), so it needs a Pass 1 extractor
change to emit values first, then stays advisory (the comparability and
identity caveats above). The **retraction tier** needs a Pass 1 `retract`
stance (extractor change) and is decidable-modulo-coreference once it lands.

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

**Unsupported commitment** (the dominant localized error in
retrieval/QA regimes — P7 is per-benchmark: a coding agent's dominant
error is a broken edit, a computer-use agent's a wrong action; here
"dominant" means the deep-research shape — a commit-filtered read of the
fold above, not a new oracle). What an
auditor most often localizes is not a wrong value but a *commitment the
evidence does not establish*: $\mathrm{commit}(c) \wedge \neg\,
\mathrm{adequately\_supported}(c)$. Neither half is new machinery.

- *The commit half is the precision filter.* A commitment is a claim with
  the `commit` attribute (the final answer) or a verification stance
  (`verified`/`confirmed` — model-marked, not a keyword table: the model
  flags an assertion that claims settled verification). Adequacy is judged
  ONLY at commitments; most `unsourced` claims are baseline narration (low
  signal), and filtering by commit is what turns the raw statuses into a
  sharp signal.
- *The support half is the fold above.* For a commitment,
  $\mathrm{status}(c)$ already decides most cases: `conflicted` and
  `unsourced`-under-complete-coverage are unsupported; `unknown` abstains;
  `supported` passes.
- *The residual is a strength gap:* `supported`, but the evidence
  establishes LESS than the commitment asserts (short-term safety cited
  for a general claim; a related page read as confirmation). Neither
  `unsourced` (evidence exists) nor `conflicted` (no contradiction) fires,
  so the binary fold is blind here. This is an irreducible model judgment
  — does the evidence establish the claim AT THE ASSERTED STRENGTH? —
  surfaced as a MONOTONE advisory (witness: the supports quote plus the
  commitment text; the auditor confirms by reading both), never a hard
  verdict.

A verification stance sharpens the flag: `verified` asserts maximal
support, so any non-`supported` status under it is a maximal gap. The
soundness caveat of §2.4 carries: the `unsourced` half is
negation-as-failure, sound only relative to retrieval recall, so the whole
judgment is an advisory lead, not a decidable fact — it localizes at the
commitment step (the `commit`/verification assertion), the claim-level
analog of the constraint layer's first-assertion localization below.
Status: the claim-status fold and the `commit` attribute exist; the
verification stance, the commit-filtered surfacing, and the strength-gap
advisory are design, not yet built.

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

Grounding carries a coverage asymmetry §2.6 flags: `ungrounded`/fabricated is
a NEGATIVE ("no grounding def anywhere"), sound only if Pass 1's obs-recall
over the value's candidate sources is complete; a missing grounding def may be
a Pass 1 miss, not a fabrication. Correction (from review): the fix is NOT
"reuse the claim sweep's coverage" — that certificate attests coverage over
the ALREADY-INDEXED obs space, and a Pass 1 obs-recall miss (a real tool
source the extractor never labeled `⟦obs⟧`) is by construction outside that
space, so `EdgeCoverage.complete` stays true on exactly the failure it would
need to catch. It cannot transfer. Two forms are actually sound: (a)
unconditional advisory demotion of the grounding negative — the direction the
authority tier (§Designed extensions) and the get_insights reframing already
lean toward — with only the decidable `E = ∅` case (`n_observation_steps = 0`,
computable from the index alone) kept as a hard trajectory-level qualifier; or
(b) a genuine per-symbol Pass 1 ATTENTION attestation — the extractor certifies
it visited every obs region where the name occurs — which is a real new
feature, not a certificate reuse. Until then the shipped grounding warnings
can over-alarm on obs under-recall.

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

The package is laid out by pass: `ir/` (IR + container + persistence +
diagnostics), `pass1_nodes/`, `pass2_edges/`, `pass3_folds/`, plus `oracle.py`
(shared one-shot model call) and the `atom.py` / `query_atom.py` presenters.
`ir.index.TrajectoryIndex` is a thin facade whose pass methods delegate to the
per-pass modules and re-export the IR, so `from trajectory_index.ir.index
import Symbol` still resolves.

| concept | where |
|---|---|
| step content $x_i$ + extractor view (one shared walk) | `pass1_nodes.serialize.message_parts`, `pass1_nodes.serialize.view_body_with_map` |
| annotation grammar, strip-and-compare, alignment | `pass1_nodes/markup.py` |
| Pass 1 populate + verification | `pass1_nodes.populate.populate_from_extraction` (method `ir.index.TrajectoryIndex.populate_from_extraction`) |
| $O_i$, segments | `ir.models.Step.obs_regions`, `observation_segment` / `action_segment` |
| value extraction (`⟦val⟧` tag) | extractor prompt `⟦val sym=…\|value⟧` |
| evidence edges, partitioned sweep, coverage | `pass2_edges.claims.build_claim_edges` |
| identity edges (alias/coreference) | `pass2_edges.identity` |
| claim status fold | `pass3_folds.claim_status.fold_claim_statuses` |
| def-use grounding, risks | `pass3_folds.grounding.build_dependencies`, `pass1_nodes.serialize._build_references` |
| value fidelity (confirm/contradict) | `pass3_folds.grounding.compare_values` |
| constraint layer | `pass3_folds.constraints.analyze_constraints` |
| value flow (timelines, constraint checks) | `pass3_folds.value_flow` |
| index insights surfaced to the auditor (§4) | `query_tools.build_insights_tool` (via `atom.py` / `query_atom.py`) |
| extractor prompt (annotation contract) | `agents/entity_extractor/prompts/default.md` |

```python
Step:         run_id, step_id, index, role, content, call_id, obs_regions
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

## Designed extensions (mechanisms, not yet built)

Each has a concrete mechanism in the model's own discipline
(propose-then-certify, monotone folds, decidable-or-advisory); these are
sequenced, not deferred.

**General value vs substrate (from adversarial review).** Judged on design
merit for ANY agent — not one benchmark — the ranking is: the **Authority
chain** is the real prize (it unlocks the multi-agent class the current
index is blind to); **agent-internal edges** give an ADVISORY
self-contradiction capability — its propositional tier is BUILT
(`self_contradicts`, no value world needed, §2.8); the **`claims_empty`
fact** is BUILT (small but general). A degraded serialization (a
benchmark record dump) can strip the substrate these presume — the value
world and the action substrate (tool calls folded into assistant prose: no
`tool_call` role, no call id) — so several extensions are meaningful in
general yet UNEXERCISABLE on such a record, independent of their design
merit. That gates *where* they can be validated (attested harness sessions,
not a degraded dump), not *whether* they are worth building.

- **Agent-internal edges** (claim↔claim, §2.8) — the missing symmetric
  half of the evidence edge. *Propositional contradiction is BUILT*
  (`self_contradicts` edges, model-judged monotone advisory) — the real
  capability lever, it lets an auditor localize self-contradiction and
  belief revision WITHOUT environment evidence (verified on 0808: "identified
  the film" ↔ "cannot yet confirm the film"). Still designed: value
  contradiction and retraction (needs a Pass 1 `retract` stance).
  Value contradiction is now EXERCISABLE — `Reference.value` is populated
  (see below) — but the comparison fold is not yet built.
- **Action as a first-class node** — *RETIRED.* The information Action
  carried is now expressed through existing IR: Step (`tool_name`, `call_id`),
  Symbol + Reference (targets, operation via reference kinds), step content
  (purpose in tool_call JSON args), and Value timeline (diffs via `⟦val⟧`
  extraction). Purpose is read directly from tool_call args in step content.
  The regex-based command classification (`classify_bash`, `ActionOp`) added
  noise without signal the other layers do not already provide.
- **Value world** — *BUILT (first tier).* `Reference.value` is no longer ⊥.
  LLM extraction via the `⟦val sym=…|value⟧` tag lets the extractor mark
  concrete values read from tool results (metrics, config parameters,
  measurements). These feed into the `Reference.value` field and the
  existing def-use machinery.
- **Value flow fold** (Pass 3) — *BUILT.* Deterministic + one
  model-judged summary over valued references
  (`pass3_folds.value_flow`):
  (1) **value timelines** — per value-symbol, deduplicated sequence of
  value transitions across steps;
  (2) **constraint checks** — LLM-judged satisfaction of each constraint
  against the final observed value table (met / violated / irrelevant).
- **Authority (source generalized)** — the source labeling widens from
  binary `{agent, env}` to an authority (principal: `agent:A`, `tool:X`,
  `user`, `harness`), collapsing to the binary when the record does not
  structurally mark finer. *Built (first tier):* a mention in a `user`
  step (an entity named in the task) or a `system` step (harness-provided)
  is now `given` — a grounding source, not agent invention — so
  question-derived entities no longer surface as `fabricated_name`
  (`pass3_folds.grounding._provenance_kind`; `given ∈ _PRODUCING_KINDS`).
  *Still designed:* grounding that reads the full authority CHAIN (does it
  terminate in a sealed external source?) rather than position — the
  taint-match of §2.3. env-laundering is the single-agent case;
  orchestrator↔worker delegation (a `require` directed at another
  authority, discharged by that authority's `act`) is the general one.
- **Lineage-aware ordering** — the linear total order of §0 relaxes to a
  step graph via lightweight `branch_id` / `supersedes` / `batch_id`
  metadata on the step. The three order-dependent computations
  (`evidence_position` §2.5, the reaching-def scan, conflict folding)
  become lineage-aware: conflicts are drawn only within a comparable
  lineage (no cross-branch fabrication — the planning-tree / ToT case),
  and concurrently-dispatched calls sharing a `batch_id` record
  within-batch order as `unknown` (the async case). Everything defaults
  to one branch, one batch — today's linear behavior — and degrades
  gracefully where the record does not mark lineage.
- **`same_as` as a retractable edge** (§2.7) — persist the identity
  decision as an edge in $R$ rather than a destructive union-find pop, so
  a merge is auditable, a false one reversible, and its contamination
  bounded.
- **Grounding coverage-gating** (§3) — the naive form ("reuse the claim
  sweep's coverage") does NOT work: that certificate cannot see a Pass 1
  obs-recall miss (§3 correction). The two sound forms are (a) unconditional
  advisory demotion of the grounding negative, keeping only `E = ∅`
  (`n_observation_steps = 0`, index-computable) as a hard qualifier — cheap,
  and the direction authority + get_insights already lean; (b) a genuine
  per-symbol Pass 1 ATTENTION attestation (the extractor certifies it visited
  every obs region where the name occurs) — a real feature, not a reuse.
- **`claims_empty` trajectory fact** (§3) — a code-decidable flag mirroring
  `evidence_empty`: Pass 1 extracted zero claims (a terse / search-heavy
  trajectory, or an extraction failure). Surfaced so a consumer reads
  claim-analysis ABSENCE as "signal unavailable, not clean" — the P2
  no-silent-gap discipline for the claim side. Small and general; keys on
  `index.claims` (empty even when populated fine if the fold did not run).

## Out of scope (genuinely bounded)

- **Wrong-source errors** (the agent verified against a source that does
  not actually establish the answer, or searched the wrong thing).
  Partially reachable: the constraint layer catches when a committed
  answer's requirements are unverified or refuted BY the trajectory's own
  evidence (a closed check). It cannot catch when the gathered source is
  itself wrong or irrelevant — judging a source's validity needs a
  reference outside the trajectory (task constraints reach some of this;
  world knowledge, which the index never imports, the rest).
- **Non-textual substrate** (pixel-only computer-use, RL / robotics) —
  the atoms, certificates, and strip-and-compare all presume a
  verbatim-checkable representation, which a raster or a continuous
  action vector lacks. Agents with a structured surrogate (accessibility
  tree, DOM) reduce to the textual case; pure pixels do not.
- **Cross-run coreference** — $\Sigma$ is global across runs (§2.1) but
  no cross-trajectory reasoning is built on it.
- **Premises embedded in tool-call arguments** (assumption smuggling) — a
  claim-adjacent node kind; reachable once the action node lands, since
  its arguments are exactly where a smuggled premise sits.
