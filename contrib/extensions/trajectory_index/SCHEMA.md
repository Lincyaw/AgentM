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

Cost note: re-emission makes output ≈ chunk size, which is what makes
small chunks the operating point. The extractor's prompt window (prefix
truncation of very long steps) is Pass 1's declared unsoundness —
annotations falling past the cut are rejected, recorded.

## 2 · Pass 2 — edges (relations between nodes)

The model proposes **local pairwise** relations; code verifies the
decidable part of every proposal and records every rejection.

**Evidence edges**:

$$\mathrm{evd} \subseteq C \times E \times \{\mathrm{supports}, \mathrm{conflicts}\}$$

judged over the full bipartite $C \times E$: code partitions $E$
deterministically into whole-step groups $P_1, \dots, P_J$ under a
character budget with

$$\bigcup_{j=1}^{J} P_j = E$$

(this coverage is what later entitles a negative), shows each partition
to the oracle with ALL claims (sampled twice, union — sampling can only
surface candidates, never assert one), and each proposal carries a
witness quote $q$. A proposed edge $(c, e_j, k, q)$ is kept only if

  * both endpoints exist;
  * the FULL quote is verbatim inside $O_j$ (whitespace-normalized
    containment, not a prefix match);
  * $q$ is not the claim's own words (a claim is not its own evidence).

Each kept edge records the timeline **fact**

$$\mathrm{position}(c, e_j) = \mathrm{sign}(j - i) \in \{\mathrm{before}, \mathrm{same}, \mathrm{after}\}, \qquad c = (i, [a,b))$$

never a filter: consistency is time-agnostic, and an after-conflicts
edge ("committed early, refuted later, never retracted") is signal, not
noise.

**Identity edges**: alias/coreference resolution — an equivalence over
$\Sigma$'s surface forms; code blocks candidate pairs, the model judges
each locally, code merges and rewrites anaphors.

## 3 · Pass 3 — judgments (folds over nodes + edges)

**Claim status** (pure code, zero model calls):

$$\mathrm{status}(c) =
\begin{cases}
\mathrm{conflicted} & \exists\, e : (c, e, \mathrm{conflicts}) \in \mathrm{evd} \quad \text{(dominates)}\\
\mathrm{supported} & \text{else } \exists\, e : (c, e, \mathrm{supports}) \in \mathrm{evd}\\
\mathrm{unsourced} & \text{else if coverage complete} \quad \text{(attested negative)}\\
\mathrm{unknown} & \text{otherwise} \quad \text{(never escalates)}
\end{cases}$$

Coverage is complete when every partition of $E$ had at least one
successful oracle call (content coverage is attested; recall within a
shown partition remains the oracle's). The flag
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
coverage sweep).

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
- Erroneous *actions* (searching the wrong thing) — an error in what was
  done, not in what was asserted; commit/constraint territory.
- Cross-run coreference; non-textual grounding.
