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

where $i$ is the position, $x_i$ a text (the step content, built
deterministically from the recorded message blocks by the single shared
walk `data.message_parts`), and the role
$r_i \in \{\mathrm{user}, \mathrm{assistant}, \mathrm{tool\_call},
\mathrm{tool\_result}, \mathrm{system}\}$. A role is **attested** when
the record's structure carries it (a real harness message); degraded
serializations (benchmark span dumps) arrive with uninformative roles.

**Authorship partition.** Every step content splits into agent-authored
and environment-authored regions:

$$x_i = A_i \uplus O_i, \qquad O_i = \mathrm{obs\_spans}_i \subseteq \mathrm{intervals}(x_i)$$

For attested $\mathrm{tool\_result}$ steps, $O_i = [0, |x_i|)$ by
structure; for everything else $O_i$ comes from Pass 1 and may be empty
or several disjoint intervals (query/content/summary sandwiches). The
derived **evidence universe** is

$$E = \{\, e_i = (i, O_i) \;:\; O_i \neq \varnothing \,\}$$

and everything outside it is agent action text. Downstream passes read
only the accessors `observation_segment` / `action_segment`; nothing
reads the role for evidence selection.

## 1 · Pass 1 — nodes (one trajectory visit)

The extractor model visits $T$ once, in chunks of 2–4 steps, and
re-emits each annotated step body **verbatim** with
`⟦tag attrs|content⟧` spans (`markup.py`). Code strips all spans and
compares against the view it sent (whitespace-tolerant alignment):
equality makes every span offset exact; inequality rejects the step's
annotations whole, into the prune log. Three node kinds:

| node | tag | definition | code verification |
|---|---|---|---|
| provenance | `⟦obs\|…⟧` | the intervals of $O_i$ | strip-compare → exact offsets; overlaps merged; attested roles never overridden |
| claim | `⟦claim\|…⟧` | $c = (i, [a,b))$ — a sentence the agent asserts **as settled fact**: verification statements, conclusions/identifications, settled negative findings; plans, questions, hedges excluded (the test is stance, not wording or polarity) | text is the content slice $x_i[a{:}b]$ — verbatim by construction |
| symbol | `⟦sym …\|…⟧` | $\sigma = (\mathrm{name}, \mathrm{kind}, \mathrm{class})$, declared at first mention; other marked surfaces of the same name become aliases automatically | occurrences located by exact name/alias match (code); an occurrence inside $O$ is a grounded (tool-backed) def |

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
annotations falling in the ellipsis are rejected, recorded.

## 2 · Pass 2 — edges (relations between nodes)

The model proposes **local pairwise** relations; code verifies the
decidable part of every proposal and records every rejection.

**Evidence edges** (`edges.py`):

$$\mathrm{evd} \subseteq C \times E \times \{\mathrm{supports}, \mathrm{conflicts}\}$$

judged over the full bipartite $C \times E$: code partitions $E$
deterministically into whole-step groups $P_1, \dots, P_J$ under a char
budget with

$$\bigcup_{j=1}^{J} P_j = E$$

(this coverage is what later entitles a negative), shows each partition
to the oracle with ALL claims (sampled twice, union — sampling can only
surface candidates, never assert one), and each proposal carries a
witness quote $q$. Code keeps an edge $(c, e_j, k, q)$ only if

  * both endpoints exist;
  * the FULL quote is verbatim in $O_j$ (whitespace-normalized
    containment, not a prefix match);
  * $q$ is not the claim's own words (a claim is not its own evidence).

Each kept edge records the timeline **fact**

$$\mathrm{position}(c, e_j) = \mathrm{sign}(j - i) \in \{\mathrm{before}, \mathrm{same}, \mathrm{after}\}, \qquad c = (i, [a,b))$$

never a filter: consistency is time-agnostic, and an after-conflicts
edge ("committed early, refuted later, never retracted") is signal, not
noise.

**Identity edges** (`adjudicate.py`): alias/coreference resolution — an
equivalence over $\Sigma$'s surface forms; code blocks candidate pairs,
the model judges each locally, code merges and rewrites anaphors.

## 3 · Pass 3 — judgments (folds over nodes + edges)

**Claim status** (`verification.py`, pure code, zero model calls):

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
$\mathrm{universe\_empty} \iff E = \varnothing$ distinguishes "swept
$E$ and found nothing" from "the record carries no observation content
at all" — itself a strong trajectory-level fact.

**Def-use grounding** (code, global — "the model gives a point, code
propagates it"): each symbol use links to its reaching def (SSA-style
versions, code-assigned); grounding propagates from occurrences in $O$;
each edge gets a risk:

- `grounded` — reaching def was tool-backed; safe.
- `premature` — used before grounding, but grounded later and consistent.
- `ungrounded` — never grounded anywhere; fabricated (name or value).
- `contradicted` — used a value a later grounded binding differs from.
- `stale` — used an older grounded version while a newer one exists
  *(defined, not yet emitted — needs coreference to an older version)*.

Value fidelity (`compare_values`): for value edges with a grounded
binding, the model judges confirm/contradict — a local pairwise call in
the Pass 2 mold.

**Constraint layer** (`constraints.py`, Pass 0/E/J/L): a third consumer
folding question-derived constraints against the same $E$, swept in the
same partitioned way, joined with the same Kleene discipline (unknown
never escalates; Omitted requires the lexical code-negative AND the
attested coverage sweep).

## 4 · What leaves the index

Everything surfaced downstream is a **fact with provenance** — a node,
an edge with its witness quote and position, a status with its coverage
record — rendered as ADVISORY context for the auditor. The index never
designates an error step and never issues a global verdict: it is the
LSP, the auditor is the analyst.

## Data model (implementation shapes)

```python
Step:         run_id, step_id, index, role, content, obs_spans          # + segments as properties
Claim:        id, run_id, step_id, text                                 # verbatim content slice
Symbol:       id, canonical_name, kind, aliases, entity_class
Reference:    symbol_id, location, kind, grounded, form, value          # an occurrence
Edge:         kind ∈ {supports, conflicts}, src=claim, dst=step,
              quote, evidence_position ∈ {before, same, after}
ClaimFinding: claim_id, status, edge_ids, universe_empty                 # Pass 3 fold output
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
