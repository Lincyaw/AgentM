# Trajectory grounding analysis — design

An enhancement to `trajectory_index`: over the extracted symbol/reference graph,
run a **def-use / grounding analysis** that catches when the model *used something
no tool ever gave it*.

The existing `Symbol` / `Reference` / `Relation` (co-occurrence graph, for
retrieval) are unchanged and kept. `Dependency` is a new, orthogonal def-use layer
over steps. Do not conflate them.

## What it catches

Frame it as a taint analysis: a value is **grounded** if it came from a tool (a
trusted source), **ungrounded** if the model conjured it. Relying on an ungrounded
value is the bug.

- **Fabricated name** *(built)* — the model referenced a structured identifier
  (file / table / column / endpoint / id) that no tool ever produced. Built now.
- **Fabricated value** *(deferred, see end)* — the model acted on a *value* that
  differs from what a tool returned (tool said CPU 45%, model diagnosed off 90%).
  The name is real; the value is wrong. Needs value-tracking + coreference; later.

## The pipeline — three passes, like a compiler

The organizing principle: everything deterministic/local goes to cheap code; only
the one irreducible *local* semantic judgment goes to a model; **global traversal
is always code** ("the model gives a point, code propagates it").

| Pass | Compiler analogue | What we do | Who |
|---|---|---|---|
| 1 · PARSE | lexing / parsing | extract entities + occurrences; tag each occurrence's `kind` from message structure | LLM (local extract) + code (tag) |
| 2 · RESOLVE | name resolution / symbol table | decide which surface forms are the same entity, then merge | code (block) → LLM (judge one pair) → code (cluster + merge) |
| 3 · DATAFLOW | reaching-defs / def-use / taint | link each use to its reaching def, propagate grounding, flag ungrounded uses | code, global |

The LLM only ever does **local** work (extract a chunk in Pass 1, judge one pair in
Pass 2). Global traversal (Pass 3) is entirely code. That invariant is enforced by
the pass structure, not by discipline.

---

## Pass 1 — PARSE: extract entities + tag occurrences

- **LLM (local):** read each chunk of steps, emit structured entities with their
  surface forms / aliases and where they occur. This is the one place per step that
  needs a model — mapping natural language to entities.
- **Code (deterministic):** for each occurrence assign a `kind` from **message
  structure**, not a model:

  | message block | `kind` | role |
  |---|---|---|
  | tool result | `tool_output` | **def** (produces a value) · grounded |
  | tool call | `tool_input` | **use** · ungrounded (unless it copies a grounded def — Pass 3) |
  | assistant text | `mention` | **use** · ungrounded |

  Also mark `structured` via the `looks_structured` gate (below).

### The structured-entity gate

A small model is reliable at **local, verbatim** extraction, not at **cross-step
coreference**. So only **structured, verbatim-matchable identifiers** drive the
def-use layer, because matching them is string equality, not semantic coreference.

- `structured = True`: file paths, DB / row keys, order/user/txn ids, URLs,
  `tool_call_id`, function names, error codes, git SHAs, line numbers.
- `structured = False`: natural-language concepts ("the customer", "the previous
  result"). Extracted and stored for retrieval, but they get **no def-use edges**
  yet — linking them needs coreference (the deferred value work).

This still catches the largest hallucination class (model fabricates an id/path,
then acts on it) and is the grounded layer that carries signal anyway.

---

## Pass 2 — RESOLVE: name resolution (same-entity)

The analysis is only as correct as its entity resolution: a use links to its def
only if the two surface forms are unified. The extractor routinely **splits one
entity into several** — a data file and the table registered from it
(`normal_traces.parquet` vs `normal_traces`), a service under two names, an endpoint
and its short form. Left unmerged, a def under one form fails to ground a use under
another, producing false "fabricated name" flags.

This is entity resolution, not a rules problem — lexical rules over-merge
(`normal_traces` ⊂ `abnormal_traces`, but they are opposites; `.../travel` vs
`.../travels` are different endpoints at 0.98 similarity). On the real RCA run the
lexical score is *anti-correlated* with mergeability: the top-scoring pairs are
traps, the true merges (`X.parquet` ↔ `X`) sit at the bottom. So it is the same
divide-and-conquer as everything else:

- **Block (code):** `alias_candidates()` proposes structured-symbol pairs on a cheap
  lexical signal (proper substring, or normalized-name similarity ≥ threshold), each
  carrying names, kinds, and sample reference snippets. It only *proposes*.
- **Judge (model):** `resolve_aliases()` asks an ~8B model, per pair, "are these the
  same entity?" — bounded and local; the model never traverses the graph.
- **Merge (code):** `apply_alias_merges(groups)` union-finds the yes-pairs into
  groups, folds each into one canonical symbol (most-referenced), re-points
  references and relations, demotes folded names to aliases, and invalidates the
  derived def-use layer. Idempotent. Runs **before** Pass 3.

`build_dependencies()` (Pass 3) does **no** merging — a pure Pass 3 run links only
exactly-normalized entities. Merge is an explicit upstream step.

---

## Pass 3 — DATAFLOW: def-use + grounding (deterministic, global)

For each structured entity, within each run, in step order:

1. **def vs use.** A reference is a **def** if `kind` is a producing kind
   (`tool_output`, …) or it is the entity's first appearance in the run; otherwise
   it is a **use**.
2. **grounding.** A def is **grounded** if its value is tool-backed
   (`grounded_from_kind`: a producing kind). A `tool_input` that copies a prior
   grounded def is upgraded to grounded (records `grounds_ref_id`) — this is how
   grounding **propagates forward**.
3. **reaching def.** Each use links to the most-recent def at a **strictly earlier
   step** (`def_step < use_step`; ties by `location.start`). A def in the use's own
   step does not count — a same-step def/use is not cross-step reliance, so it emits
   no edge.
4. **risk** on each edge:

   | condition | risk | meaning |
   |---|---|---|
   | reaching def is grounded | `grounded` | safe |
   | reaching def ungrounded, but entity IS grounded at a later step | `premature` | ran ahead of evidence, held up (records `grounded_by_step_id`) |
   | entity never grounded anywhere in the run | `ungrounded` | **fabricated-name candidate** |

Whether an ungrounded use was *bold* (model put the name into a tool call) or *idle*
(named it only in reasoning) is recoverable from the use's `kind` (`tool_input` vs
`mention`) — it is not stored as a separate grade.

### The query this enables

```
Dependency where risk == "ungrounded"
  → a use relied on an identifier no tool ever produced
  → fabricated-name candidate (with its use site)
```

`premature` is a weaker, secondary signal (impatience, not hallucination). This is
the signal pure topology (edge shape / run length) cannot produce — it needs the
grounding bit on the producing def, the one label a generic graph model has no
channel for.

---

## Data model

```python
type Risk = Literal["grounded", "premature", "ungrounded"]

@dataclass(frozen=True, slots=True)
class Reference:          # ...existing fields unchanged...
    grounded: bool = False           # is this occurrence's value tool-backed?
    grounds_ref_id: str | None = None  # if grounded by copying a prior def, which
    structured: bool = False

@dataclass(frozen=True, slots=True)
class Dependency:         # one def-use edge
    id: str
    symbol_id: str
    run_id: str
    def_step_id: str
    def_ref_id: str
    use_step_id: str
    use_ref_id: str
    risk: Risk
    grounded_by_step_id: str | None = None  # if ungrounded at use, a later grounding step
    confidence: float = 1.0
```

### Determinism notes

- reaching-def ordering: by step index, ties by `location.start`.
- `premature` requires the grounding step **after** the use step.
- `Dependency.confidence` = min(def, use) reference confidence.
- read-modify-write (a tool that both reads and produces one resource): `kind` tags
  one role; treat as a def, note as an edge case.

---

## Deferred — fabricated value (extends Pass 3, not a new pass)

The dangerous RCA error is often not a fake name but a **wrong value on a real
name**: the tool returned CPU 45%, the model reasoned "90%, that's the root cause".
Every name is grounded, so the grounded/ungrounded partition says "fine" — it misses
it entirely.

This is not a fourth pass. It is Pass 3 with a **richer lattice**: carry the
*value* along the def-use chain (not just existence), and add one **local** semantic
compare at the end (LLM: "does the tool's value confirm or contradict the value the
model acted on?"). It additionally needs (a) value binding in Pass 1 (extract the
asserted value, not just the entity) and (b) coreference for natural-language
subjects — which is why it rides on top of the structured layer, later.

## Out of scope (noted, not built)

- **Natural-language-concept reliance** — needs cross-step coreference; the def-use
  layer is structured-entity-only for now.
- **Staleness / revalidate-before-act** — a grounded value that later goes stale (a
  quoted price that changes). A distinct failure mode (grounded-but-stale) from
  ungrounded reliance; needs resource timestamps/versions. Not modeled here.
