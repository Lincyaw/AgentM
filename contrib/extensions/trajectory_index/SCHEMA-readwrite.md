# Trajectory grounding analysis — complete schema

Over the extracted symbol graph, run a **def-use / grounding analysis** that catches
when the agent *relied on something no tool ever gave it*. The unifying frame is
**SSA (static single assignment)**: every entity is a cell with versioned bindings;
a use resolves to a reaching binding; risk is read off the binding's grounding.

Two failure worlds, one model:

- **Fabricated name** — the agent used an identifier (file / table / id / endpoint)
  no tool produced. The "value" of an identifier is its own existence.
- **Fabricated value** — the agent acted on a *value* (`cpu=90%`, `tier=premium`)
  that a tool never returned or later contradicted. The name is real; the value is wrong.

Names are the **value-free special case** of values. One schema covers both.

---

## The PL picture

| PL concept | here |
|---|---|
| **const** | an *identifier* entity — bound once, name rigidly denotes (file path, id). Name world. |
| **variable** | a *value* entity — a slot whose bound value changes over time (`user.tier`). Value world. |
| **SSA version** | each (re)definition of an entity = a new binding version (`tier₁`, `tier₂`). |
| **def / use** | a binding (production) / a read of it. |
| **reaching definition** | the binding a use resolves to (most-recent prior version). |
| **pointer / deref** | an *anaphor* (`this`, `it`, `the previous result`) — a use that points at an entity indirectly; resolving it = dereferencing. |
| **taint** | grounded (from a tool, trusted) vs ungrounded (model-conjured). |

---

## Entities, bindings, references

An **entity** is a cell. Its `entity_class` (LLM-decided, Pass 1):

- `identifier` — const/name world. The binding's "value" is existence.
- `value` — variable/SSA world. Each binding carries an actual value.
- `unknown` — vague/anaphoric surface that could not be tied to a concrete entity;
  excluded from def-use until resolved (Pass 2 coreference may promote it).

A **binding** is a def of an entity at a step, with an SSA `version` (ordinal among
that entity's defs in the run), a `value` (for `value` entities), and a `grounded`
bit. Bindings are not a separate table — they are the def-role references, ordered.

A **reference** is one occurrence. Its `form`:

- `direct` — the name appears verbatim (`abnormal_traces`), string-matchable.
- `anaphor` — a pronoun/description (`this`, `it`, `that table`, `the previous
  result`). Carries no entity until Pass 2 resolves it to a target entity + version.

Grounding of a reference is derived from message structure (`tool_output` →
grounded def; `tool_input`/`mention` → ungrounded use) and propagates: a use that
copies a grounded binding is grounded.

---

## The four local judgments the LLM makes (divide-and-conquer)

Code does all global traversal (versioning, reaching-def, taint, risk). The LLM
only ever answers a **local** question:

1. **Name vs value recognition** (Pass 1, per entity) — is this an `identifier` or a
   `value`, and if `value`, what value does this occurrence assert/observe? Replaces
   the old `looks_structured` regex — the model knows a rigid name from a slot, and a
   proper noun (`Big Stone Gap`) from anaphora (`the previous one`).
2. **Coreference resolution** (Pass 2, per anaphor) — code proposes recent in-scope
   candidate entities/versions; the model picks which one `this`/`it` binds to. The
   anaphor is then rewritten to a direct reference of that entity+version.
3. **Alias resolution** (Pass 2, per candidate pair) — "are these two surface forms
   the same entity?" (existing `resolve_aliases`). Coreference is its sibling: both
   are "resolve a reference to an entity."
4. **Value comparison** (Pass 3.5, per flagged value edge) — "does the tool's
   grounded value confirm or contradict the value the agent acted on?" Sets
   `contradicted`.

Everything else is deterministic code.

---

## Passes

```
1 PARSE      LLM: extract entities, classify identifier/value, extract asserted values,
             flag anaphors.  code: def/use + grounded from message structure; SSA versions.
2 RESOLVE    code blocks candidates → LLM decides → code rewrites:
               (a) alias:      same-entity surface forms      (resolve_aliases)
               (b) coreference: anaphor → antecedent+version   (resolve_references)
             After Pass 2 every reference points at a concrete entity+version.
3 DATAFLOW   code, global: each use → reaching binding (SSA), grounding propagates,
             risk graded per edge.
3.5 COMPARE  LLM, local, on flagged value edges only: confirm vs contradict → risk.
```

Iron rule preserved: the LLM only does local point-judgments; every global
traversal is code.

---

## Unified risk taxonomy (one axis, name ⊂ value)

Per def-use edge (`reaching binding → use`):

| risk | condition | world |
|---|---|---|
| `grounded` | reaching binding grounded and current | both |
| `premature` | reaching binding ungrounded, but the entity is grounded at a later step and consistent | both |
| `ungrounded` | entity never grounded anywhere — fabricated (name: fake id; value: made-up number) | both |
| `contradicted` | used an asserted value; a later grounded binding of the entity has a **different** value (Pass 3.5) | value |
| `stale` | used an older grounded version while a **newer grounded version** exists | value |

For `identifier` entities only `grounded`/`premature`/`ungrounded` can arise (no
value to contradict or go stale) — the name world falls out as the degenerate case.

---

## Data model

```python
type EntityClass = Literal["identifier", "value", "unknown"]
type RefForm     = Literal["direct", "anaphor"]
type Risk        = Literal["grounded", "premature", "ungrounded", "contradicted", "stale"]

@dataclass  # Symbol/entity
class Symbol:
    ...canonical_name, kind, aliases, summary...
    entity_class: EntityClass = "identifier"   # LLM (Pass 1)

@dataclass(frozen=True)
class Reference:               # one occurrence (a def or a use)
    ...id, symbol_id, run_id, step_id, location, text, role, kind...
    grounded: bool = False              # tool-backed? (taint bit)
    grounds_ref_id: str | None = None   # if grounded by copying a prior def
    form: RefForm = "direct"            # direct name vs anaphor (LLM flags anaphor)
    value: str | None = None            # value asserted/observed here (value entities)
    resolved_from: str | None = None    # original anaphor text, if Pass 2 rewrote it
    structured: bool = True             # derived: entity_class != "unknown"

@dataclass(frozen=True)
class Dependency:              # one def-use edge, version-aware
    ...id, symbol_id, run_id...
    def_step_id: str; def_ref_id: str; def_version: int
    use_step_id: str; use_ref_id: str
    risk: Risk
    grounded_by_step_id: str | None = None   # later grounding step, if any
    def_value: str | None = None             # for value edges: the two sides compared
    use_value: str | None = None
```

SSA `version` = ordinal of the def among the entity's defs in the run (code-assigned).
No separate binding table.

---

## Scenarios this schema must cover (and how)

| scenario | mechanism |
|---|---|
| fabricated file/id name | identifier entity, `ungrounded` |
| used a name before the tool confirmed it | identifier, `premature` |
| `this table` / `it` → prior entity | anaphor → coreference (Pass 2) → direct ref |
| pointer chain (`this` → `it` → …) | transitive resolution; union-find in code |
| ambiguous `this` (order vs user) | LLM picks by type/salience (Pass 2) |
| `this result` → a step's output | anaphor whose antecedent is a def-binding, not a named entity |
| wrong value on a real name (`cpu=90%` vs tool 45%) | value entity, `contradicted` (Pass 3.5) |
| value changed then used old one (`tier` premium→basic) | SSA versions, `stale` |
| proper noun (`Big Stone Gap`) repeated verbatim | identifier by LLM; string-matched — no longer dropped by a regex gate |
| vague concept (`the customer`, `the approach`) | `unknown`; excluded unless coreference ties it down |

---

## Determinism & staging notes

- Pass 1 & 3 are deterministic given the LLM's per-occurrence labels; Pass 2/3.5 are
  the model seams. `looks_structured` (regex gate) is **removed** — the model owns
  the name/value/vague judgment.
- SSA `version` and reaching-def use step index, ties by `location.start`.
- `contradicted` (value world) is emitted by Pass 3.5 (`compare_values`). `stale` is
  defined but **not yet emitted**: it needs a use bound to an *older* version while a
  newer grounded one exists — reaching-def always takes the most-recent, so it only
  arises once coreference resolves an anaphor to a specific older version. The field
  and risk value are in place for that refinement.
- Build order per run: Pass 1 → alias-merge (2a) → coreference (2b) → Pass 3 →
  Pass 3.5. Each is idempotent; the derived layer is rebuilt wholesale.
- Every model pass is best-effort: a model failure leaves the deterministic layer
  intact (name-axis risks still computed).

## Out of scope (still)

- Cross-run / cross-session coreference.
- Non-textual grounding (an image/plot a tool returned).
