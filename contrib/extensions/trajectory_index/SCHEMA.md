# Trajectory grounding analysis — design

Over the extracted symbol graph, catch when the agent **relied on something no tool
ever gave it** — a fabricated *name*, or a fabricated *value* on a real name.

A **taint** analysis: a value is *grounded* if it came from a tool (trusted),
*ungrounded* if the model conjured it. Relying on an ungrounded value is the bug.
A name is the value-free special case (it just exists or not); a value can also be
*wrong* — contradict what the tool returned.

## Three stages (compiler-style)

Rule: the LLM only makes **local** judgments; **all global traversal is code**
("the model gives a point, code propagates it").

**1 · PARSE** — extract + tag.
- LLM: extract entities and set `entity_class`. Decisive test — *could a tool report
  different content for this while it stays the same thing?* yes → `value`,
  no → `identifier`, vague/anaphoric → `unknown`.
- Code: from message structure, tag each occurrence def/use and `grounded`
  (tool_output = grounded def; tool_input / mention = ungrounded use); assign SSA
  versions.

**2 · RESOLVE** — make every reference point at the right entity.
- alias (`resolve_aliases`): "are these two surface forms the same entity?"
- coreference (`resolve_references`): "which earlier entity does this anaphor
  (`this` / `it`) denote?"
- Code blocks candidates, the LLM decides each locally, code merges / rewrites.

**3 · DATAFLOW** — def-use + grounding (code, global).
- Each use links to its reaching def (most-recent at a strictly earlier step);
  grounding propagates forward; each edge gets a risk.
- value fidelity (`compare_values`): for value edges with a grounded binding, the
  LLM judges confirm / contradict.

## Risk (per def-use edge)

- `grounded` — reaching def was tool-backed; safe.
- `premature` — used before grounding, but grounded later and consistent.
- `ungrounded` — never grounded anywhere; fabricated (name or value).
- `contradicted` — used a value a later grounded binding differs from.
- `stale` — used an older grounded version while a newer one exists *(defined, not
  yet emitted — needs coreference to an older version)*.

`entity_class` is independent of `kind`: `kind` is the type (from the vocabulary),
`entity_class` is the name/value axis. There is no regex gate — the LLM owns the
name/value/vague call.

## Data model

```python
type EntityClass = Literal["identifier", "value", "unknown"]
type RefForm     = Literal["direct", "anaphor"]
type Risk        = Literal["grounded", "premature", "ungrounded", "contradicted", "stale"]

Symbol:     ... entity_class
Reference:  ... grounded: bool, form, value, resolved_from
Dependency: def_step/def_ref/def_version, use_step/use_ref, risk,
            grounded_by_step_id, def_value, use_value
```
SSA `version` = the ordinal of a def among an entity's defs (code-assigned, no
separate table).

## Out of scope

- Reasoning errors where every name and value is grounded but the conclusion does
  not follow (an unsupported inference) — leaves no grounding trace.
- Cross-run coreference; non-textual grounding.

Each stage is best-effort and idempotent: a model failure leaves the deterministic
layer intact; the derived layer is rebuilt wholesale.
