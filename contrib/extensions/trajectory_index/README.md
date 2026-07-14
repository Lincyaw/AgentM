# Trajectory Semantic Index

An LSP-style symbol/reference index over an agent trajectory, plus a
grounding (def-use) layer that flags fabricated identifiers. A small
extraction model visits the trajectory once per chunk and re-emits each
message with inline annotation markup (`⟦tag attrs|content⟧`); code
verifies every annotation by strip-and-compare, then builds references
and the def-use graph deterministically.

The model is a *linker*, not a summarizer: each extraction call sees only
the new messages plus a compact registry of already-known symbols, so the
prompt stays `O(registry) + O(chunk)` regardless of trajectory length.

## Runtime atom

`atom.py` registers three tools for the main agent to index and query its
own trajectory:

- `index_trajectory` — incrementally index new messages: Pass 1
  (markup extraction) → Pass 2 (same-entity alias merge + coreference) →
  Pass 3 (def-use / grounding) → Pass 3.5 (value fidelity). Best-effort:
  any model-judgment pass that fails degrades to the deterministic layers.
- `search_symbols` — name/summary search over indexed symbols.
- `get_symbol_context` — definition, timeline, related symbols for one id.

Extraction runs in a child session on a small model
(`TrajectoryIndexConfig.model`, default `qwen`).

## Module map

```
markup.py     ⟦tag|content⟧ annotation language: parse + whitespace-tolerant
              strip-and-compare alignment (the Pass 1 verification primitive)
index.py      TrajectoryIndex — data model (Symbol/Reference/Dependency/…),
              write path, alias blocking, def-use build, warnings, persistence
data.py       message cleaning, JSON extraction, vocabulary validation,
              deterministic reference generation
atom.py       runtime tools + extraction child-session orchestration
adjudicate.py Pass 2/3.5 model judgments (alias, coreference, value fidelity)
diagnostics.py the shared transcript (P3) + prune-log (P2) sink
agents/entity_extractor/  the extraction agent (schema + prompt + context)
vocabulary*.yaml          symbol-kind vocabularies (default/coding/research/…)
```

The claim/edge/constraint analysis layer (`edges.py`, `verification.py`,
`constraints.py`, and the `claims`/`constraints` nodes populated from the
markup) is the L1–L4 direction described in `designs/`. It is **not wired
into the runtime atom** and currently has no in-repo consumer — see those
design docs and SCHEMA.md before building on it.

### Data model (LSP-inspired)

| Type | LSP analogue | Description |
|------|-------------|-------------|
| `Symbol` | SymbolInformation | A named concept with kind, aliases, summary |
| `Reference` | Location + ReferenceContext | One occurrence of a symbol at a step |
| `Dependency` | — | A def-use edge with a grounding risk verdict |
| `Relation` | — | Directed edge between two symbols |
| `Step` | TextDocument | One message in the trajectory |
| `Location` | Range | Character span within a step |

### Pass 1 output: annotation markup

The extractor re-emits each annotated message body verbatim with spans
inserted:

- `⟦sym kind=… class=…|surface⟧` — first mention of a symbol; a canonical
  `name` attr is added when the surface is not itself canonical, and other
  surfaces become aliases automatically.
- `⟦obs|…⟧` — a retrieved/environment segment (several per message allowed).
- `⟦claim role=…|…⟧` — a settled-fact assertion (`role=commit` marks the
  final answer).
- input side: `⟦known|name⟧` marks registry symbols already extracted.

Verification is strip-and-compare: removing every `⟦…⟧` must reproduce the
extractor's view of the message (whitespace-tolerant), which makes every
span offset exact. A message that fails is rejected whole and logged to the
prune log. Delimiters are U+27E6/27E7 (they don't collide with `[[…]]`
wiki links in fetched content).

### Vocabulary

Symbol kinds live in `vocabulary.yaml` (plus `vocabulary.<name>.yaml`
variants selected via `TrajectoryIndexConfig.vocabulary`). The extraction
prompt's kind list is generated from the selected file, and model output is
validated against it at parse time.

## Grounding / def-use layer

`build_dependencies()` chains each structured symbol's references within a
run into def-use edges and reads a grounding risk off the reaching binding:

- `grounded` — reaching def was tool-backed;
- `premature` — used before grounded, but grounded consistently later;
- `ungrounded` — never grounded anywhere (a fabricated name/value);
- `contradicted` / `stale` — value-world verdicts (Pass 3.5).

`warnings()` folds this into per-symbol flags (`fabricated_name`,
`blind_query`, `premature_use`, …) with pure code, no model. See SCHEMA.md
for the full build rule.

## Programmatic use

```python
from trajectory_index.index import TrajectoryIndex

index = TrajectoryIndex.load("index.json")
for r in index.search("recommendation"):
    print(r.symbol.canonical_name, r.symbol.kind)
print(index.warning_summary())
```

Offline teacher extraction and evaluation over recorded sessions live in
the `agentm_eval.benchmarks.index_eval` benchmark (adapter name `index`),
which loads sessions from ClickHouse, runs the extraction pipeline, and
saves per-chunk results plus grounding warnings for review.
