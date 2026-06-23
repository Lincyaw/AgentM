# Trajectory Semantic Index

LSP-style symbol-reference-relation index over agent trajectories. Extracts
a semantic graph from conversation history — symbols (services, files,
metrics, errors, …), their references (where each is mentioned and how),
and relations between them (causes, depends_on, derived_from, …).

Designed for incremental indexing: each extraction call processes only new
messages and a compact symbol registry, not the full history.

## Quick start

**All commands must be run from `contrib/extensions/trajectory_index/`** —
the extraction child session needs `trajectory_index` on the Python path.

```bash
cd contrib/extensions/trajectory_index

# One-shot extraction on a trace file (all messages in one chunk)
uv run python -m trajectory_index.data collect \
  path/to/session.jsonl \
  --model doubao \
  --output /tmp/traj_out \
  --index-output /tmp/traj_out/index.json \
  --chunk-size 200

# Incremental extraction (recommended: 20 messages per chunk)
uv run python -m trajectory_index.data collect \
  path/to/session.jsonl \
  --model litellm-dsv4flash-nothink \
  --output /tmp/traj_out \
  --index-output /tmp/traj_out/index.json \
  --chunk-size 20

# With real-time streaming of model thinking/output
uv run python -m trajectory_index.data collect \
  path/to/session.jsonl \
  --model litellm-dsv4flash-nothink \
  --output /tmp/traj_out \
  --index-output /tmp/traj_out/index.json \
  --chunk-size 20 \
  --debug
```

## Architecture

```
vocabulary.yaml          ← single source of truth for symbol/reference/relation types
        │
        ├── index.py     ← in-memory index (Symbol, Reference, Relation, TrajectoryIndex)
        ├── atom.py      ← runtime agent tools (index_trajectory, search_symbols, get_symbol_context)
        ├── data.py      ← SFT data pipeline CLI (collect, export-messages)
        └── agents/
            └── entity_extractor/
                ├── schema.py      ← Pydantic extraction result schema
                ├── context.py     ← system prompt injection (auto-generates vocabulary section)
                └── prompts/
                    └── default.md ← extraction rules (vocabulary appended at runtime)
```

### Data model (LSP-inspired)

| Type | LSP analogue | Description |
|------|-------------|-------------|
| `Symbol` | SymbolInformation | A named concept with kind, aliases, summary |
| `Reference` | Location + ReferenceContext | One occurrence of a symbol at a specific step |
| `Relation` | — | Directed edge between two symbols (causes, uses, …) |
| `Step` | TextDocument | One message in the trajectory |
| `Location` | Range | Character span within a step |

### Vocabulary (`vocabulary.yaml`)

All enum values and their descriptions are defined in `vocabulary.yaml`.
The extraction prompt is auto-generated from this file — add a new kind
or relation type here and it propagates everywhere.

Validated at three levels:
1. **Import time** — YAML keys must match StrEnum members exactly (bidirectional)
2. **Extraction output** — model output kind/type values checked against vocabulary
3. **Index integrity** — reference/relation edges must point to existing symbols

## Data collection for SFT distillation

### Step 1: Prepare trace sources

Traces can come from:
- **Local JSONL files**: `.agentm/observability/*.jsonl` under any case directory
- **ClickHouse session IDs**: `--session <id>` (repeatable)

```bash
# Find available trace files
find datasets/ -name '*.jsonl' -path '*observability*' | head

# Or use ClickHouse session IDs
agentm trace index | head
```

### Step 2: Run teacher extraction

Use a strong model (doubao, DS Flash) as the teacher to generate
extraction labels:

```bash
uv run python -m trajectory_index.data collect \
  /path/to/traces/*.jsonl \
  --model litellm-dsv4flash-nothink \
  --output data/ \
  --index-output data/index.json \
  --split train \
  --chunk-size 20 \
  --concurrency 2
```

This produces:
- `data/train.jsonl` — HuggingFace-compatible SFT dataset (one example per chunk)
- `data/index.json` — merged index from all chunks (updated live after each chunk)
- `data/dataset_info.json` — HF dataset metadata

Each SFT example has the format:
```json
{
  "messages": [
    {"role": "system", "content": "Extract symbols, references, and relations..."},
    {"role": "user", "content": "{\"known_symbols\": [...], \"messages\": [...]}"},
    {"role": "assistant", "content": "{\"symbols\": [...], \"references\": [...], \"relations\": [...]}"}
  ]
}
```

### Step 3: Inspect the index

```bash
# Pretty-print the index
python3 -c "
import json
with open('data/index.json') as f:
    idx = json.load(f)
print(json.dumps(idx['stats'], indent=2))
for s in idx['symbols']:
    print(f\"{s['name']:40s}  kind={s['kind']:10s}  summary={s.get('summary','')[:50]}\")
"

# Or load programmatically
from trajectory_index.index import TrajectoryIndex
index = TrajectoryIndex.load("data/index.json")
results = index.search("recommendation")
```

### CLI reference

```
uv run python -m trajectory_index.data collect [OPTIONS] [PATHS]...
uv run python -m trajectory_index.data export-messages TRACE_FILE
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `doubao` | config.toml model profile for extraction |
| `--output` | `data` | HuggingFace dataset output directory |
| `--index-output` | — | Write live index JSON to this path |
| `--split` | `train` | Dataset split name |
| `--chunk-size` | `20` | Messages per extraction chunk |
| `--concurrency` | `2` | Max concurrent trace sources |
| `--debug` | off | Stream model thinking/output to stderr |
| `--session` | — | ClickHouse session ID (repeatable) |

## Evaluation

```bash
# Run eval on the 4 built-in cases
uv run python contrib/extensions/trajectory_index/eval/run.py \
  --model litellm-dsv4flash-nothink \
  --cases contrib/extensions/trajectory_index/eval/cases/

# Use a different model
uv run python contrib/extensions/trajectory_index/eval/run.py \
  --model qwen
```

Reports entity recall/precision, kind accuracy, relation recall, and latency
per case. Cases are YAML files in `eval/cases/` — add new ones for your domain.

## Incremental extraction design

The key insight: **don't make the small model understand the full
trajectory — make it a linker.**

Each extraction call receives:
1. **`known_symbols`** — compact registry of symbols already extracted
   (name + kind + summary + aliases)
2. **`messages`** — only the new turns since last indexing (with sequential
   IDs: 0, 1, 2, …)

The model's job:
- Extract new symbols from the current chunk
- Produce references (occurrences) for both new and known symbols
- Declare relations between symbols

This keeps the prompt size bounded by `O(registry) + O(chunk)` instead of
`O(full_trajectory)`.

### Chunk boundaries

Chunks respect tool_call/tool_result pairs — a tool call and its result
are never split across chunks.

### Validation and retry

If the model's output fails JSON parsing, Pydantic validation, or
vocabulary checks, the error is sent back to a fresh session for
correction (one retry attempt).

## Model recommendations

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| DeepSeek V4 Flash (no-think) | Fast | Best | 65K output, no truncation. Recommended teacher |
| doubao-seed-2-0-pro | Moderate | Good | May truncate on large chunks |
| Qwen 3.5 2B (local) | Fastest | Fair (zero-shot) | Needs SFT to follow turn_id format |

For distillation: use DS Flash as teacher, Qwen 2B as student.
