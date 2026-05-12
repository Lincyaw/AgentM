# llmharness

LLM-as-harness for AgentM: a two-phase cognitive audit pipeline
(extractor + auditor) that supervises a main agent, plus an offline
distill workflow that turns recorded sessions into SFT data for
training a small (~4B) model into the same harness role.

## Two product surfaces

1. **Live supervision.** Mount the adapter onto a session; every
   `TurnEnd` triggers an extractor child (graph builder), every k
   turns triggers an auditor child (verdict producer). The auditor's
   verdict, if it says so, is injected as a one-line advisory before
   the next agent turn.

   ```bash
   agentm --extension llmharness.adapters.agentm
   ```

2. **Distill data collection.** Run the live pipeline with reminders
   off + sample-id binding to record clean trajectories, then run
   `llmharness-distill {label, export}` offline to produce
   `extractor.jsonl` + `auditor.jsonl` SFT files.

   ```bash
   LLMHARNESS_DISTILL_SAMPLE_ID=<id> \
   LLMHARNESS_DISTILL_DATASET=<dataset.jsonl> \
     agentm --extension llmharness.adapters.agentm \
            --extension llmharness.distill.binding \
            --scenario rca ...

   llmharness-distill label  --replay-dir <cwd>/.agentm/audit_replay \
                             --dataset    <dataset.jsonl> \
                             --out        ./distill_labels
   llmharness-distill export --labels ./distill_labels \
                             --replay-dir <cwd>/.agentm/audit_replay \
                             --out  ./sft
   ```

## Docs

| File | When to read it |
|---|---|
| [docs/01-architecture.md](docs/01-architecture.md) | Components, static dependency graph, runtime data flow. Read this first. |
| [docs/02-schemas.md](docs/02-schemas.md) | Authoritative shapes: `Event` / `Edge` / `Finding` / `Verdict`, `ReplayRecord`, meta sidecar, SFT JSONL. Use as a reference. |
| [docs/03-distill-recipe.md](docs/03-distill-recipe.md) | End-to-end recipe: from a labeled fault-injection dataset to `sft/{extractor,auditor}.jsonl`. Includes the six design decisions that make the labels safe to learn from. |
| [docs/04-extending.md](docs/04-extending.md) | Registering a new audit check; adapting the distill flow to a non-rca dataset. |
| [docs/05-profiles-and-prompts.md](docs/05-profiles-and-prompts.md) | Pluggable tool profiles + prompt variants for the extractor / auditor children. Read this when running A/B experiments. |
| [docs/06-case-aggregation.md](docs/06-case-aggregation.md) | Per-case directory layout produced by `llmharness-aggregate`. Read this for human review of a run or before exporting trajectories. |

## CLI entry points

| Script | Purpose |
|---|---|
| `llmharness` | one-shot session demo (see `cli.py`) |
| `llmharness-replay {extractor,auditor}` | replay a recorded phase with a different provider/prompt for A/B |
| `llmharness-distill {label,export}` | distill pipeline driver |
| `llmharness-aggregate` | replay sidecar → per-case directories for review + training-data export |

## Schema stability

`src/llmharness/schema.py` is the public contract for downstream
consumers (e.g. rca-autorl). Breaking changes bump the package
version in `pyproject.toml`. The v3 schema break is documented in
that module's docstring.
