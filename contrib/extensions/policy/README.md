# policy-engine

Evidence-driven failure detection and intervention for coding agents.

A coding agent's trajectory is a sequence of tool calls that read, write,
and execute against a codebase. This package detects structural failure
patterns as queries over the trajectory, recommends relevant checklist
items, and intervenes when those items are confirmed.

```
① data plane      trajectory (PG) + repository symbol index → queryable facts
② tagger          per-turn LLM annotation → phase + semantic predicates
③ retrieval       predicate matching → candidate checklist items
④ critic          LLM judgment → confirmed violations
⑤ intervention    inject / critic / compact
⑥ evolution       mine → compile → evaluate → select → diversify
```

## Data plane

The data plane provides **neutral facts** over the trajectory. No pattern
judgments — failure detection belongs in retrieval/signals, not here.

### Two inputs

**1. Agent tool events (from SDK trajectory, PG).**

Each tool event produces an **action → region** edge:
- `read(path, offset, limit)` → read edge to (path, [offset, offset+limit])
- `edit(path, old, new)` → edit edge to (path, file-level approximation)
- `write(path, content)` → write edge to (path, whole file)
- `bash(cmd)` → command text + exit code (no classification)

Exposed as PG views in the `policy` schema:
- `v_tool_calls` — all tool calls flattened
- `v_regions` — action → region (path, start_line, end_line, relation)
- `v_reads` / `v_edits` / `v_mutations` — convenience filters
- `v_bash_runs` — bash commands with exit_code
- `v_source_content` — source text the agent observed

**2. Repository symbol index (optional, degradable).**

AST-parsed symbol definitions and references, built lazily by
`RepositoryIndex` using `ast-grep` in the sandbox. When the agent
first reads or edits a file, that file + its 1-hop import neighbors
are parsed. The index expands as the agent's activity expands; edits
trigger re-parsing of the affected file.

Only the agent's activity subset is synced to PG — not the whole repo.
Stored in `policy.file_symbols`, exposed via `policy.v_file_deps`
(file → file dependencies via shared symbol names).

When no index is available, `file_symbols` is empty, `v_file_deps`
returns nothing, and signals that depend on symbol data silently
degrade.

### Failure detection = gap between the two

The action graph says what the agent touched. The symbol index says
what depends on what. A signal query joins the two to find gaps. The
data plane never encodes these gaps itself — signals do.

## Tagger

The tagger is a **per-turn lightweight LLM call** that reads one step
of the agent's work — its tool calls (with arguments and results),
its reasoning text (if any), and the task description (on the first
step) — and outputs structured annotations.

### Output

**Phase** (one per turn, mutually exclusive):
exploring / diagnosing / implementing / validating / concluding.

**Tags** (per turn, from the predicate vocabulary):
Boolean facts observable from the step's content. Examples:
`completion_claim`, `validation_failure`, `agent_has_edited`,
`task_requires_reference_parity`. The full tag list is in
`vocabulary.yaml` — the tagger prompt is generated from it.

There is no separate task classifier. On the first turn, the tagger
sees the task description and can set task-level tags (e.g.,
`performance_or_quantitative_context`, `symptom_repetition_wording`).
On later turns, it tags agent behavior.

### Prompt generation

The tagger prompt is **automatically generated** from `vocabulary.yaml`.
When compile produces new predicates or prunes old ones, the prompt
updates. No manual prompt maintenance — the vocabulary is the single
source of truth for what tags exist.

## Retrieval

Two-stage cascade selecting which checklist items are relevant:

### Stage 1: Predicate matching (cheap, high recall)

Each checklist item declares a `trigger` expression — a boolean
combination (AND / OR / NOT) of predicates from the vocabulary:

```yaml
- id: belief_consistency_1
  when:
    trigger: validation_failure AND completion_claim
    checkpoint: stop
```

A predicate is true when the tagger has emitted that tag for any turn
in the session. Matching is deterministic boolean logic over the
tagger's cached output — no LLM call.

### Stage 2: Critic (expensive, high precision)

Candidate items from Stage 1 go to the critic (`agents/critic.yaml`):
an LLM that reads the session evidence + the item's review question
and judges whether the item is actually violated. Only confirmed
violations produce interventions.

### Suppression gates

A matching item is not enough. Three gates decide whether the inject
actually lands, all calibrated on a 30-session GPT run (104 injects,
measuring whether the agent used any tool in the turn after the inject):

| gate | default | why |
|---|---|---|
| `max_injections` | 5 | Total per session. |
| `min_work_turns` | 1 | Tool-using turns required since the last inject. With zero, injects drew a tool response 11% of the time; with one or more, 75–100%. Zero-work means the agent is answering the previous note in prose — re-injecting there only deepens the loop. |
| `max_stop_injections` | 2 | Stop-checkpoint injects decay hard: 1st 64%, 2nd 71%, 3rd 38%, 4th+ 0%. |

Suppression does not consume the item — it stays unfired and can land
once the agent has done real work. Continuous-checkpoint injects are the
productive ones (≈92% draw a tool response vs ≈63% at stop), so the
budget should not be spent at stop.

## Intervention

| delivery | behavior |
|---|---|
| `inject` | Append a message quoting the agent's own evidence. |
| `critic` | Spawn a reviewer subagent; inject confirmed verdict. |
| `compact` | Compress context: keep facts, discard subjective reasoning. Via SDK `ContextCompactionService`. |

### Inject wording

The message must ask for a correction, not a verdict. An earlier
phrasing ("audit your process against each point below") was answered
literally: agents replied with a point-by-point `Pass / Partial` writeup
and never reopened the work — in one session the agent graded itself
`Partial` on evidence adequacy and then stopped. Every inject now ends
with an explicit instruction to act rather than reply, and stop-checkpoint
injects are prefixed with a note that the work is still reopenable.

## Evolution

The checklist, predicate vocabulary, and tagger co-evolve through an
adaptive loop.

### The loop

```
① Mine       miner queries failing trajectories along D1–D7,
             produces candidate items with when_notes

② Compile    compiler agent decomposes each when_note into a trigger
             expression over the predicate vocabulary; proposes new
             predicates when needed

③ Deploy     tagger prompt regenerated from vocabulary;
             checklist updated with trigger expressions

④ Evaluate   replay over the corpus; measure per-item fitness
             (when it fires, does the critic confirm?)

⑤ Select     prune low-fitness items and unreferenced predicates

⑥ Diversify  ensure D1–D7 coverage; deduplicate overlapping items
```

### CLI

```bash
python -m policy_engine compile              # ② when_notes → triggers + vocabulary
python -m policy_engine tag <dsn>            # tag all sessions with tagger LLM
python -m policy_engine replay <dsn> <sid>   # ④ replay one session through the gates
python -m policy_engine evaluate <dsn>       # ④ evaluate across all sessions
python -m policy_engine select               # ⑤ prune by fitness
python -m policy_engine evolve <dsn>         # ②→④ in one command
```

### Predicate vocabulary lifecycle

| event | action |
|---|---|
| New item needs a new concept | Compile proposes predicate, added to vocabulary |
| Predicate unreferenced by any item | Pruned from vocabulary |
| Vocabulary too large | Merge near-synonymous predicates |

### Miner framework (D1–D7)

The miner works along seven dimensions derived from the representation
chain:

```
task → understanding → diagnosis → plan → change → evidence → belief → claim/stop
```

| Dimension | Comparand pair |
|---|---|
| D1 Interpretation fidelity | task ↔ understanding |
| D2 Requirement coverage | understanding → plan/change |
| D3 Implementation fidelity | claimed behavior ↔ actual code semantics |
| D4 Diagnosis validity | claimed cause ↔ collected evidence |
| D5 Evidence adequacy | change ↔ executed validation |
| D6 Belief-evidence consistency | evidence ↔ belief/claim |
| D7 Loop dynamics | the chain over time |

The full dimension model is in `docs/policy-feedback-dimensions.md`.

## Related work

- **AdaMAST** (arXiv 2607.16387): adaptive failure taxonomies, fixed
  axes + induced codes. Key finding: passive context injection
  outperforms forced mechanical auditing.
- **Act·onomy** (arXiv 2605.13625): behavioral taxonomy for agent
  runtime. Orthogonal — classifies what the agent is doing, not
  where the mismatch is.
- **ActPlane** (arXiv 2606.25189v1): OS-level policy enforcement via
  eBPF + IFC DSL. `after/since` gate model deferred in favor of
  predicate matching for simplicity.
