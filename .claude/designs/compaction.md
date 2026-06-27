# Compaction

How a long session is kept under the model's context window.

## Model: full compress

When the durable session branch crosses the token threshold, **every turn
since the previous compaction is summarized into a single `user` message that
replaces the whole prior context**. There is no verbatim "recent tail" — the
summary is the only carried-over representation of the compressed turns.

This is a deliberate point on the tradeoff curve:

- **Simpler than keep-a-tail.** Because nothing is kept verbatim, there is no
  cut-point search, no tool_call/tool_result pairing hazard, and no split-turn
  handling. The summary is one clean message.
- **Loses recent verbatim fidelity** (including the exact wording of the
  triggering turn). Recovered on demand via `read_history` — see below.

Compaction runs in the `before_send_to_llm` handler, between turns, where the
branch's trailing message is always a user message or tool_result (a pending
completion request). Compressing the entire prefix is therefore always a
valid, self-contained operation.

### Why the summary is a `user` message

After a full compress the rebuilt context can be exactly `[summary]`. A
provider completion request must not end on an assistant turn, so the
`compaction` entry materializes as a **user** message (`compaction_prompts`'s
`_CompactionEntryMaterializer`). New turns after the compaction follow it
normally: `[user(summary), assistant(...), user(...), ...]`.

## Incremental chaining (not append)

Compaction happens many times in a long session. Each pass folds **only the
new turns** (those after the previous compaction's `covered_through_turn`)
into the previous summary, using the `update` prompt that *rewrites/merges*
rather than concatenates. This keeps the running checkpoint:

- **bounded** — it does not grow without limit (pure append would),
- **internally consistent** — In-Progress → Done migrations, stale blockers
  removed, no contradictory stacked checkpoints,
- **cheap** — O(total turns), not O(n²): already-summarized turns are never
  re-read.

The previous summary's `read_files` / `modified_files` are carried forward too.

## Turn indexing and `read_history`

The summary tags each chunk with a `[Turn N]` marker. Turns are numbered by
`agentm.core.lib.enumerate_turns`: a turn begins at each `UserMessage` and
includes the assistant/tool replies until the next user message;
compaction / branch-summary entries are skipped. Because the session tree only
ever appends — compaction adds an entry, it never deletes the original message
entries — **turn indices are stable for the life of the session**, and
`get_branch()` always retains the full original trajectory.

The `read_history(start, end?)` tool reads `get_branch()`, re-derives the same
turn numbering, and returns the verbatim messages for the requested turn(s). So
when the agent needs detail behind a `[Turn 14]` reference in the summary, it
recovers it on demand rather than carrying it in-context.

`read_history` is **in-session** (reads the live SessionManager — the source
of truth, no flush lag). It is distinct from `query_traces` / `agentm trace`,
which read the observability JSONL for *cross-session, post-hoc* analysis.

## Trigger

`should_compact`: `context_tokens > context_window − reserve_tokens`. Token
count prefers the last assistant message's real provider `Usage` and uses the
shared tiktoken helper for messages after that usage measurement. When
`reserve_tokens >= context_window` the threshold is non-positive (a
misconfiguration, or a tiny test model) and auto-compaction is disabled rather
than firing every turn — lower `reserve_tokens` to compact on small-window
models.

## Configuration

`llm_compaction` config: `enabled`, `reserve_tokens`, `custom_instructions`
(appended to the summary prompt as a focus hint), `tool_result_max_tokens`.
Prompt bodies live in the `compaction_prompts` floor atom and are overridable
through its config. `read_history` config: `tool_result_max_tokens`,
`total_max_tokens` (output caps).

## Boundaries

- Engine logic lives in the `llm_compaction` atom; only the value types
  (`CompactionSettings` / `CompactionResult` / `CompactionPrompts` / …) are in
  `core.abi.compaction`. Turn enumeration is the one piece shared with
  `read_history`, so it lives in `core.lib.turns` (atoms import it as stdlib;
  §11 forbids atom-to-atom imports).
- The kernel owns no English prompt text (issue #76): bodies are resolved from
  the prompt registry; missing prompts degrade to empty strings with a
  diagnostic.

## Deferred

- **Segmented summarization.** Today a compaction pass summarizes its new
  turns in one LLM call. If quality degrades on a very large backlog, split
  the backlog into N-turn segments, summarize each, then merge — the chaining
  interface already accommodates this.
- **Separate summary model.** Summarization reuses the session's model. A
  cheaper dedicated summary model could be a future knob.
