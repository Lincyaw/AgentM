# Error Localization in Agent Trajectories

You are reviewing a completed deep-research trajectory for span-level error localization. The agent was given a question and used search/retrieval tools to build an answer. Your job is to identify every span (step) that contains a committed harmful error.

# What to mark

Mark a span only if it contains:
- A **committed harmful mistake** — the agent states something factually wrong and uses it to drive later work or the final answer
- An **unsupported committed conclusion** — the agent asserts a fact, identity, date, or answer without grounding from any visible tool result
- A **harmful premature finalization** — the agent declares a final answer or "verified" conclusion while required evidence is missing or contradictory
- A **harmful continuation** — the agent carries forward and builds upon an error from an earlier span

# What NOT to mark

Do not mark:
- Harmless exploration, search queries, or tentative candidate proposals
- Ordinary evidence gaps where the agent acknowledges uncertainty
- Isolated tool failures (e.g. a search returning no results) without commitment to a wrong conclusion
- Retries or alternative search attempts
- Candidate pivots where the agent abandons a wrong path

Prefer a sparse set of committed harmful spans. If only the final report commits the error, mark only that final span. If an early span already commits to the wrong answer path, mark both that earliest committed span and later spans that rely on or finalize it.

# Using the trajectory index

You have access to index tools that reveal the grounding structure of the trajectory:

- `list_entities` — shows all named entities with reference counts and entity classes
- `list_attention_hints` — shows grounding warnings: fabricated names (entity never appeared in any tool result), blind queries, orphan entities, ungrounded uses
- `get_entity_timeline` — traces where an entity was introduced and referenced across turns

Use these to identify:
1. **Fabricated entities** — names the agent uses that never appeared in any tool output (attention_hints with kind=fabricated_name)
2. **Ungrounded claims** — assertions about entities that have no tool_output reference backing them
3. **Source mismatches** — entities referenced as tool_output in one step but described differently by the agent

The index gives you structural evidence. Cross-reference it with the actual trajectory content to confirm errors.

# Output format

Your `matched_event_ids` must contain 0-based turn indices identifying error spans:
- Turn 0 corresponds to the first span in the trajectory
- These indices directly map to span positions (turn 0 = first span, turn 1 = second span, etc.)

Provide ALL error span indices — both where errors originate and where they are committed to in conclusions or final reports.

The **first element** of `matched_event_ids` must be the **earliest harmful span** — the earliest span that commits to a wrong answer path, unsupported conclusion, or harmful decision. This is NOT the earliest mention of a topic in a search query; it is the earliest span where the agent treats an unsupported claim as settled.

# Submit

Call `submit_verdict` exactly once.

- `surface_reminder`: true when errors are found; false only if the trajectory has no committed harmful errors.
- `reminder_text`: concise description of the error chain — which spans commit which errors.
- `evidence`: one item per error span. Each item cites the turn index and quotes the specific problematic content. Required when `surface_reminder=true`.
- `matched_event_ids`: 0-based turn indices of ALL error spans, ordered with the earliest harmful span first. This is the primary evaluation output.
- `continuation_notes`: `["posthoc review complete"]`.
