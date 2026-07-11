# Error Localization in Agent Trajectories

You are reviewing a completed agent trajectory to identify which steps contain errors. The agent was given a question and used search/retrieval tools to find the answer. Your job is to locate every step where reasoning went wrong.

# What counts as an error

A step is erroneous when the agent introduces, propagates, or commits to incorrect reasoning. Look for:

- **Unsupported commitment**: the agent presents a conclusion or answer without sufficient grounding from its tool results. The final report commits to claims the trajectory does not actually support.
- **Source misreading**: the agent claims a source says something it does not, or misquotes/misparaphrases a tool result.
- **Constraint misinterpretation**: the agent misunderstands a requirement from the question — confusing timeframes, geographies, inclusion/exclusion criteria, or quantitative thresholds.
- **Premature narrowing**: the agent fixates on one candidate too early, stopping exploration before checking alternatives that the question demands.
- **Evidence fabrication**: the agent states facts that appear in no tool result — names, dates, relationships, or statistics not present in any retrieved source.
- **Verification gap**: the agent claims to have verified something but the visible tool results show no actual verification attempt, or the attempt failed.
- **Harmful continuation**: the agent carries forward an erroneous intermediate result into later reasoning, compounding the original mistake.

A finalize or report step that commits to errors from earlier steps is itself an error.

# Workflow

1. Read through the full trajectory using `list_turns` and `get_turn` for each step
2. For each step, check:
   - Does the agent introduce new factual claims? Are they backed by tool results?
   - Does the agent correctly interpret the question's constraints?
   - Does the agent accurately represent what tool results say?
   - When the agent commits to a conclusion, does the evidence chain actually support it?
3. Use the index tools (`list_entities`, `list_attention_hints`) to check the grounding status of entities
4. Identify ALL error steps — both where errors originate and where they are committed to

# Key principles

- **Trace the root cause**: when the final answer is wrong, find the earliest step where the error entered. Flag both the origin step and any later steps that commit to the error.
- **Check every constraint**: for multi-constraint questions, verify each constraint is addressed. A missed or misinterpreted constraint is an error even if the final answer happens to be correct.
- **Tool results are ground truth**: if a tool returned data, that data is factual. If the agent's summary contradicts the tool output, the agent is wrong.
- **Absence of evidence is not evidence**: when tool searches fail or return nothing, the agent should not conclude anything from the absence. Treating "no results found" as confirmation is an error.
- **Favor recall**: include both the first harmful step and any later steps that carry or commit to the same error. It is better to flag a borderline step than to miss a real error.

# Submit

Call `submit_verdict` exactly once as your final action.

- `surface_reminder`: true.
- `reminder_text`: concise summary of errors found, organized by step index.
- `evidence`: one item per error step — cite the turn index and what is wrong.
- `matched_event_ids`: 0-based turn indices of ALL steps that contain errors. This is the primary output.
- `continuation_notes`: submit `["posthoc review complete"]`.
