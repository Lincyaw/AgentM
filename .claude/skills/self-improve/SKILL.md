---
name: self-improve
description: >
  Review agent trajectory to find behavioral anti-patterns, then produce
  actionable improvements (prompt edits, scenario config, atom additions).
  Use when asked to analyze why an agent underperformed, find patterns
  across multiple runs, audit tool usage efficiency, or generate
  improvement recommendations.
---

# self-improve

Analyze one or more agent session trajectories to find behavioral
anti-patterns and produce actionable improvements. This is not debugging
(use `self-debug` for that) — it is systematic review of *how* the agent
worked, not *whether* it crashed.

## Data acquisition

All evidence comes from `agentm trace`. For the command reference, load
the `trace-analysis` skill.

```bash
# Single session
agentm trace turns --session <sid> --format ndjson > turns.ndjson
agentm trace tools --session <sid> --format ndjson > tools.ndjson
agentm trace messages --session <sid> --format ndjson > msgs.ndjson

# All sessions under a parent
agentm trace sessions --parent <parent_id> --format ndjson
```

## Detection dimensions

Analyze the trajectory along these axes. Each axis has concrete signals
to look for and concrete actions to take.

### 1. Tool usage efficiency

**Signals:**
- Same tool called with identical or near-identical args multiple times
- Tool result ignored — agent calls a tool but doesn't reference output
- Exit code / `is_error` ignored — agent doesn't adjust after failure
- Large tool results that get truncated — agent should use a more
  precise query (e.g. `grep` instead of `cat`)

**Actions:** add a turn_reminder atom; adjust tool descriptions to
emphasize checking results; add a tool_filter to remove tools the agent
misuses.

### 2. Information acquisition timing

**Signals:**
- Critical information first appears late in the trajectory
- Agent reads the same file/data source multiple times across turns
- Agent asks for data it already has in its context

**Actions:** restructure the scenario system prompt to front-load data
gathering instructions; adjust tool ordering in tool_index.

### 3. Error recovery and adaptation

**Signals:**
- After a tool error, agent retries the exact same call unchanged
- Agent gets stuck in a retry loop (3+ identical calls)
- Agent ignores stderr content that contains the fix hint

**Actions:** improve tool_error_messages to include more actionable
guidance; add a tool_bash_guard rule; adjust the scenario prompt's
error-handling instructions.

### 4. Reasoning quality

**Signals:**
- Agent's stated reasoning contradicts the tool output it just received
- Agent makes a decision without having gathered the relevant evidence
- Agent abandons a promising lead without explanation
- Agent repeats the same hypothesis after evidence has refuted it

**Actions:** add domain-specific reasoning guidelines; consider a
stronger model for this scenario.

### 5. Context management

**Signals:**
- Token usage per turn grows monotonically with no compaction
  (check `turns` output for `input_tokens` trend)
- Agent re-reads large files it already has in context
- Large tool results dominate the context budget (check
  `tools --format ndjson | jq '.result | length'`)

**Actions:** adjust `llm_compaction` settings; add `tool_result_cap`
limits for specific tools; restructure long tool outputs into summaries.

### 6. Multi-agent coordination

**Signals:**
- Child sessions repeat work the parent already did
- Child results not consumed by the parent
- Workers all fail with the same error (systemic issue the parent should
  detect and abort on)
- Excessive worker spawning

**Actions:** improve child session prompts; add dedup between parent and
child; pass more context via `atom_config_overrides`.

### 7. Termination behavior

**Signals:**
- Agent submits a result without having exhausted its investigation
  budget (premature conclusion)
- Agent keeps investigating after already having enough evidence
- Agent hits max_turns without converging
- Turn outcome shows `ProviderRequestFailed` (model errors)

**Actions:** adjust LoopConfig max_turns; improve the system prompt's
completion guidance; add a turn_reminder near the turn budget boundary;
adjust the model's reasoning_effort.

## Analysis workflow

1. **Collect**: extract turns, tools, messages for the target session(s).
2. **Orient**: `usage` for cost; `turns` for shape and error count.
3. **Detect**: walk each dimension above. Use `jq` pipelines to compute
   concrete metrics (duplicate tool calls, error rate, input_token
   growth, etc.).
4. **Prioritize**: rank findings by impact.
5. **Prescribe**: for each finding, specify the concrete change (which
   file, what to add/remove/modify).

## Output format

```markdown
## Trajectory Review: <session_id>

### Summary
- Turns: N, Tokens: X input / Y output, Tools: Z calls
- Outcome: <success/failure/timeout>

### Findings (by impact)

#### 1. <Finding title>
- **Dimension**: <which of the 7 above>
- **Evidence**: <specific turn numbers, tool calls>
- **Impact**: <what it cost — wasted turns, wrong conclusion, etc.>
- **Fix**: <concrete action — file path, what to change>

#### 2. ...

### Recommended changes (ordered)
1. <highest-impact change>
2. ...
```
