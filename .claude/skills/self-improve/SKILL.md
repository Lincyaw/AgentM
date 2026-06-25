---
name: self-improve
description: >
  Review agent trajectory and harness logs to find behavioral anti-patterns,
  then produce actionable improvements (prompt edits, scenario config, atom
  additions). Use when asked to analyze why an agent underperformed, find
  patterns across multiple runs, audit tool usage efficiency, or generate
  improvement recommendations. Triggers on: 提升, 改进, anti-pattern,
  自我检查, 为什么表现差, 优化 agent, review trajectory, improve agent,
  analyze runs, what went wrong across runs.
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
agentm trace logs --session <sid> --format ndjson > logs.ndjson

# All sessions in a trace tree
agentm trace index --format ndjson | jq -r --arg t "$TID" \
  'select(.trace_id==$t) | .session_id'
```

## Detection dimensions

Analyze the trajectory along these axes. Each axis has concrete signals
to look for and concrete actions to take.

### 1. Tool usage efficiency

**Signals:**
- Same tool called with identical or near-identical args multiple times
  (wasted tokens; the agent forgot it already queried this)
- Tool result ignored — agent calls a tool but the next assistant message
  doesn't reference its output
- Exit code / `is_error` ignored — agent doesn't adjust after a failed
  tool call (retries the same command, or proceeds as if it succeeded)
- Large tool results that get truncated — agent should have used a more
  precise query (e.g. `grep` instead of `cat`, `--limit` flag)

**Actions:** add a `<system-reminder>` via turn_reminder atom; adjust
tool descriptions to emphasize checking results; add a tool_filter to
remove tools the agent misuses.

### 2. Information acquisition timing

**Signals:**
- Critical information (the data that ultimately drives the conclusion)
  first appears late in the trajectory — agent spent early turns on
  low-value actions
- Agent reads the same file/data source multiple times across turns
  (should have cached or extracted what it needed the first time)
- Agent asks for data it already has in its context (re-querying what a
  prior tool already returned)

**Actions:** restructure the scenario system prompt to front-load data
gathering instructions; add a skill with a checklist of "first things to
do"; adjust tool ordering in tool_index.

### 3. Error recovery and adaptation

**Signals:**
- After a tool error (is_error=True), agent retries the exact same call
  without changing arguments
- After a `user_rejected` error, agent retries the same tool
- Agent gets stuck in a retry loop (3+ identical or near-identical calls)
- Agent ignores stderr content that contains the fix hint

**Actions:** improve tool_error_messages to include more actionable
guidance; add a tool_bash_guard rule for the problematic pattern; adjust
the scenario prompt's error-handling instructions.

### 4. Reasoning quality

**Signals:**
- Agent's stated reasoning contradicts the tool output it just received
  (confabulation / reading from training data instead of the tool result)
- Agent makes a decision without having gathered the relevant evidence
  (conclusion before investigation)
- Agent abandons a promising lead without explanation
- Agent repeats the same hypothesis after evidence has refuted it

**Actions:** add a llmharness auditor with a prompt targeting the
specific failure mode; add a skill with domain-specific reasoning
guidelines; consider a stronger model for this scenario.

### 5. Context management

**Signals:**
- Token usage per turn grows monotonically with no compaction
  (check `turns` output for `input_tokens` trend)
- Agent re-reads large files it already has in context
- Compaction fires at a bad time (mid-critical-reasoning) and loses
  essential context — check `agentm trace spans --name compaction`
- Large tool results dominate the context budget (check
  `tools --format ndjson | jq '.result | length'`)

**Actions:** tune `tool_result_budget` max_chars; adjust
`llm_compaction` audit_interval_turns; add `tool_result_cap` limits for
specific tools; restructure long tool outputs into summaries.

### 6. Multi-agent coordination (workflow / sub_agent)

**Signals:**
- Child sessions repeat work the parent already did
- Child results not consumed by the parent (parent ignores the
  artifact/result a child produced)
- Workers all fail with the same error (indicates a systemic issue the
  orchestrator should detect and abort on)
- Excessive worker spawning — parent creates more children than needed

**Actions:** improve the workflow script's `agent()` prompts; add
dedup between parent and child; add circuit-breaker logic to the
workflow; pass more context via `atom_config`.

### 7. Termination behavior

**Signals:**
- Agent submits a result without having exhausted its investigation
  budget (premature conclusion)
- Agent keeps investigating after already having enough evidence
  (context waste / indecision)
- Agent hits max_turns without submitting (never converged)
- `stop_reason` in `turns` output shows `max_tokens` truncation
  (model output being cut off)

**Actions:** adjust loop_budget max_turns; improve the finalize tool's
prompt guidance; add a turn_reminder that fires near the turn budget
boundary; adjust the model's reasoning_effort.

## Analysis workflow

1. **Collect**: extract turns, tools, messages, logs for the target
   session(s).
2. **Orient**: `agentm trace stats` for event histogram; `usage` for
   cost; `turns` for the shape of the session.
3. **Detect**: walk each dimension above. Use `jq` pipelines to compute
   concrete metrics (duplicate tool calls, error rate, input_token
   growth, etc.).
4. **Prioritize**: rank findings by impact — a pattern that wastes 50%
   of turns matters more than a minor style issue.
5. **Prescribe**: for each finding, specify the concrete change (which
   file, what to add/remove/modify). Changes should be testable by
   re-running the same scenario.

## Output format

```markdown
## Trajectory Review: <session_id>

### Summary
- Turns: N, Tokens: X input / Y output, Tools: Z calls
- Outcome: <success/failure/timeout>

### Findings (by impact)

#### 1. <Finding title>
- **Dimension**: <which of the 7 above>
- **Evidence**: <specific turn numbers, tool calls, or log lines>
- **Impact**: <what it cost — wasted turns, wrong conclusion, etc.>
- **Fix**: <concrete action — file path, what to change>

#### 2. ...

### Recommended changes (ordered)
1. <highest-impact change>
2. ...
```
