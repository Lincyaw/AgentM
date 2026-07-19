---
name: policy-engine
description: >
  Policy engine authoring and debugging guide — write detection rules,
  understand the DSL syntax, query primitives, IFC labels, entity evidence,
  observe/enforce modes, and policy composition. Trigger when the user wants
  to: write a new policy rule, debug why a rule fires (or doesn't), check
  what query primitives are available, understand taint propagation, compose
  multiple policy files, or tune thresholds. Also trigger when the user says
  "policy", "规则", "检测规则", "policy engine", "add a rule", "why did it
  block", "policy stats", or mentions any rule name from base_policy.yaml.
---

# Policy Engine

Deterministic, DSL-driven detection over agent events. The engine evaluates
rules without LLM calls; LLM reasoning activates only on escalation (via
context injection, not a separate call).

## Quick Start

Policy files live at:
- `src/agentm/extensions/builtin/policy/base_policy.yaml` — ships with the engine
- `~/.agentm/policies/<name>.yaml` — user-authored
- `<scenario>/policies/<name>.yaml` — scenario-specific

Enable in a scenario manifest:
```yaml
extensions:
  - policy_engine

atom_config:
  policy_engine:
    policy_files:
      - policies/custom.yaml
```

The base policy (13 rules) loads automatically. User/scenario files layer
on top with composition semantics (can add rules, can disable non-abort rules).

## File Structure

```yaml
version: 1

labels:          # IFC taint labels (optional)
  secret:
    source: { tool: "read|bash", result_matches: '...' }
    min_length: 8

disable:         # names of inherited rules to disable (optional)
  - exploration-without-action

rules:           # detection rules
  - name: my-rule
    on: tool_call_pre        # event channel
    match: { tool: bash }    # fast-reject guard (optional)
    when: "..."              # Python expression predicate
    effect: block            # notify | block | escalate | abort
    mode: observe            # observe | enforce
    cooldown: 5 turns        # suppress re-fire (optional)
    reason: "..."            # diagnostic message template
    escalate_context: "..."  # expression for attached evidence (optional)
```

## Event Channels

| YAML `on:` value | Fires when | Can block? |
|---|---|---|
| `tool_call_pre` | Before a tool executes | Yes (block/abort) |
| `tool_call_post` | After a tool completes | No (notify/escalate only) |
| `turn_end` | After each agent turn commits | No |
| `session_spawn` | When a child session is created | Yes |

## Match Clause (Guard)

Fast-reject filter — checked BEFORE the `when` expression. Cannot access
state tables. Only checks event fields.

```yaml
match: { tool: "edit" }                # exact tool name
match: { tool: "read|edit|write" }     # OR over tool names
match: { tool: bash, args.cmd: "git*" } # tool + arg pattern
```

If `match` is omitted, the rule subscribes to ALL events on its channel.

## When Expression

Restricted Python expression evaluated per-event. Has access to the full
query namespace. Must evaluate to a truthy value for the rule to fire.

### Available Names

| Name | Type | Description |
|---|---|---|
| `event` | EventProxy | Current event (`.tool_name`, `.args`, `.result`, `.taint`) |
| `session` | SessionProxy | `.turn_count`, `.id` |
| `labels` | LabelsProxy | For IFC: `labels.secret in event.taint` |
| `tool_log` | TableQuery | Rolling-window tool call log (max 500) |
| `file_state` | FileStateQuery | Per-file read/write tracking |
| `effect_log` | EffectLogQuery | Rule firing history |
| `EMPTY` | sentinel | Falsy, `.attr` returns -1 |

### Query Primitives (on `tool_log` / `effect_log`)

| Method | Returns | Example |
|---|---|---|
| `.count(where={...}, last=N)` | int | `tool_log.count(where={tool: "read"}, last=10)` |
| `.count(where={...}, since=T)` | int | `tool_log.count(where={exit_code: "!=0"}, since=5)` |
| `.count(scope=user, ttl="14d")` | int | Cross-session count within TTL |
| `.distinct(field, last=N)` | int | `tool_log.distinct("args_hash", last=5)` |
| `.exists(where={...}, last=N)` | bool | `tool_log.exists(where={tool: "pytest*", exit_code: 0}, since=T)` |
| `.last(where={...})` | Entry\|EMPTY | `tool_log.last(where={tool: "edit"}).turn` |

### Standalone Primitives

| Function | Signature | Returns |
|---|---|---|
| `streak(where, last=N)` | → int | Consecutive matches from most recent backward |
| `trend(field, where, last=N)` | → str | "increasing" \| "decreasing" \| "stable" |
| `ratio(a, b)` | → float | Safe division (0 if b=0) |
| `sequence(steps, last=N)` | → bool | Ordered pattern match |
| `group(field, where, last=N, top=K)` | → list[(val, count)] | Group-by |
| `diff(set_a, set_b)` | → set | Set difference |
| `lookup(table, key, field)` | → value\|EMPTY | Cross-table point lookup |
| `entity_evidence(name)` | → EvidenceList\|EMPTY | Entity evidence records |
| `fingerprint(text)` | → str\|None | Normalized error hash |

### `file_state` Query

```python
file_state.get(path)           # → FileStateEntry | EMPTY
file_state.get(path).read_count
file_state.get(path).write_count
file_state.get(path).last_read_turn
file_state.get(path).last_write_turn
```

### Entity Evidence

```python
entity_evidence(path).has(type='tool_success')  # bool
entity_evidence(path).strongest()               # Evidence | EMPTY
```

Evidence types: `structural`, `tool_success`, `tool_failure`, `lexical_match`, `dict_recall`, `user_provided`.

### Where-Clause Pattern Language

| Pattern | Meaning | Example |
|---|---|---|
| `"exact"` | Exact string match | `{tool: "bash"}` |
| `"a\|b\|c"` | OR over literals | `{tool: "read\|edit\|write"}` |
| `"prefix*"` | Prefix glob | `{cmd: "git commit*"}` |
| `"*suffix"` | Suffix glob | `{path: "*.py"}` |
| `"*mid*"` | Contains | `{cmd: "*--force*"}` |
| `"!=value"` | Not equal | `{exit_code: "!=0"}` |
| `42` | Exact numeric | `{exit_code: 0}` |

### Field Names Available in `where`

```
tool, path, cmd, exit_code, error, error_fingerprint,
error_category, args_hash, turn, rule_id, effect,
mode, duration_ms, is_repeat, repeat_count, result_length
```

These are string constants in the eval namespace — bare names resolve
to themselves: `{tool: "read"}` means `{"tool": "read"}`.

### Expression Constraints

Allowed: comparisons, boolean ops, arithmetic, `in`/`not in`,
comprehension predicates (`any()/all()`), attribute access, dict
literals, string methods, query primitives.

Forbidden: `import`, assignment, function/class defs, lambdas, I/O,
`eval`/`exec`, `__dunder__` access.

## Effects

| Effect | Behavior | Agent sees |
|---|---|---|
| `notify` | Soft hint injected into context | Info-level diagnostic |
| `block` | Tool call rejected | Error with reason |
| `escalate` | Prominent warning + evidence | Warning-level diagnostic + `escalate_context` data |
| `abort` | Session terminated | RuntimeError (always enforced) |

## Mode: observe vs enforce

| Mode | Evaluates? | Logs? | Produces effect? |
|---|---|---|---|
| `enforce` | yes | yes (effect_log) | yes |
| `observe` | yes | yes (effect_log) | no |

Rules with `effect: abort` are ALWAYS enforced regardless of mode.

Use `observe` for new/unvalidated rules. Graduate to `enforce` once
precision data confirms they work.

## IFC Labels (Information Flow Control)

Track secret values via literal substring matching:

```yaml
labels:
  secret:
    source:
      tool: "read|bash"
      result_matches: '(?:API_KEY|TOKEN)=(\S{8,})'
    min_length: 8
```

When a tool result matches `result_matches`, the captured group (the
actual secret value) is tracked. Subsequent tool calls whose args
contain that literal string carry the `secret` taint.

Query in rules: `"'secret' in event.taint"`

## Reason Template

Interpolates event args:
```yaml
reason: "Editing {event.args[path]} without reading first."
reason: "Blocked: {event.args[cmd]}"
reason: "Context at {session.turn_count} turns."
```

Simple `{key}` also works for top-level arg names.

## Policy Composition

Layers (in order):
1. Base policy (`base_policy.yaml`, ships with engine)
2. User/scenario policy files (from `atom_config.policy_files`)

Rules from later layers override earlier ones (same `name` → replace).
`disable: [rule-name]` removes inherited rules EXCEPT `abort`-effect
rules (safety rules cannot be weakened).

## Runtime Tools

| Tool | Purpose |
|---|---|
| `reload_policies` | Hot-reload rules from disk |
| `policy_explain(rule="name")` | Show rule state, last fired turn, guard |
| `policy_stats` | Turn count, rule counts, recent firings |

## Debugging

1. Check if a rule compiled: `policy_explain(rule="my-rule")`
2. Check effect_log: `policy_stats` shows recent firings
3. A rule doesn't fire? Check:
   - Channel matches (`on:` → correct event)
   - Guard matches (`match:` → tool name correct)
   - Predicate evaluates truthy
   - Cooldown not active
4. A rule fires too often? Add `cooldown: N turns`
5. Observe mode: rule evaluates + logs but doesn't affect agent

## Shipped Rules (base_policy.yaml)

| Rule | Mode | Effect | What it detects |
|---|---|---|---|
| edit-without-read | enforce | notify | Editing a file never read |
| dangerous-bash | enforce | block | rm -rf, force push, etc. |
| budget-warning | enforce | notify | Context >80% full |
| spawn-storm | enforce | block | >5 concurrent sub-agents |
| test-before-commit | observe | block | Commit without passing tests |
| stuck-loop | observe | escalate | 3+ failures, no strategy change |
| weak-grounding-edit | observe | escalate | Edit path with no tool evidence |
| premature-irreversible | observe | escalate | Push/delete at turn ≤4 |
| exploration-without-action | observe | notify | 10+ read turns, 0 writes |
| file-churn | observe | notify | Same file edited 3+ times |
| cross-session-repeat | observe | escalate | Error seen in 3+ sessions |
| no-secrets-to-network | enforce | abort | Tainted value in bash args |
| abort-on-persistent-stagnation | enforce | abort | stuck-loop escalated 2+ times |

## Common Patterns

### "Block X unless Y happened first"
```yaml
- name: commit-without-test
  on: tool_call_pre
  match: { tool: bash }
  when: |
    'git commit' in event.args.get('cmd', '')
    and not tool_log.exists(
      where={tool: "bash", cmd: "pytest*", exit_code: 0},
      since=tool_log.last(where={tool: "edit"}).turn
    )
  effect: block
  reason: "Run tests before committing."
```

### "Escalate after N consecutive failures"
```yaml
- name: stuck
  on: turn_end
  when: |
    streak(where={exit_code: "!=0"}, last=10) >= 4
  effect: escalate
  cooldown: 5 turns
  reason: "4+ consecutive failures."
```

### "Notify on resource exhaustion trend"
```yaml
- name: context-growing
  on: turn_end
  when: |
    trend("result_length", last=10) == "increasing"
    and tool_log.count(last=10) > 5
  effect: notify
  reason: "Tool output growing — watch context usage."
```

### "Evidence-gated write"
```yaml
- name: unverified-write
  on: tool_call_pre
  match: { tool: edit }
  when: "not entity_evidence(event.args.get('path', '')).has(type='tool_success')"
  effect: escalate
  escalate_context: "entity_evidence(event.args.get('path', ''))"
  reason: "No tool evidence for {event.args[path]}."
```

### "Cross-session learning"
```yaml
- name: known-bad-pattern
  on: tool_call_post
  when: |
    event.result.get('error', '')
    and tool_log.count(scope=user, ttl="7d") >= 2
  effect: escalate
  reason: "This error pattern recurs across sessions."
```
