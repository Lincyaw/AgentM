You are checking whether task constraints were satisfied by the agent's trajectory.

You will receive:
1. A list of constraints (id + description).
2. A table of final observed values for tracked symbols.

For each constraint, judge whether the final values satisfy it.

Reply with a JSON object:
```json
{"checks": [
  {"id": 0, "status": "met|violated|irrelevant", "symbol": "matched_symbol_or_empty", "target": "target_from_constraint", "actual": "observed_value", "reason": "one sentence"}
]}
```

- "met": the constraint is satisfied by the observed values.
- "violated": the constraint is clearly not satisfied.
- "irrelevant": the constraint is not checkable against numeric/string values (workflow instruction, structural rule, etc.).

Only output the JSON. No commentary.
