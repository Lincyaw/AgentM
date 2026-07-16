An agent performed tool-call actions during a task. Each action has a declared
purpose and (when available) a tool result showing what actually happened.

You receive two sections to judge:

## Section A — addresses

(action, constraint) pairs. For each pair, judge whether the action's purpose
directly relates to satisfying the constraint. Be strict: a tangential or
incidental connection is not enough; the purpose must target the constraint's
requirement.

## Section B — fulfills

(action, tool_result) pairs. For each pair, judge whether the action achieved
its declared purpose based on the tool result. Be strict: partial achievement
or ambiguous output counts as "no".

Return ONLY:
{"addresses": [{"id": 0, "outcome": "yes|no|unclear"}], "fulfills": [{"id": 0, "outcome": "yes|no|unclear"}]}
