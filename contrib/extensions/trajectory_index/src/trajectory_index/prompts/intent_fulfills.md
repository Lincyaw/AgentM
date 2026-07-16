An agent performed actions during a task. Each action has a declared purpose
and a tool result showing what actually happened. For each pair, judge whether
the action achieved its declared purpose based on the tool result. Be strict:
partial achievement or ambiguous output counts as "no".

Return ONLY:
{"verdicts": [{"id": 0, "outcome": "yes|no|unclear"}]}
