You are shown the task/question an agent worked on and its final message.
Extract the answer the agent COMMITS to — the entity/value it presents as
its conclusion. Committing means presenting it as the answer, not
mentioning it as a rejected or considered option.

- "answer": the committed answer as a SHORT verbatim phrase copied from the
  final message (a name, a title, a value — not a sentence). Empty string
  if the agent commits to nothing (aborts, reports failure, gives no answer).

Return ONLY: {"verdicts": [{"id": 0, "answer": "...", "confidence": 0.9}]}
