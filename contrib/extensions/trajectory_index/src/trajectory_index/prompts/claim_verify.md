An agent made claims during a task. A relevance scan flagged observation
excerpts (tool output, fetched pages) as possibly bearing on them. For each
claim, judge each of its linked excerpts independently and adversarially: try
to REFUTE the pairing before accepting it.

  - "supports": the excerpt itself states the same specific fact about the
    same entities as the claim.
  - "conflicts": the excerpt states something incompatible with the claim.
  - "neutral": on the same topic, but it does not settle the claim either way
    (no specific value, status, or fact in common to compare).

When the claim asserts a specific value — a date, number, status, name, or
count — find that same attribute in the excerpt and COMPARE the values. A
different value is a "conflicts", even when a matching-looking value also
appears nearby: an excerpt may hold both a requested date and the actual date,
or an asked-for figure and the returned one. Anchor on what the source
actually reports, not on the value that echoes the claim. This value mismatch
is the most commonly missed contradiction; do not overlook it.

  - For supports/conflicts, copy the decisive passage VERBATIM into "quote".
    It is checked mechanically against the excerpt; a paraphrase is discarded.
    The quote must be observation content, NEVER the claim's own sentence or a
    restatement of it. For a value mismatch, quote the passage carrying the
    source's actual value.
  - Every linked excerpt gets exactly one verdict.

Return ONLY:
{"verdicts": [{"id": 0, "verdicts": [{"step": "12", "relation": "supports|conflicts|neutral", "quote": "..."}]}]}
