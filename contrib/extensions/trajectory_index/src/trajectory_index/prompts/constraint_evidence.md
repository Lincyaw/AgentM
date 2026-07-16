An agent answered a question with the candidate named below. You are given
the question's constraints and excerpts from the tool/search results the
agent gathered (its evidence). For each constraint, judge what THESE EXCERPTS
establish about the candidate — judge only the presented content:

  - "establish": the excerpts contain facts showing the candidate satisfies
    the constraint. Quote the decisive fact verbatim in "quote".
  - "refute": the excerpts contain facts showing the candidate does NOT
    satisfy it. Quote the decisive fact verbatim in "quote".
  - "absent": these excerpts contain NO evidence bearing on this constraint
    at all — not about its topic, not about any relevant entity or value.

"absent" is a strong negative: it means you searched these excerpts and found
nothing relevant. If an excerpt mentions the right topic but doesn't settle
the constraint, use "establish" or "refute" with the closest available fact,
or leave "quote" empty with outcome "absent".

For constraints involving dates or numbers, "quote" must be the MINIMAL
phrase containing only the decisive value (e.g. "born 1965", not the whole
sentence) — and never do the comparison arithmetic yourself.
List the excerpt ids you relied on in "steps".

Return ONLY:
{"verdicts": [{"id": 0, "outcome": "establish|refute|absent", "quote": "...", "steps": ["3"], "confidence": 0.9, "reason": "..."}]}
