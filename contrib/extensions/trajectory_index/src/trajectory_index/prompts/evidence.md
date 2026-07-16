You are given observation excerpts from an agent's task execution. Match these
excerpts against the items below.

For EVERY claim, find excerpts bearing on it and judge adversarially — try to
REFUTE the pairing before accepting it.

  - "supports": the excerpt states the same specific fact about the same entities
  - "conflicts": the excerpt states something incompatible with the claim
  - "neutral": same topic but doesn't settle the claim

When the claim asserts a specific value (date, number, status, name, count),
find that attribute in the excerpt and COMPARE. A different value is "conflicts"
— the most commonly missed contradiction.

For supports/conflicts, copy the decisive passage VERBATIM into "quote". It is
checked mechanically; a paraphrase is discarded. The quote must be observation
content, NEVER the claim's own sentence. A claim with no matches gets an empty
matches list — the row is still required.

If constraints are listed, judge what THESE EXCERPTS establish about the
candidate for each constraint:

  - "establish": facts showing the candidate satisfies the constraint.
    Quote the decisive fact verbatim.
  - "refute": facts showing the candidate does NOT satisfy it.
    Quote the decisive fact verbatim.
  - "absent": NO evidence bearing on this constraint at all.

For date/number constraints, "quote" must be the MINIMAL phrase with the
decisive value only. List excerpt ids in "steps". Never do comparison
arithmetic yourself.

Return ONLY:
{"claims": [{"id": 0, "matches": [{"step": "12", "relation": "supports|conflicts|neutral", "quote": "..."}]}], "constraints": [{"id": 0, "outcome": "establish|refute|absent", "quote": "...", "steps": ["3"], "confidence": 0.9, "reason": "..."}]}
