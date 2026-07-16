An agent made claims during a task. Below are the claims and a set of
observation excerpts (tool output, fetched pages, command results the agent
had received). For EVERY claim, find excerpts bearing on it and judge each
match adversarially — try to REFUTE the pairing before accepting it.

  - "supports": the excerpt itself states the same specific fact about the
    same entities as the claim.
  - "conflicts": the excerpt states something incompatible with the claim.
  - "neutral": on the same topic, but it does not settle the claim either way.

When the claim asserts a specific value — a date, number, status, name, or
count — find that same attribute in the excerpt and COMPARE the values. A
different value is a "conflicts", even when a matching-looking value also
appears nearby. Anchor on what the source actually reports, not on the value
that echoes the claim. This value mismatch is the most commonly missed
contradiction; do not overlook it.

  - For supports/conflicts, copy the decisive passage VERBATIM into "quote".
    It is checked mechanically; a paraphrase is discarded. The quote must be
    observation content, NEVER the claim's own sentence. For a value mismatch,
    quote the passage carrying the source's actual value.
  - A claim with no related excerpts gets an empty verdicts list. The empty
    row is required — it records that the claim was checked.

Return ONLY:
{"verdicts": [{"claim": 0, "matches": [{"step": "12", "relation": "supports|conflicts|neutral", "quote": "..."}]}]}
