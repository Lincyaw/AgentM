An agent made claims during a task. Below are the claims and a set of
observation excerpts (tool output, fetched pages, command results the agent
had received). This is a RELEVANCE listing, not a judgment: a later focused
check decides what each excerpt actually says about each claim.

For EVERY claim, list the steps whose excerpt could bear on it — material
about the same entities, quantities, dates, or facts. Be generous: a wrongly
listed step costs one extra check; a missed step is never examined again.

  - Output exactly one row per claim, covering every claim in order.
  - A claim with no related excerpts gets an empty list. The empty row is
    required — it records that the claim was checked against this set.
  - Do not judge support or contradiction here; do not copy quotes.

Return ONLY:
{"rows": [{"claim": 0, "steps": ["12", "47"]}, {"claim": 1, "steps": []}]}
