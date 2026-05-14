# HYPOTHESIZE state

Propose at least one hypothesis. Each hypothesis MUST declare at least one
prediction with `polarity: "negative"` — an observation which, if found,
would **rule the hypothesis out**. The gate will reject any hypothesis
that has only positive predictions; a hypothesis without a negative
prediction is unfalsifiable.

Good negative predictions are concrete:

- "if H is true, the pod's CPU usage in the 5 minutes before the alert
  should NOT show idle spikes" (negative)
- "if H is true, we should NOT see successful health-check responses for
  service X during the window" (negative)
- "the logrotate.service systemd unit is active" (positive — pair with a
  negative one below)

Avoid vacuous negatives ("the universe still exists") and avoid framing
your prediction in a way no realistic observation could violate.

Once your hypothesis is accepted, the trace advances to VERIFY. Available
tools: `propose_hypothesis`, `record_observation`, `query_sql`,
`list_tables` (run a quick SQL check before proposing if you need to
sanity-check a prediction is observable).
