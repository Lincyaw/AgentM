# Example: per-project review standards

WorkGraph injects this file's content into coder and verifier context as
`review_standards` when the state directory contains `review_standards.md`
(or when `review_standards_file` / `args.review_standards` is set). Project
contract rules belong here — not in the generic agent prompts — so the
scenario stays reusable and the rules version together with the project.

The content below is the standards block used by the train-ticket DDD
benchmark project; replace it with your own project's rules.

---

Contract conformance: cross-context shapes must match `docs/08-contracts/`
(shared-primitives.md, events/, api/, messaging.md when they exist) — field
names, Money as currency+minorUnits, RFC3339 UTC timestamps, envelope
fields, SCREAMING_SNAKE enums, camelCase JSON.

EventEnvelope ground truth (do not contradict it): EXACTLY these 8 fields —
eventId, eventType, occurredAt, correlationId, causationId, producer,
schemaVersion, payload. causationId is OPTIONAL (omitting it is
conformant). sourceCommandId and attributes are NOT contract fields —
flagging their absence, or requiring causationId, is a false finding.

Recurring REJECTED patterns from past reviews — check each one:

- Facade adapters: an HTTP handler that builds domain-shaped records inline
  instead of calling the existing aggregate; wire the domain layer, and
  prove it with a test where a domain invariant violation surfaces as
  DOMAIN_RULE_VIOLATION.
- Stub subscribers: a consumer that ACKs events without real handling is a
  blocker, as is DLQ retry counting kept in memory instead of the broker's
  delivery count (Redis PEL).
- Idempotency shortcuts: Idempotency-Key must be REQUIRED on state-changing
  POSTs AND validated as UUID v7 (parse + version check; malformed or v4 ->
  canonical 400 VALIDATION_FAILED, with negative tests); cache a request
  fingerprint so same key + different body returns 422
  IDEMPOTENCY_KEY_REUSED; replay returns the original response. Never reuse
  the key as causationId or any envelope field. Generated IDs
  (evt-/cmd-/corr-) are UUID v7 everywhere, via one shared helper.
- Non-canonical envelope IDs: eventId `evt-`+UUID, correlationId
  `corr-`+UUID, causationId `cmd-`/`evt-` prefixed; occurredAt RFC3339 UTC;
  exactly the 8 contract envelope fields, publish only AFTER state is
  committed, and never swallow publish errors.
- Decorative runtime: writing publisher/subscriber classes is not
  integration — the production bootstrap/startup must construct the Redis
  publisher from REDIS_URL (default redis://localhost:6379) and start/stop
  the subscriber lifecycle for the streams in messaging.md's table.
  Injection/None defaults are for tests only. Optional envelope fields are
  OMITTED when absent, never empty strings.
- Documents must stay masked in logs, events, and error messages
  (e.g. maskedDocumentRef); if the Makefile has a `contract-lint` target,
  run `make contract-lint` — its violations are blockers.
