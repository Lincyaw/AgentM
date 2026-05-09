# Issue #92 — Retry Policy Atom and Provider Wiring

## Outcome

- Added `agentm.core.abi.retry.RetryPolicy` as the policy port for retrying async provider operations.
- Added the single-file builtin `retry_policy` atom, which registers an exponential-backoff implementation through `ExtensionAPI.set_service("retry_policy", ...)` and can optionally wrap provider stream functions.
- Moved OpenAI rate-limit retry out of `llm/openai.py` module globals and into injected policy wiring; OpenAI and Anthropic now use provider-typed `RateLimitError` predicates.
- Made `verify_ssl=False` visible at composition time by emitting a warning `DiagnosticEvent` from the OpenAI provider install path.

## Verification

- `grep -R -n "_RATE_LIMIT_\|_is_rate_limit\|_create_with_retry" src/agentm/llm/` returns no hits.
- `validate_builtin()` reports 0 issues with `core-manifest.yaml` configured.
- Unit tests cover OpenAI retry, Anthropic retry parity, and the `verify_ssl=False` diagnostic warning.
