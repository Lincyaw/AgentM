# Issue 95 — Provider Wire-Policy Leak Cleanup

## Requirement

Provider wire modules must keep policy knobs explicit and keep registration types in the core ABI boundary:

- Anthropic thinking budgets are constructor/config policy, not a module-scope provider table.
- OpenAI assistant `ThinkingBlock` encoding is an explicit round-trip strategy with drop diagnostics.
- Provider registration records are declared in `agentm.core.abi.provider`; harness re-exports for compatibility.

## Implementation Notes

- Added `ProviderConfig` and lightweight `ProviderManifest` to `agentm.core.abi.provider`.
- Anthropic exposes `thinking_budgets` on `AnthropicStreamFn` and provider manifest config schema.
- OpenAI exposes `thinking_round_trip={drop,system_note,raise}` and emits one info diagnostic per stream function when dropping reasoning.
