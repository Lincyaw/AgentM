# Task 2B: Implement Context Compression

**Status**: PENDING
**Depends on**: 1A
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [orchestrator.md](../designs/orchestrator.md) § Context Compression, [sub-agent.md](../designs/sub-agent.md) § Context Compression
**Assignee**: implementer

## Objective

Implement all 4 functions in `src/agentm/core/compression.py`.

## Functions

### `count_tokens(messages, model="gpt-4") -> int`
Use `tiktoken` (`cl100k_base` fallback). Sum token counts across all messages' `content` field. Handle both `BaseMessage` and plain dicts.

### `sub_agent_compression_hook(state) -> dict`
`pre_model_hook` for Sub-Agents. Check token count vs threshold (80% of 128k). Below → `{'messages': state['messages']}`. Above → `{'llm_input_messages': compressed}`. Phase 1: just pass-through, defer LLM summarization.

### `build_compression_hook(config: CompressionConfig) -> Callable`
Closure capturing `config`, calls `sub_agent_compression_hook` with bound threshold/model settings.

### `compress_completed_phase(notebook, completed_phase) -> DiagnosticNotebook`
Extract steps for `completed_phase`, build `PhaseSummary`, return new notebook with pruned history. Immutable pattern.

## Verification

- `count_tokens` returns positive int for non-empty messages
- `build_compression_hook` returns callable
- `compress_completed_phase` returns new notebook (original unchanged)
- `test_interface_consistency.py::TestCompressionHookSignature` passes

## Notes

- Add `tiktoken` dependency: `uv add tiktoken`
- LLM summarization deferred to later phase
