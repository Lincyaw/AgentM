# Issue 93 — LLM Stream Common Provider Internals

## Requirement

REQ-093-llm-provider-stream-common

## Change

- Added provider-internal `agentm.llm._common.StreamAccumulator` for assembling assistant messages from provider event mappers.
- Added `ToolSpecAdapter` plus shared `encode_tool_args(..., ensure_ascii=False)` and used it in Anthropic/OpenAI adapters.
- Added typed `ToolCallArgsParseError` stream/bus event for malformed streamed tool-call JSON while preserving `ToolCallBlock.arguments` as a dict.
- Injected provider clocks so `MessageEnd` timestamps are real provider timestamps rather than `0.0` placeholders.
- Kept Anthropic tool-result packing linear by tracking whether the previous emitted user message was synthetic tool-result content.

## Axis / Layer

- Axis: LLM stream.
- Layer: `agentm.llm` provider internals plus additive `agentm.core.abi.stream` event contract.
