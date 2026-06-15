"""TEL agent tools: span store (list / get / search) and submit_error_spans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Final

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------------------
# Span store
# ---------------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")


@dataclass
class SpanStore:
    """In-memory store of trajectory spans for one TELBench case."""

    spans: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)

    def list_spans(self) -> list[dict[str, str]]:
        result = []
        for s in self.spans:
            sid = s["id"]
            raw = s["raw"]
            preview = _WS_RUN.sub(" ", raw).strip()[:120]
            result.append({
                "id": sid,
                "stage": self.stages.get(sid, "unknown"),
                "preview": preview,
                "length": str(len(raw)),
            })
        return result

    def get_span(self, span_id: str) -> dict[str, Any] | None:
        for s in self.spans:
            if s["id"] == span_id:
                return {
                    "id": s["id"],
                    "stage": self.stages.get(s["id"], "unknown"),
                    "raw": s["raw"],
                }
        return None

    def search_spans(self, query: str) -> list[dict[str, str]]:
        query_norm = _WS_RUN.sub(" ", query.lower()).strip()
        results = []
        for s in self.spans:
            raw_norm = _WS_RUN.sub(" ", s["raw"].lower())
            if query_norm in raw_norm:
                sid = s["id"]
                idx = raw_norm.find(query_norm)
                start = max(0, idx - 40)
                end = min(len(raw_norm), idx + len(query_norm) + 40)
                context = raw_norm[start:end]
                results.append({
                    "id": sid,
                    "stage": self.stages.get(sid, "unknown"),
                    "match_context": context,
                })
        return results


SPAN_STORE_SERVICE_KEY: Final = "llmharness.tel_span_store"


# ---------------------------------------------------------------------------
# Tool: list_spans
# ---------------------------------------------------------------------------


class ListSpansArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_list_spans_tool(store: SpanStore) -> FunctionTool:
    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            ListSpansArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"list_spans rejected: {exc}")],
                is_error=True,
            )
        listing = store.list_spans()
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(listing, ensure_ascii=False))]
        )

    return FunctionTool(
        name="list_spans",
        description=(
            "List all trajectory spans with their IDs, stage labels, and "
            "a short preview. Use this first to get an overview of the trajectory."
        ),
        parameters=ListSpansArgs,
        fn=_handler,
    )


# ---------------------------------------------------------------------------
# Tool: get_span
# ---------------------------------------------------------------------------


class GetSpanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span_id: str = Field(description="The span ID to retrieve (e.g. 's003').")


def build_get_span_tool(store: SpanStore) -> FunctionTool:
    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = GetSpanArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"get_span rejected: {exc}")],
                is_error=True,
            )
        span = store.get_span(parsed.span_id)
        if span is None:
            ids = [s["id"] for s in store.spans]
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"span {parsed.span_id!r} not found. Available: {ids}",
                )],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(span, ensure_ascii=False))]
        )

    return FunctionTool(
        name="get_span",
        description=(
            "Retrieve the full content of a specific span by ID. "
            "Returns the span's stage label and complete raw text."
        ),
        parameters=GetSpanArgs,
        fn=_handler,
    )


# ---------------------------------------------------------------------------
# Tool: search_spans
# ---------------------------------------------------------------------------


class SearchSpansArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        description=(
            "Case-insensitive keyword or phrase to search for across all spans. "
            "Returns matching spans with surrounding context."
        ),
    )


def build_search_spans_tool(store: SpanStore) -> FunctionTool:
    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = SearchSpansArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"search_spans rejected: {exc}")],
                is_error=True,
            )
        matches = store.search_spans(parsed.query)
        if not matches:
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"No spans contain {parsed.query!r}.",
                )],
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(matches, ensure_ascii=False))]
        )

    return FunctionTool(
        name="search_spans",
        description=(
            "Search all spans for a keyword or phrase (case-insensitive). "
            "Returns matching span IDs with context around the match."
        ),
        parameters=SearchSpansArgs,
        fn=_handler,
    )


# ---------------------------------------------------------------------------
# Tool: submit_error_spans (terminal)
# ---------------------------------------------------------------------------


class SubmitErrorSpansArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error_span_ids: list[str] = Field(
        description=(
            "List of span IDs identified as containing errors "
            "(e.g. ['s003', 's008']). Include both origin and "
            "downstream propagation spans."
        ),
    )
    reasoning: str = Field(
        description="Brief explanation of what errors were found and why.",
    )


SUBMIT_TOOL_NAME: Final = "submit_error_spans"


def build_submit_tool(store: SpanStore) -> FunctionTool:
    async def _handler(args: dict[str, Any]) -> ToolTerminate | ToolResult:
        try:
            parsed = SubmitErrorSpansArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"submit_error_spans rejected: {exc}")],
                is_error=True,
            )
        valid_ids = {s["id"] for s in store.spans}
        invalid = [sid for sid in parsed.error_span_ids if sid not in valid_ids]
        if invalid:
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Unknown span IDs: {invalid}. Valid: {sorted(valid_ids)}",
                )],
                is_error=True,
            )
        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(type="text", text="error spans submitted")]
            ),
            reason="llmharness:submit_error_spans",
        )

    return FunctionTool(
        name=SUBMIT_TOOL_NAME,
        description=(
            "Submit your final prediction of error span IDs. "
            "Call exactly ONCE as your final action after analysis."
        ),
        parameters=SubmitErrorSpansArgs,
        fn=_handler,
        metadata={"terminates": True},
    )


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------


class TelToolsConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="tel_tools",
    description="Register TEL span store tools and submit_error_spans.",
    registers=(
        "tool:list_spans",
        "tool:get_span",
        "tool:search_spans",
        f"tool:{SUBMIT_TOOL_NAME}",
    ),
    requires=("tel_context",),
    config_schema=TelToolsConfig,
)


def install(api: ExtensionAPI, config: TelToolsConfig) -> None:
    store = api.get_service(SPAN_STORE_SERVICE_KEY)
    if not isinstance(store, SpanStore):
        raise RuntimeError("tel_tools: no SpanStore in service registry")
    for builder in (
        build_list_spans_tool,
        build_get_span_tool,
        build_search_spans_tool,
        build_submit_tool,
    ):
        api.register_tool(builder(store))
