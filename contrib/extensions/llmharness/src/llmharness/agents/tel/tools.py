"""TEL agent tools: span store (list / get / search) and submit_error_spans."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _err(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _make_tool(
    name: str,
    description: str,
    params: type[BaseModel],
    handler: Callable[..., ToolResult | ToolTerminate],
    *,
    metadata: dict[str, Any] | None = None,
) -> FunctionTool:
    """Build a FunctionTool with automatic Pydantic validation."""

    async def _fn(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            parsed = params.model_validate(args)
        except ValidationError as exc:
            return _err(f"{name} rejected: {exc}")
        return handler(parsed)

    kw: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": params,
        "fn": _fn,
    }
    if metadata is not None:
        kw["metadata"] = metadata
    return FunctionTool(**kw)


# ---------------------------------------------------------------------------
# Span store
# ---------------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")


@dataclass
class SpanStore:
    """In-memory store of trajectory spans for one TELBench case."""

    spans: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)

    @property
    def valid_ids(self) -> set[str]:
        return {s["id"] for s in self.spans}

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

    def check_ids(self, ids: list[str]) -> list[str]:
        """Return span IDs not present in the store."""
        valid = self.valid_ids
        return [sid for sid in ids if sid not in valid]


SPAN_STORE_SERVICE_KEY: Final = "llmharness.tel_span_store"


# ---------------------------------------------------------------------------
# Notepad — externalises cross-span observations so mismatches invisible
# span-by-span become visible on review.
# ---------------------------------------------------------------------------


@dataclass
class Notepad:
    notes: list[str] = field(default_factory=list)
    path: Path | None = None

    def add(self, text: str) -> int:
        self.notes.append(text.strip())
        self._persist(self.render())
        return len(self.notes)

    def render(self) -> str:
        if not self.notes:
            return "(empty)"
        return "\n".join(f"- {n}" for n in self.notes)

    def _persist(self, rendered: str) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f"# notepad\n\n{rendered}\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("tel notepad persist failed ({}): {}", self.path, exc)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ListSpansArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetSpanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span_id: str = Field(description="The span ID to retrieve (e.g. 's003').")


class SearchSpansArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        description=(
            "Case-insensitive keyword or phrase to search for across all spans. "
            "Returns matching spans with surrounding context."
        ),
    )


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(
        description=(
            "One short observation to add to your notepad: what a span did and "
            "anything worth scrutinising. Reference the span id (e.g. 's003')."
        ),
    )


class SubmitErrorSpansArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error_span_ids: list[str] = Field(
        description=(
            "List of span IDs where the agent itself committed an error "
            "(e.g. ['s003', 's008']) — the culprit spans, NOT their "
            "downstream consequences."
        ),
    )
    reasoning: str = Field(
        description="Brief explanation of what errors were found and why.",
    )


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------

SUBMIT_TOOL_NAME: Final = "submit_error_spans"
_TERMINAL: Final[dict[str, Any]] = {"terminates": True}


def build_list_spans_tool(store: SpanStore) -> FunctionTool:
    def handler(_: ListSpansArgs) -> ToolResult:
        return _ok(json.dumps(store.list_spans(), ensure_ascii=False))

    return _make_tool(
        "list_spans",
        "List all trajectory spans with their IDs, stage labels, and "
        "a short preview. Use this first to get an overview of the trajectory.",
        ListSpansArgs, handler,
    )


def build_get_span_tool(store: SpanStore) -> FunctionTool:
    def handler(p: GetSpanArgs) -> ToolResult:
        span = store.get_span(p.span_id)
        if span is None:
            return _err(
                f"span {p.span_id!r} not found. Available: {sorted(store.valid_ids)}"
            )
        return _ok(json.dumps(span, ensure_ascii=False))

    return _make_tool(
        "get_span",
        "Retrieve the full content of a specific span by ID. "
        "Returns the span's stage label and complete raw text.",
        GetSpanArgs, handler,
    )


def build_search_spans_tool(store: SpanStore) -> FunctionTool:
    def handler(p: SearchSpansArgs) -> ToolResult:
        matches = store.search_spans(p.query)
        if not matches:
            return _ok(f"No spans contain {p.query!r}.")
        return _ok(json.dumps(matches, ensure_ascii=False))

    return _make_tool(
        "search_spans",
        "Search all spans for a keyword or phrase (case-insensitive). "
        "Returns matching span IDs with context around the match.",
        SearchSpansArgs, handler,
    )


def build_note_tool(notepad: Notepad) -> FunctionTool:
    def handler(p: NoteArgs) -> ToolResult:
        count = notepad.add(p.text)
        return _ok(f"note saved ({count} total)")

    return _make_tool(
        "note",
        "Append one observation to your notepad as you read the trajectory. "
        "Returns only a short acknowledgement; the full notepad is persisted "
        "for the next pass.",
        NoteArgs, handler,
    )


def build_submit_tool(store: SpanStore) -> FunctionTool:
    def handler(p: SubmitErrorSpansArgs) -> ToolTerminate | ToolResult:
        invalid = store.check_ids(p.error_span_ids)
        if invalid:
            return _err(f"Unknown span IDs: {invalid}. Valid: {sorted(store.valid_ids)}")
        return ToolTerminate(
            result=_ok(json.dumps({
                "error_span_ids": p.error_span_ids,
                "reasoning": p.reasoning,
            }, ensure_ascii=False)),
            reason="llmharness:submit_error_spans",
        )

    return _make_tool(
        SUBMIT_TOOL_NAME,
        "Submit your final prediction of error span IDs. "
        "Call exactly ONCE as your final action after analysis.",
        SubmitErrorSpansArgs, handler, metadata=_TERMINAL,
    )


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------


class TelToolsConfig(BaseModel):
    model_config = {"extra": "allow"}
    notepad_dir: str = ""


MANIFEST = ExtensionManifest(
    name="tel_tools",
    description="Register TEL span store tools and submit_error_spans.",
    registers=(
        "tool:list_spans",
        "tool:get_span",
        "tool:search_spans",
        "tool:note",
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
    npath = Path(config.notepad_dir) / f"{api.session_id}.md" if config.notepad_dir else None
    api.register_tool(build_note_tool(Notepad(path=npath)))
