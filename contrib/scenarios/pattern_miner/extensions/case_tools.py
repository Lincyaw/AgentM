# code-health: ignore-file[AM025] -- atom tools validate untyped bundle JSON payloads
"""Scenario-local ``case_tools`` atom -- read-only access to one mining case.

The runner prepares a case bundle directory and starts the session with
``cwd`` pointing at it:

- ``task.md``          task description shown to the original agent
- ``trajectory.jsonl`` rendered trajectory records, one JSON object per line
- ``eval.json``        evaluation evidence (reward, verifier, judge)
- ``agent.patch``      final patch produced by the agent
- ``oracle.patch``     ground-truth patch (offline-only evidence)

The tools expose exactly this bundle; the miner reads nothing else.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import (
    RESOURCE_WRITER,
    AtomAPI,
    FunctionTool,
    JsonValue,
    ToolResult,
)
from agentm.core.lib import error_result, pydantic_to_tool_schema, text_result
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="case_tools",
    description=(
        "Read-only tools over one mining case bundle: task text, rendered "
        "trajectory, evaluation evidence, and patches."
    ),
    registers=(
        "tool:case_overview",
        "tool:read_task",
        "tool:read_eval",
        "tool:read_patch",
        "tool:list_turns",
        "tool:read_turns",
        "tool:search_trajectory",
    ),
    requires=(RESOURCE_WRITER.capability,),
)

_CLIP_NOTE = "\n[truncated {} chars; pass full=true for the rest]"


def _clip(text: str, limit: int, full: bool) -> str:
    if full or len(text) <= limit:
        return text
    return text[:limit] + _CLIP_NOTE.format(len(text) - limit)


class _NoArgs(BaseModel):
    pass


@dataclass(slots=True)
class _TurnStats:
    tools: list[str] = dc_field(default_factory=list)
    errors: int = 0
    chars: int = 0


class _ReadTaskArgs(BaseModel):
    full: bool = Field(default=False, description="true: full task text")


class _ReadEvalArgs(BaseModel):
    section: str | None = Field(
        default=None,
        description="Top-level key of eval.json; omit for a clipped view of all sections",
    )
    full: bool = Field(default=False, description="true: no clipping")


class _ReadPatchArgs(BaseModel):
    which: str = Field(
        description="'agent' (the agent's final patch) or 'oracle' (ground truth, offline-only)"
    )


class _ListTurnsArgs(BaseModel):
    start: int = Field(default=0, description="Start turn index (inclusive)")
    limit: int = Field(default=80, description="Max turns to list")


class _ReadTurnsArgs(BaseModel):
    turn: int | None = Field(
        default=None, description="Read every record of one turn index"
    )
    offset: int = Field(default=0, description="Skip this many records")
    limit: int = Field(default=20, description="Max records to return")
    role: str | None = Field(
        default=None, description="Filter by record role: assistant, tool_result"
    )
    full: bool = Field(default=False, description="true: full record text")


class _SearchArgs(BaseModel):
    pattern: str = Field(description="Regular expression, case-insensitive")
    limit: int = Field(default=20, description="Max matches to return")
    context: int = Field(default=200, description="Chars of context around each match")


class _CaseToolsRuntime:
    def __init__(self, api: AtomAPI) -> None:
        self._api = api
        self._text_cache: dict[str, str | None] = {}
        self._records: list[dict[str, object]] | None = None

    # -- bundle access ----------------------------------------------------

    async def _read_text(self, name: str) -> str | None:
        if name in self._text_cache:
            return self._text_cache[name]
        try:
            reader = self._api.services.require_role(RESOURCE_WRITER)
            data = await reader.read(name)  # resolved against the session cwd
        except Exception as exc:  # noqa: BLE001
            logger.warning("case_tools: cannot read {}: {}", name, exc)
            self._text_cache[name] = None
            return None
        text = (
            data.decode("utf-8", errors="replace")
            if isinstance(data, (bytes, bytearray))
            else str(data)
        )
        self._text_cache[name] = text
        return text

    async def _load_records(self) -> list[dict[str, object]]:
        if self._records is not None:
            return self._records
        raw = await self._read_text("trajectory.jsonl")
        records: list[dict[str, object]] = []
        for line_no, line in enumerate((raw or "").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("case_tools: bad trajectory line {}: {}", line_no, exc)
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        self._records = records
        # The parsed records supersede the raw text; keep one copy, not two.
        self._text_cache.pop("trajectory.jsonl", None)
        return records

    @staticmethod
    def _record_text(record: Mapping[str, object]) -> str:
        parts: list[str] = [str(record.get("text", "") or "")]
        calls = record.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    parts.append(
                        f"[tool_call: {call.get('name', '')}"
                        f"({json.dumps(call.get('arguments', {}), ensure_ascii=False)})]"
                    )
        return "\n".join(p for p in parts if p)

    # -- tools ------------------------------------------------------------

    async def case_overview(self, args: dict[str, JsonValue]) -> ToolResult:
        del args
        task = await self._read_text("task.md")
        records = await self._load_records()
        turn_count = 0
        if records:
            turns = [r.get("turn") for r in records if isinstance(r.get("turn"), int)]
            turn_count = (max(turns) + 1) if turns else 0  # type: ignore[operator]
        eval_raw = await self._read_text("eval.json")
        reward = "unknown"
        if eval_raw:
            try:
                loaded = json.loads(eval_raw)
                if isinstance(loaded, dict):
                    reward = str(loaded.get("reward", "unknown"))
            except json.JSONDecodeError as exc:
                logger.warning("case_tools: eval.json unparseable: {}", exc)
        lines = [
            f"Case bundle at {self._api.ctx.cwd}",
            f"- reward: {reward}",
            f"- trajectory: {len(records)} records across ~{turn_count} turns",
            "- files: task.md, trajectory.jsonl, eval.json, agent.patch, oracle.patch",
            "",
            "Task description (clipped; read_task for the rest):",
            _clip(task or "(missing)", 2500, False),
        ]
        return text_result("\n".join(lines))

    async def read_task(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ReadTaskArgs.model_validate(args)
        task = await self._read_text("task.md")
        if task is None:
            return text_result("task.md is missing from this bundle")
        return text_result(_clip(task, 6000, parsed.full))

    async def read_eval(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ReadEvalArgs.model_validate(args)
        raw = await self._read_text("eval.json")
        if raw is None:
            return text_result("eval.json is missing from this bundle")
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            return text_result(f"eval.json is not valid JSON: {exc}")
        if not isinstance(loaded, dict):
            return text_result("eval.json has an unexpected shape")
        if parsed.section is not None:
            if parsed.section not in loaded:
                return text_result(
                    f"no section {parsed.section!r}; available: "
                    + ", ".join(sorted(loaded))
                )
            body = json.dumps(loaded[parsed.section], ensure_ascii=False, indent=1)
            return text_result(_clip(body, 8000, parsed.full))
        parts = [f"eval.json sections: {', '.join(sorted(loaded))}", ""]
        for key in sorted(loaded):
            body = json.dumps(loaded[key], ensure_ascii=False)
            parts.append(f"## {key}\n{_clip(body, 1200, False)}\n")
        return text_result("\n".join(parts))

    async def read_patch(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ReadPatchArgs.model_validate(args)
        name = {"agent": "agent.patch", "oracle": "oracle.patch"}.get(parsed.which)
        if name is None:
            return error_result("which must be 'agent' or 'oracle'")
        text = await self._read_text(name)
        if text is None:
            return text_result(f"{name} is missing from this bundle")
        return text_result(_clip(text, 20000, False))

    async def list_turns(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ListTurnsArgs.model_validate(args)
        records = await self._load_records()
        by_turn: dict[int, _TurnStats] = {}
        for record in records:
            turn = record.get("turn")
            if not isinstance(turn, int):
                continue
            stats = by_turn.setdefault(turn, _TurnStats())
            calls = record.get("tool_calls")
            if isinstance(calls, list):
                stats.tools.extend(
                    str(c.get("name", "")) for c in calls if isinstance(c, dict)
                )
            if record.get("is_error"):
                stats.errors += 1
            stats.chars += len(self._record_text(record))
        indexes = sorted(by_turn)
        window = [i for i in indexes if i >= parsed.start][: parsed.limit]
        lines = [f"{len(indexes)} turns total; showing {len(window)}"]
        for i in window:
            stats = by_turn[i]
            tool_str = ", ".join(stats.tools) if stats.tools else "-"
            err = f" errors={stats.errors}" if stats.errors else ""
            lines.append(f"  [{i}] tools=[{tool_str}]{err} chars={stats.chars}")
        return text_result("\n".join(lines))

    async def read_turns(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ReadTurnsArgs.model_validate(args)
        records = await self._load_records()
        if parsed.turn is not None:
            records = [r for r in records if r.get("turn") == parsed.turn]
        if parsed.role is not None:
            records = [r for r in records if r.get("role") == parsed.role]
        total = len(records)
        window = records[parsed.offset : parsed.offset + parsed.limit]
        parts = [
            f"{total} records (showing {parsed.offset}..{parsed.offset + len(window) - 1})"
        ]
        for record in window:
            role = record.get("role", "?")
            turn = record.get("turn", "?")
            tool = record.get("tool")
            head = f"[turn {turn}] {role}" + (f" <- {tool}" if tool else "")
            if record.get("is_error"):
                head += " (error)"
            parts.append(head)
            parts.append(_clip(self._record_text(record), 1500, parsed.full))
            parts.append("")
        return text_result("\n".join(parts))

    async def search_trajectory(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _SearchArgs.model_validate(args)
        try:
            pattern = re.compile(parsed.pattern, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            return error_result(f"bad pattern: {exc}")
        records = await self._load_records()
        hits: list[str] = []
        for record in records:
            text = self._record_text(record)
            match = pattern.search(text)
            if match is None:
                continue
            lo = max(0, match.start() - parsed.context)
            hi = min(len(text), match.end() + parsed.context)
            role = record.get("role", "?")
            tool = record.get("tool")
            head = f"[turn {record.get('turn', '?')}] {role}" + (
                f" <- {tool}" if tool else ""
            )
            hits.append(f"{head}\n...{text[lo:hi]}...")
            if len(hits) >= parsed.limit:
                break
        if not hits:
            return text_result("no matches")
        return text_result(f"{len(hits)} matching records\n\n" + "\n\n".join(hits))

    # -- wiring -----------------------------------------------------------

    def install(self) -> None:
        surface: tuple[tuple[str, str, type[BaseModel], object], ...] = (
            (
                "case_overview",
                (
                    "Orient in the case: reward, trajectory size, clipped task "
                    "text, and the bundle file list. Call this first."
                ),
                _NoArgs,
                self.case_overview,
            ),
            (
                "read_task",
                "Read the task description the original agent received.",
                _ReadTaskArgs,
                self.read_task,
            ),
            (
                "read_eval",
                (
                    "Read the evaluation evidence (ground truth, offline-only): "
                    "reward, per-check verifier results, judge output."
                ),
                _ReadEvalArgs,
                self.read_eval,
            ),
            (
                "read_patch",
                (
                    "Read the agent's final patch, or the oracle patch "
                    "(ground truth, offline-only)."
                ),
                _ReadPatchArgs,
                self.read_patch,
            ),
            (
                "list_turns",
                (
                    "Per-turn summary of the trajectory: tools called, error "
                    "counts, sizes. The map for deciding where to read."
                ),
                _ListTurnsArgs,
                self.list_turns,
            ),
            (
                "read_turns",
                (
                    "Read trajectory records: one whole turn, or paginate with "
                    "offset/limit, optionally filtered by role."
                ),
                _ReadTurnsArgs,
                self.read_turns,
            ),
            (
                "search_trajectory",
                (
                    "Regex search across all trajectory records; returns turn "
                    "indexes with context windows."
                ),
                _SearchArgs,
                self.search_trajectory,
            ),
        )
        for name, description, args_model, handler in surface:
            self._api.register_tool(
                FunctionTool(
                    name=name,
                    description=description,
                    parameters=pydantic_to_tool_schema(args_model),
                    fn=handler,  # type: ignore[arg-type]
                )
            )


def install(api: AtomAPI, config: Mapping[str, JsonValue] | None = None) -> None:
    del config
    _CaseToolsRuntime(api).install()
