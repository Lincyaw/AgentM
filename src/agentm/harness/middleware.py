"""Harness middleware — new-protocol implementations for SimpleAgentLoop.

Each class implements the Middleware protocol (on_llm_start, on_llm_end,
on_tool_call) and overrides only the hooks it needs.  No dependency on
the legacy AgentMMiddleware base class or NodePipeline.

All middleware is self-contained and designed for composition via the
SimpleAgentLoop's sequential chaining (on_llm_start/end) and nested
wrapping (on_tool_call).
"""
from __future__ import annotations

import json
import logging
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

import tiktoken

from agentm.config.schema import CompressionConfig, ModelConfig
from agentm.core.trajectory import TrajectoryCollector
from agentm.harness.types import LoopContext

logger = logging.getLogger(__name__)

_THINK_TOOL = "think"


# ---------------------------------------------------------------------------
# 1. BudgetMiddleware
# ---------------------------------------------------------------------------


class BudgetMiddleware:
    """Injects urgency messages when step/tool budgets run low.

    Tracks budget via LoopContext fields (ctx.step, ctx.tool_call_count)
    rather than scanning message history.  Sets ``exhausted`` flag when
    any budget is fully consumed.
    """

    def __init__(
        self,
        max_steps: int,
        tool_call_budget: int | None = None,
    ) -> None:
        self._max_steps = max_steps
        self._tool_call_budget = tool_call_budget
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """True once any budget is fully consumed."""
        return self._exhausted

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        step_remaining = max(0, self._max_steps - ctx.step)
        tool_remaining = (
            max(0, self._tool_call_budget - ctx.tool_call_count)
            if self._tool_call_budget is not None
            else None
        )

        urgency = self._step_urgency(step_remaining)
        tool_urgency = self._tool_urgency(tool_remaining)

        if urgency is None and tool_urgency is None:
            return messages

        parts = [u for u in (urgency, tool_urgency) if u is not None]
        return [*messages, {"role": "human", "content": "\n".join(parts)}]

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        return await call_next(tool_name, tool_args)

    # -- internal helpers --

    def _step_urgency(self, remaining: int) -> str | None:
        ms = self._max_steps
        if remaining <= 0:
            self._exhausted = True
            return (
                f"BUDGET EXHAUSTED: All {ms} steps used. "
                "STOP immediately -- do NOT call any tool including think. "
                "Write your conclusion as plain text now."
            )
        if remaining <= 3:
            return (
                f"WARNING: {remaining}/{ms} steps left. "
                "Summarize your findings NOW. Do NOT call any more tools."
            )
        if remaining <= ms // 3:
            return (
                f"BUDGET: {remaining}/{ms} steps left. "
                "Start wrapping up -- prioritize the most important remaining queries."
            )
        return None

    def _tool_urgency(self, remaining: int | None) -> str | None:
        if remaining is None:
            return None
        budget = self._tool_call_budget
        if remaining <= 0:
            self._exhausted = True
            return (
                f"TOOL BUDGET EXHAUSTED: All {budget} tool calls used. "
                "STOP calling tools. Write your conclusion as plain text now."
            )
        if remaining <= 3:
            return (
                f"TOOL WARNING: {remaining}/{budget} tool calls left. "
                "Wrap up your investigation -- use remaining calls wisely."
            )
        if budget is not None and remaining <= budget // 3:
            return (
                f"TOOL BUDGET: {remaining}/{budget} tool calls left. "
                "Start wrapping up -- prioritize the most critical queries."
            )
        return None


# ---------------------------------------------------------------------------
# 2. CompressionMiddleware
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_PRESERVE_N = 2
_MAX_CHUNK_TOKENS = 100_000

_SUMMARIZE_PROMPT = (
    "Summarize the following agent execution history into a structured summary. "
    "Preserve: key findings, tool call results, data values, and decisions made. "
    "Be concise but retain specific data points (numbers, names, timestamps)."
)


def _count_tokens(messages: list[Any], model: str = "gpt-4o") -> int:
    """Count tokens across all messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            total += len(encoding.encode(content))
    return total


def _format_messages_for_summary(messages: list[Any]) -> list[str]:
    """Format messages into text lines for the summarization prompt."""
    formatted = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
        tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", None) or []
        if tool_calls:
            tool_info = ", ".join(tc.get("name", "?") for tc in tool_calls)
            formatted.append(f"[{role}] Called tools: {tool_info}")
        elif content:
            formatted.append(f"[{role}] {content}")
    return formatted


def _summarize_messages(
    messages: list[Any],
    model: str = "gpt-4o-mini",
    model_config: ModelConfig | None = None,
) -> str:
    """Summarize a list of messages using an LLM, chunking if needed."""
    from agentm.config.schema import create_chat_model

    formatted_lines = _format_messages_for_summary(messages)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for line in formatted_lines:
        line_tokens = len(encoding.encode(line))
        if current_tokens + line_tokens > _MAX_CHUNK_TOKENS and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    llm = create_chat_model(model=model, temperature=0, model_config=model_config)

    if len(chunks) == 1:
        prompt = f"{_SUMMARIZE_PROMPT}\n\nMessages:\n{chunks[0]}"
        result = llm.invoke([{"role": "human", "content": prompt}])
        content = result.content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)

    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        prompt = (
            f"{_SUMMARIZE_PROMPT}\n\n"
            f"This is part {i + 1} of {len(chunks)} of the execution history.\n\n"
            f"Messages:\n{chunk}"
        )
        result = llm.invoke([{"role": "human", "content": prompt}])
        content = result.content
        if isinstance(content, list):
            chunk_summaries.append(" ".join(str(part) for part in content))
        else:
            chunk_summaries.append(str(content))

    combined = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    final_prompt = (
        "Combine the following partial summaries into one coherent, structured summary. "
        "Remove redundancy but preserve all key data points.\n\n"
        f"{combined}"
    )
    result = llm.invoke([{"role": "human", "content": final_prompt}])
    content = result.content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


class CompressionMiddleware:
    """Compresses message history via LLM summarization when token count
    exceeds threshold.

    on_llm_start: replaces older messages with a summary when over threshold.
    Preserves the 8-section compression format from the original implementation.
    """

    def __init__(
        self,
        config: CompressionConfig,
        model_config: ModelConfig | None = None,
    ) -> None:
        self._config = config
        self._model_config = model_config
        self._threshold_tokens = int(
            _DEFAULT_CONTEXT_WINDOW * config.compression_threshold
        )
        self._preserve_n = config.preserve_latest_n

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        token_count = _count_tokens(messages, model=self._config.compression_model)

        if token_count < self._threshold_tokens:
            return messages

        if len(messages) <= self._preserve_n:
            return messages

        older = messages[: -self._preserve_n]
        recent = messages[-self._preserve_n :]

        summary_text = _summarize_messages(
            older,
            model=self._config.compression_model,
            model_config=self._model_config,
        )
        summary_msg = {
            "role": "system",
            "content": f"[Compressed History Summary]\n{summary_text}",
        }

        logger.info(
            "CompressionMiddleware: compressed %d messages (%d tokens) into summary",
            len(older),
            token_count,
        )
        return [summary_msg, *recent]

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        return await call_next(tool_name, tool_args)


# ---------------------------------------------------------------------------
# 3. LoopDetectionMiddleware
# ---------------------------------------------------------------------------


def _count_trailing_think_only(messages: list[Any]) -> int:
    """Count consecutive AI messages from the tail where the only tool is think."""
    streak = 0
    for msg in reversed(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role != "ai":
            continue
        tool_calls = (
            msg.get("tool_calls", [])
            if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None) or []
        )
        tool_names = {tc.get("name", "") for tc in tool_calls}
        if tool_names and tool_names != {_THINK_TOOL}:
            break
        streak += 1
    return streak


class LoopDetectionMiddleware:
    """Detects repetitive tool call patterns and think-stalls.

    on_llm_start: scans recent AI messages for repeated (tool_name, args)
    pairs and consecutive think-only rounds, injecting warnings.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_size: int = 15,
        think_stall_limit: int = 3,
    ) -> None:
        self._threshold = threshold
        self._window_size = window_size
        self._think_stall_limit = think_stall_limit

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        # Think-stall detection (more specific, fires first)
        streak = _count_trailing_think_only(messages)
        if streak >= self._think_stall_limit:
            warning_text = (
                f"THINK-STALL WARNING: You have called only `think` for the "
                f"last {streak} rounds without taking any action. Thinking "
                f"alone does not advance the investigation.\n\n"
                f"You MUST call an action tool NOW. "
                f"Do NOT call think again until you have taken an action."
            )
            return [*messages, {"role": "human", "content": warning_text}]

        # Exact-match loop detection
        ai_messages = []
        for msg in messages:
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "type", "")
            if role == "ai":
                ai_messages.append(msg)

        recent_ai = ai_messages[-self._window_size :] if self._window_size else ai_messages

        call_counter: Counter[str] = Counter()
        for msg in recent_ai:
            tool_calls = (
                msg.get("tool_calls", [])
                if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None) or []
            )
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                args_key = json.dumps(tc_args, sort_keys=True, default=str)
                call_counter[f"{tc_name}:{args_key}"] += 1

        repeated = [key for key, count in call_counter.items() if count >= self._threshold]

        if repeated:
            lines = []
            for key in repeated:
                name, _, args_json = key.partition(":")
                lines.append(f"- `{name}({args_json})`")
            warning_text = (
                "LOOP DETECTION WARNING: The following tool calls have been "
                f"repeated {self._threshold}+ times in the last {self._window_size} steps. "
                "You appear to be stuck in a loop. Stop and reconsider your "
                "approach -- try a different strategy or tool.\n" + "\n".join(lines)
            )
            return [*messages, {"role": "human", "content": warning_text}]

        return messages

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        return await call_next(tool_name, tool_args)


# ---------------------------------------------------------------------------
# 4. TrajectoryMiddleware
# ---------------------------------------------------------------------------


class TrajectoryMiddleware:
    """Records trajectory events for LLM calls and tool executions.

    on_llm_start: record llm_start event with message summary.
    on_llm_end: record llm_end event with response content.
    on_tool_call: record tool_start before call_next, tool_end after.
    """

    def __init__(
        self,
        trajectory: TrajectoryCollector,
        agent_path: list[str],
        task_id: str | None = None,
    ) -> None:
        self._trajectory = trajectory
        self._agent_path = agent_path
        self._task_id = task_id
        self._last_message_count = 0

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        total = len(messages)
        new_messages = messages[self._last_message_count :]
        self._last_message_count = total

        self._trajectory.record_sync(
            event_type="llm_start",
            agent_path=self._agent_path,
            data={
                "message_count": total,
                "new_message_count": len(new_messages),
                "messages": new_messages,
            },
            task_id=self._task_id,
        )
        return messages

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        tool_calls = getattr(response, "tool_calls", None) or []
        content = getattr(response, "content", "")

        if tool_calls:
            for tc in tool_calls:
                self._trajectory.record_sync(
                    event_type="tool_call",
                    agent_path=self._agent_path,
                    data={
                        "tool_name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    },
                    task_id=self._task_id,
                )
        elif content:
            self._trajectory.record_sync(
                event_type="llm_end",
                agent_path=self._agent_path,
                data={"content": str(content)},
                task_id=self._task_id,
            )
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        # Note: tool_call event is already recorded in on_llm_end when
        # the LLM decides to call a tool. Here we only record tool_result
        # after actual execution, matching the format the frontend expects.
        result = await call_next(tool_name, tool_args)
        self._trajectory.record_sync(
            event_type="tool_result",
            agent_path=self._agent_path,
            data={"tool_name": tool_name, "result": result},
            task_id=self._task_id,
        )
        return result


# ---------------------------------------------------------------------------
# 5. DedupMiddleware
# ---------------------------------------------------------------------------


class DedupTracker:
    """Per-task cache for tool call deduplication.

    Uses OrderedDict with FIFO eviction to bound memory.
    """

    def __init__(self, max_cache_size: int = 50) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_cache_size

    def make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Deterministic cache key -- arg order does not matter."""
        args_str = json.dumps(args, sort_keys=True, default=str)
        return f"{tool_name}:{args_str}"

    def has(self, key: str) -> bool:
        return key in self._cache

    def lookup(self, key: str) -> str | None:
        return self._cache.get(key)

    def store(self, key: str, result: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._cache)


_EXCLUDED_TOOLS = frozenset({"think"})


class DedupMiddleware:
    """Deduplicates tool calls via caching.

    on_tool_call: check cache -> return cached result if found, else
    call_next and cache the result.
    on_llm_start: prepend reminder about already-cached results (soft dedup).
    """

    def __init__(self, max_cache_size: int = 50) -> None:
        self._tracker = DedupTracker(max_cache_size=max_cache_size)

    @property
    def tracker(self) -> DedupTracker:
        return self._tracker

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        if self._tracker.size == 0:
            return messages

        # Scan for tool calls in the last AI message that are already cached
        reminders: list[str] = []
        for msg in reversed(messages):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "type", "")
            if role == "ai":
                tool_calls = (
                    msg.get("tool_calls", [])
                    if isinstance(msg, dict)
                    else getattr(msg, "tool_calls", None) or []
                )
                for tc in tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name in _EXCLUDED_TOOLS:
                        continue
                    tc_args = tc.get("args", {})
                    key = self._tracker.make_key(tc_name, tc_args)
                    if self._tracker.has(key):
                        args_str = json.dumps(tc_args, sort_keys=True, default=str)
                        reminders.append(
                            f"- `{tc_name}({args_str})` -- you already have this result"
                        )
                break  # Only check the last AI message

        if reminders:
            reminder_text = (
                "DEDUP WARNING: You have already called the following tools with "
                "identical arguments. Do NOT repeat these calls -- use your earlier results:\n"
                + "\n".join(reminders)
            )
            return [*messages, {"role": "human", "content": reminder_text}]

        return messages

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        if tool_name in _EXCLUDED_TOOLS:
            return await call_next(tool_name, tool_args)

        key = self._tracker.make_key(tool_name, tool_args)
        cached = self._tracker.lookup(key)
        if cached is not None:
            return cached

        result = await call_next(tool_name, tool_args)
        self._tracker.store(key, result)
        return result


# ---------------------------------------------------------------------------
# 6. SkillMiddleware
# ---------------------------------------------------------------------------


class SkillMiddleware:
    """Injects skill context from a MarkdownVault into the system prompt.

    on_llm_start: appends a <skills> section to the first system message,
    or prepends a new system message if none exists.
    """

    def __init__(self, vault: Any, skill_paths: list[str]) -> None:
        self._skill_descriptions: list[dict[str, str]] = []
        for path in skill_paths:
            note = vault.read(path)
            if note is None:
                logger.warning("Skill not found in vault, skipping: %s", path)
                continue
            fm = note["frontmatter"]
            self._skill_descriptions.append({
                "path": path,
                "name": fm.get("name", path),
                "description": fm.get("description", ""),
            })
        logger.info(
            "SkillMiddleware initialized: %d/%d skills loaded",
            len(self._skill_descriptions),
            len(skill_paths),
        )

    @property
    def skill_count(self) -> int:
        return len(self._skill_descriptions)

    def _build_skills_section(self) -> str:
        if not self._skill_descriptions:
            return ""

        skill_entries = "\n".join(
            f'  <skill path="{s["path"]}">'
            f'{s["name"]}: {s["description"]}'
            f"</skill>"
            for s in self._skill_descriptions
        )

        return (
            "<skills>\n"
            "You have access to domain-specific skills stored in the knowledge vault.\n"
            "\n"
            "<skill_list>\n"
            f"{skill_entries}\n"
            "</skill_list>\n"
            "\n"
            "<usage>\n"
            "1. Review the skill list above to find relevant skills for your task.\n"
            '2. Load a skill: `vault_read(path="skill/diagnose-sql")` to get full instructions.\n'
            "3. Follow references: loaded skills may contain `[[wikilinks]]` pointing to\n"
            "   sub-skills or related notes. Load them with `vault_read` to go deeper.\n"
            "4. Discover more: use `vault_search(query=..., filters={\"type\": \"skill\"})` to\n"
            "   find skills beyond this list, or `vault_list(path=\"skill\")` to browse.\n"
            "5. Notes are interconnected -- a skill can reference concepts, other skills, or\n"
            "   episodic memories. Follow the links to build the context you need.\n"
            "</usage>\n"
            "</skills>"
        )

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        if not self._skill_descriptions:
            return messages

        skills_section = self._build_skills_section()

        new_messages: list[Any] = []
        injected = False

        for msg in messages:
            is_system = (
                (isinstance(msg, dict) and msg.get("role") == "system")
                or getattr(msg, "type", None) == "system"
            )
            if not injected and is_system:
                if isinstance(msg, dict):
                    new_msg = {**msg, "content": str(msg.get("content", "")) + "\n\n" + skills_section}
                else:
                    # LangChain message object -- create a new dict-based message
                    new_msg = {
                        "role": "system",
                        "content": str(getattr(msg, "content", "")) + "\n\n" + skills_section,
                    }
                new_messages.append(new_msg)
                injected = True
            else:
                new_messages.append(msg)

        if not injected:
            new_messages.insert(0, {"role": "system", "content": skills_section})

        return new_messages

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        return await call_next(tool_name, tool_args)


# ---------------------------------------------------------------------------
# 7. DynamicContextMiddleware
# ---------------------------------------------------------------------------


class DynamicContextMiddleware:
    """Injects dynamic state context into the system prompt before each LLM call.

    Replaces the old orchestrator's _prepare_messages + _build_context_message.
    The base system prompt is static (cacheable); dynamic context (state snapshot,
    round info) is appended as an AIMessage prefill so the LLM sees it as its own
    prior reasoning and continues naturally.

    ``format_context_fn`` is always zero-arg. Scenarios bind their own state
    via closures in ``Scenario.setup()``.
    """

    def __init__(
        self,
        format_context_fn: Callable[[], str],
        base_system_prompt: str,
        max_rounds: int = 30,
    ) -> None:
        self._format_fn = format_context_fn
        self._base_prompt = base_system_prompt
        self._max_rounds = max_rounds

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        context_text = self._format_fn()

        round_num = ctx.step + 1

        # Build round context with urgency warnings
        round_block = f"Round: {round_num}/{self._max_rounds}"
        if round_num >= self._max_rounds:
            round_block += (
                "\n\nLAST ROUND -- MUST output "
                "<decision>finalize</decision> now. "
                "Do NOT dispatch any more workers."
            )
        elif round_num >= self._max_rounds - 1:
            round_block += (
                f"\n\nRound {round_num}/{self._max_rounds} -- 1 round remaining. "
                "Consider finalizing if evidence is sufficient."
            )

        # Replace system message with static prompt, keep conversation history
        new_messages: list[Any] = [{"role": "system", "content": self._base_prompt}]
        for m in messages:
            is_system = (
                (isinstance(m, dict) and m.get("role") == "system")
                or getattr(m, "type", None) == "system"
            )
            if not is_system:
                new_messages.append(m)

        # Append state context as AIMessage prefill (same as old orchestrator)
        if context_text:
            prefill = (
                f"<current_state>\n{context_text}\n</current_state>"
                f"\n\n<round_context>\n{round_block}\n</round_context>"
                "\n\nBased on the current state, my next action:"
            )
            new_messages.append({"role": "assistant", "content": prefill})

        return new_messages

    async def on_llm_end(self, response: Any, ctx: LoopContext) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        return await call_next(tool_name, tool_args)
