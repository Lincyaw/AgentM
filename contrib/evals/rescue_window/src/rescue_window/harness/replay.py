"""Action replay: re-execute side-effect tool calls in a fresh environment.

For scenarios where the data plane is mutable (e.g. terminal-bench: agent
modifies files in a Docker container), forking the conversation alone is not
enough — the environment state must be restored to match the prefix point.

Action replay extracts tool calls with side effects (bash, write_file,
edit_file) from a recorded trajectory up to the fork turn, then executes them
sequentially in the target environment to reconstruct the state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock

SIDE_EFFECT_TOOLS = frozenset({
    "bash",
    "tool_bash",
    "write_file",
    "edit_file",
})


@dataclass(frozen=True)
class ReplayAction:
    """One side-effect action to replay."""

    turn_index: int
    tool_name: str
    arguments: dict[str, Any]


class ReplayTarget(Protocol):
    """Execute a replayed tool call in the target environment."""

    async def replay_bash(self, command: str, *, cwd: str | None = None) -> str:
        ...

    async def replay_write_file(self, path: str, content: str) -> None:
        ...

    async def replay_edit_file(
        self, path: str, old_string: str, new_string: str
    ) -> None:
        ...


def extract_replay_actions(
    messages: list[AgentMessage],
    *,
    up_to_turn: int,
    side_effect_tools: frozenset[str] = SIDE_EFFECT_TOOLS,
) -> list[ReplayAction]:
    """Extract side-effect tool calls from messages up to (inclusive) a turn index.

    Turn index counts assistant messages (0-based, matching the sampler).
    """

    actions: list[ReplayAction] = []
    assistant_index = -1
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        assistant_index += 1
        if assistant_index > up_to_turn:
            break
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name in side_effect_tools:
                actions.append(
                    ReplayAction(
                        turn_index=assistant_index,
                        tool_name=block.name,
                        arguments=dict(block.arguments),
                    )
                )
    return actions


async def replay_actions(
    actions: list[ReplayAction],
    target: ReplayTarget,
) -> int:
    """Execute replay actions sequentially on the target. Returns count executed."""

    executed = 0
    for action in actions:
        if action.tool_name in ("bash", "tool_bash"):
            command = action.arguments.get("command", "")
            if command:
                await target.replay_bash(command)
                executed += 1
        elif action.tool_name == "write_file":
            path = action.arguments.get("file_path", "")
            content = action.arguments.get("content", "")
            if path:
                await target.replay_write_file(path, content)
                executed += 1
        elif action.tool_name == "edit_file":
            path = action.arguments.get("file_path", "")
            old = action.arguments.get("old_string", "")
            new = action.arguments.get("new_string", "")
            if path and old != new:
                await target.replay_edit_file(path, old, new)
                executed += 1
    return executed


@dataclass
class DockerReplayTarget:
    """Replay actions inside a Docker container via docker exec."""

    container_name: str

    async def replay_bash(self, command: str, *, cwd: str | None = None) -> str:
        import asyncio

        exec_cmd = ["docker", "exec"]
        if cwd:
            exec_cmd.extend(["-w", cwd])
        exec_cmd.extend([self.container_name, "bash", "-lc", command])
        proc = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode(errors="replace")

    async def replay_write_file(self, path: str, content: str) -> None:
        import asyncio
        import shlex

        escaped = shlex.quote(content)
        cmd = f"mkdir -p $(dirname {shlex.quote(path)}) && printf %s {escaped} > {shlex.quote(path)}"
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", self.container_name, "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await proc.communicate()

    async def replay_edit_file(
        self, path: str, old_string: str, new_string: str
    ) -> None:
        import asyncio

        read_proc = await asyncio.create_subprocess_exec(
            "docker", "exec", self.container_name, "cat", path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await read_proc.communicate()
        content = stdout.decode(errors="replace")
        updated = content.replace(old_string, new_string, 1)
        if updated != content:
            await self.replay_write_file(path, updated)
