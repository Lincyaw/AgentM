"""How every loop agent is invoked, in one place.

An agent here is: a manifest holding its prompt, an explicit extension list, and
exactly one tool that ends the session by carrying the answer. The tool schema
is the output contract -- there is no other way for the model to finish, so a
missing field is a retry rather than a parse failure downstream.

The extension list is written out rather than borrowed from a named scenario.
These agents are part of this package's contract; if a scenario elsewhere gains
memory or compaction, a diagnosis should not silently change shape.

Two shapes of agent, chosen by whether ``operations`` is supplied:

* **host** -- tools run on this machine, over the batch's artifacts on disk.
* **sandbox** -- tools run inside the environment the failing attempt used, so
  the agent can look at the repository as it stood at the decision point rather
  than reconstructing it from a patch. Same extensions minus the local backend,
  because execution comes from the bound environment instead.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from agentm import AgentSession
from agentm.config import DefaultSessionSpecResolver
from agentm.core.abi import (
    RESOURCE_WRITER,
    AgentSessionConfig,
    EnvironmentOperations,
    ExtensionInput,
    FunctionTool,
    JsonValue,
    ResourceWriter,
    ServiceRegistry,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.roles import bind_environment_operations
from policy_engine.shared.manifest import AgentManifest, load_manifest

from .contracts import Record

#: Enough to read, search and run, and nothing that rewrites history behind the
#: agent's back. Mirrors the harbor scenario's base set, which is the one known
#: to work with tools routed into a sandbox.
BASE_EXTENSIONS: tuple[str, ...] = (
    "agentm.extensions.builtin.observability",
    "agentm.extensions.builtin.system_prompt",
    "agentm.extensions.builtin.file_tools",
    "agentm.extensions.builtin.tool_bash",
    "agentm.extensions.builtin.tool_result_cap",
    "agentm.extensions.builtin.tool_error_messages",
)

#: Only when nothing else provides execution. With an environment bound, the
#: local backend would silently give the agent a second, wrong filesystem.
HOST_BACKEND = "agentm.extensions.builtin.local_backend"


#: Where this package's stage manifests live. Passed to the shared loader rather
#: than reached by a second one: the manifest shape is a package-wide contract,
#: and two readers of it drift -- the first pair already had, one honouring
#: ``tools`` and the other hard-coding it.
_AGENTS_DIR = str(Path(__file__).parent / "agents")


def load_stage_manifest(name: str) -> AgentManifest:
    """One loop stage's prompt and shape, from ``loop/agents/<name>.yaml``."""
    return load_manifest(name, _AGENTS_DIR)


@dataclass(frozen=True, slots=True)
class ResultTool:
    """The one tool an agent finishes with. ``payload`` is the stage's output
    contract: a pydantic model whose schema the tool presents and whose
    validation the call must pass, so a missing field is a retry rather than an
    empty string downstream."""

    name: str
    description: str
    payload: type[Record]


@dataclass(slots=True)
class _Sink:
    payload: Mapping[str, JsonValue] | None = None


def _terminal_tool(spec: ResultTool, sink: _Sink) -> FunctionTool:
    async def submit(args: dict[str, JsonValue]) -> ToolResult | ToolTerminate:
        try:
            validated = spec.payload.model_validate(args)
        except ValidationError as exc:
            # The error text goes back to the model, which gets another go —
            # the session only ends when this tool accepts.
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Not recorded — fix these fields and call "
                        f"{spec.name} again: {problems}",
                    )
                ],
                is_error=True,
            )
        sink.payload = validated.to_json()
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="Recorded.")]),
            reason=f"policy-loop:{spec.name}",
        )

    return FunctionTool(
        name=spec.name,
        description=spec.description,
        parameters=spec.payload,
        fn=submit,
    )


@dataclass(slots=True)
class AgentRun:
    """A configured agent, reusable across cases."""

    manifest: AgentManifest
    result_tool: ResultTool
    #: Model profile from the user's config.toml. Empty takes whatever the
    #: resolver's default is, which is rarely what an expensive analysis wants.
    provider: str = ""
    cwd: str = "."
    user_config: str = ""
    #: Bind to run the agent inside the failing attempt's environment.
    operations: EnvironmentOperations | None = None
    #: The writer for that same environment. Required alongside ``operations``:
    #: without it the file tools and the output cap have no port to bind to.
    writer: ResourceWriter | None = None
    extensions: tuple[str, ...] = field(default=BASE_EXTENSIONS)

    async def run(self, prompt: str) -> Mapping[str, JsonValue] | None:
        """The agent's answer, or ``None`` when it produced none.

        Returning ``None`` rather than raising: one case failing to diagnose
        should cost that case, not the batch.
        """
        sink = _Sink()
        modules = list(self.extensions)
        if self.operations is None:
            modules.append(HOST_BACKEND)
        specs: list[ExtensionInput] = [(module, {}) for module in modules]

        env = dict(os.environ)
        if self.provider:
            env["AGENTM_PROVIDER"] = self.provider

        services = ServiceRegistry()
        if self.operations is not None:
            bind_environment_operations(services, self.operations)
            # The file tools and the output cap are written against the writer
            # port, not against bash, so an environment bound without one leaves
            # them unsatisfied and the session refuses to start. Whoever supplies
            # the environment supplies the writer for the same filesystem; a host
            # writer here would let the agent read this machine while believing
            # it was reading the repository.
            if self.writer is not None:
                services.bind(RESOURCE_WRITER, self.writer)

        allowlist = [self.result_tool.name, *self.manifest.tools]
        config = AgentSessionConfig(
            cwd=self.cwd,
            extensions=specs,
            extra_tools=[_terminal_tool(self.result_tool, sink)],
            tool_allowlist=allowlist,
            system=self.manifest.system,
            purpose=f"policy-loop:{self.manifest.name}",
            spec_resolver=DefaultSessionSpecResolver(
                user_config=self.user_config or None, env=env
            ),
            loop_config=None,
        )

        session = None
        try:
            session = await AgentSession.create(config, host_services=services)
            session.start()
            receipt = await session.prompt(prompt)
            await receipt.wait()
        except Exception as exc:  # noqa: BLE001 - one case must not kill the batch
            logger.warning("{}: run failed: {}", self.manifest.name, exc)
            return None
        finally:
            if session is not None:
                try:
                    await session.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("{}: shutdown failed: {}", self.manifest.name, exc)

        if sink.payload is None:
            logger.warning(
                "{}: finished without calling {}",
                self.manifest.name,
                self.result_tool.name,
            )
        return sink.payload


async def fan_out[T, R](
    items: Sequence[T],
    work: Callable[[T], Awaitable[R | None]],
    *,
    concurrency: int,
    label: str,
    noun: str,
    on_result: Callable[[list[R]], None] | None = None,
) -> list[R]:
    """Run ``work`` over ``items`` at a bounded width, dropping the failures.

    Every stage wants exactly this and each held its own copy: a semaphore, a
    gather, a ``None`` filter and a count. Concurrency is bounded because each
    item holds a session, a model connection and sometimes a sandbox, so the
    limit is a real resource ceiling rather than a tuning knob -- which is why
    it is one decision and not three.

    ``None`` from ``work`` means that item produced nothing, and it costs that
    item only: one case failing to diagnose must not cost the batch.

    ``on_result`` is handed everything finished so far, each time one finishes.
    Without it a stage holds its whole batch in memory until the last item
    returns, and a stop anywhere loses all of it -- twenty-one diagnoses at
    roughly an hour of model time, in the case that prompted this. ``replay``
    already wrote after every arm for exactly this reason; putting it here
    rather than in one stage means diagnose, abstract, notes, compile and align
    stop being the exception.
    """
    limit = asyncio.Semaphore(max(1, concurrency))
    done: list[R] = []

    async def one(item: T) -> R | None:
        async with limit:
            result = await work(item)
        if result is not None:
            done.append(result)
            if on_result is not None:
                # A checkpoint that can end the batch defeats its own purpose,
                # so a failing sink costs the checkpoint and nothing else.
                try:
                    on_result(list(done))
                except Exception as exc:  # noqa: BLE001 - the work still stands
                    logger.warning("{}: could not checkpoint: {}", label, exc)
        return result

    await asyncio.gather(*(one(item) for item in items))
    logger.info("{}: {} {} from {} case(s)", label, len(done), noun, len(items))
    return done


__all__ = [
    "BASE_EXTENSIONS",
    "HOST_BACKEND",
    "AgentManifest",
    "AgentRun",
    "ResultTool",
    "fan_out",
    "load_stage_manifest",
]
