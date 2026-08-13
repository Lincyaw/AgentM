"""Let the agent write a tool and install it into its own running session.

The model authors a single-file atom, this atom writes it to disk, hands it to
the loader, and the session gains the tool from its next turn onward.

Policy lives here, mechanism does not. Everything this atom does is reachable
through ``AtomAPI``: it writes source through the resource writer, and installs
through ``install_extension``. Validation failures come back as tool results so
the model can read the error and fix its own code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import (
    RESOURCE_WRITER,
    AtomAPI,
    ExtensionManifest,
    ExtensionSource,
    ExtensionSpec,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.lib.tool_schema import pydantic_to_tool_schema


class ToolAuthoringConfig(BaseModel):
    """Where authored atoms are written."""

    directory: str = Field(
        default=".agentm/authored_tools",
        description="Directory, relative to cwd, holding authored atom source.",
    )


MANIFEST = ExtensionManifest(
    name="tool_authoring",
    description="Write a new tool as an atom and install it into this session.",
    registers=("tool:write_tool",),
    config_schema=ToolAuthoringConfig,
    requires=(RESOURCE_WRITER.capability,),
)


class WriteToolArgs(BaseModel):
    """Arguments for authoring one tool."""

    name: str = Field(
        description=(
            "Tool name the model will call. Also names the atom module, so it "
            "must be a valid Python identifier."
        )
    )
    source: str = Field(
        description=(
            "Complete source of a single-file atom. It must define MANIFEST = "
            "ExtensionManifest(...) and def install(api, config). Register the "
            "tool with api.register_tool(FunctionTool(...))."
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=True,
    )


class _ToolAuthor:
    """Writes authored atom source and installs it into the live session."""

    def __init__(self, api: AtomAPI, config: ToolAuthoringConfig) -> None:
        self._api = api
        self._directory = Path(api.ctx.cwd) / config.directory

    async def write_tool(self, args: dict[str, object]) -> ToolResult:
        parsed = WriteToolArgs.model_validate(args)
        if not parsed.name.isidentifier():
            return _error(
                f"tool name {parsed.name!r} is not a valid Python identifier, "
                "so it cannot name a module. Use letters, digits and "
                "underscores, not starting with a digit."
            )

        source_bytes = parsed.source.encode()
        digest = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
        path = self._directory / f"{parsed.name}_{digest[7:15]}.py"

        writer = self._api.services.get_role(RESOURCE_WRITER)
        if writer is None:
            return _error(
                "this session has no resource writer, so authored tool source "
                "cannot be saved. Nothing was installed."
            )
        write_result = await writer.write(
            str(path),
            source_bytes,
            rationale=f"author tool {parsed.name}",
        )
        if write_result.error is not None:
            return _error(
                f"could not save the tool source: {write_result.error}. "
                "Nothing was installed."
            )

        spec = ExtensionSpec(
            source=ExtensionSource(
                kind="file",
                location=str(path.resolve()),
                digest=digest,
            ),
            config={},
        )
        try:
            await self._api.install_extension(spec, trigger="authored")
        except Exception as exc:  # noqa: BLE001 - reported to the model verbatim
            logger.warning("authored tool {} failed to install: {}", parsed.name, exc)
            return _error(
                f"the tool did not install: {exc}\n\n"
                "The source is saved but inactive. Fix the code and call "
                "write_tool again."
            )

        logger.info("authored tool installed: {} from {}", parsed.name, path)
        return _ok(
            f"Installed. {parsed.name!r} is available from your next turn "
            f"onward; it is not callable in this one. Source saved at {path}."
        )


def install(api: AtomAPI, config: ToolAuthoringConfig) -> None:
    """Register the authoring tool."""

    author = _ToolAuthor(api, config)
    api.register_tool(
        FunctionTool(
            name="write_tool",
            description=(
                "Write a new tool and install it into this session. Provide the "
                "complete source of a single-file atom defining MANIFEST and "
                "install(api, config). The tool becomes callable on your next "
                "turn, not the current one."
            ),
            parameters=pydantic_to_tool_schema(WriteToolArgs),
            fn=author.write_tool,
        )
    )
