"""Let the agent write a tool and install it into its own running session.

The model supplies what the tool does; this atom supplies the contract around
it. That split matters: the atom module shape, the manifest, the registration
call and the result wrapping are mechanism the model has no reason to know, and
asking it to reproduce them from memory turns every authoring attempt into a
guess at an API. The model writes a function body and a parameter schema.

Policy lives here, mechanism does not. Source is written through the resource
writer and installed through ``install_extension``; failures come back as tool
results so the model can read the error and fix its own code.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
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
    JsonValue,
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
    description="Write a new tool and install it into this running session.",
    registers=("tool:write_tool",),
    config_schema=ToolAuthoringConfig,
    requires=(RESOURCE_WRITER.capability,),
)


class WriteToolArgs(BaseModel):
    """Arguments for authoring one tool."""

    name: str = Field(
        description=(
            "Name the tool will be called by. Must be a valid Python identifier."
        )
    )
    description: str = Field(
        description="What the tool does, as the model calling it will read it."
    )
    parameters: dict[str, JsonValue] = Field(
        description=(
            "JSON Schema object describing the arguments, with 'type': "
            "'object' and a 'properties' map."
        )
    )
    body: str = Field(
        description=(
            "Python body of the tool. It receives the arguments as a dict "
            "named args and must return a value; the return is converted to "
            "text. Write only the body, no def line. The standard library is "
            "importable inside the body."
        )
    )


_TEMPLATE = '''\
"""Tool authored at runtime by the agent."""

import json

from agentm.core.abi import (
    ExtensionManifest,
    FunctionTool,
    TextContent,
    ToolResult,
)

MANIFEST = ExtensionManifest(
    name={name!r},
    description={description!r},
    registers=({registers!r},),
)

_PARAMETERS = json.loads({parameters!r})


def _impl(args):
{body}


async def _run(args):
    try:
        value = _impl(args)
    except Exception as exc:
        return ToolResult(
            content=[TextContent(type="text", text=f"{{type(exc).__name__}}: {{exc}}")],
            is_error=True,
        )
    return ToolResult(content=[TextContent(type="text", text=str(value))])


def install(api, config):
    api.register_tool(
        FunctionTool(
            name={name!r},
            description={description!r},
            parameters=_PARAMETERS,
            fn=_run,
        )
    )
'''


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=True,
    )


def _render_atom(parsed: WriteToolArgs) -> str:
    body = textwrap.indent(textwrap.dedent(parsed.body).strip("\n"), "    ")
    if not body.strip():
        body = "    return None"
    source = _TEMPLATE.format(
        name=parsed.name,
        description=parsed.description,
        registers=f"tool:{parsed.name}",
        parameters=json.dumps(parsed.parameters),
        body=body,
    )
    return source


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

        source = _render_atom(parsed)
        try:
            compile(source, parsed.name, "exec")
        except SyntaxError as exc:
            return _error(
                f"the body does not compile: line {exc.lineno}: {exc.msg}. "
                "Nothing was written or installed. Send a corrected body."
            )

        source_bytes = source.encode()
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
                "The source is saved but inactive. Fix it and call write_tool "
                "again."
            )

        logger.info("authored tool installed: {} from {}", parsed.name, path)
        return _ok(
            f"Installed. {parsed.name!r} is callable from your next turn "
            f"onward, not this one. Source saved at {path}."
        )


def install(api: AtomAPI, config: ToolAuthoringConfig) -> None:
    """Register the authoring tool."""

    author = _ToolAuthor(api, config)
    api.register_tool(
        FunctionTool(
            name="write_tool",
            description=(
                "Create a new tool and install it into this session. You supply "
                "the tool's name, description, JSON Schema parameters, and a "
                "Python body that receives the arguments as a dict named args "
                "and returns a value. The tool becomes callable on your next "
                "turn, not the current one."
            ),
            parameters=pydantic_to_tool_schema(WriteToolArgs),
            fn=author.write_tool,
        )
    )
