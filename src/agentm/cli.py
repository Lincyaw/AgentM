"""AgentM CLI (typer-based).

Single runtime: auto-discover every builtin atom by default. A curated
extension list is opted into via ``--scenario X``. Subsystems are turned
off via ``--no-*`` flags. Failures during construction emit diagnostics
through the EventBus rather than raising; only a missing provider is
fatal.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from agentm.core.abi.events import DiagnosticEvent, EventBus
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
)
from agentm.harness.events import ExtensionInstallEvent


def _load_dotenv() -> None:
    """Load ``KEY=value`` pairs from the nearest ``.env`` walking up from cwd.

    Existing env vars win — explicit ``KEY=... agentm`` invocations still
    override file values. Quoted values get unwrapped (``"x"`` → ``x``).
    Lines starting with ``#`` and blank lines are ignored. No interpolation,
    no exports, no shell semantics — keep it boring.
    """
    cur = Path.cwd().resolve()
    for candidate in (cur, *cur.parents):
        env_path = candidate / ".env"
        if env_path.is_file():
            break
    else:
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if (len(value) >= 2) and (
            (value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _print_final(final_messages: list) -> None:
    text_blocks: list[str] = []
    tool_calls = 0
    for msg in final_messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_blocks.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    tool_calls += 1

    typer.echo("\n" + "=" * 60)
    typer.echo("AGENT FINAL OUTPUT")
    typer.echo("=" * 60)
    typer.echo("\n\n".join(text_blocks) if text_blocks else "<no text output>")
    typer.echo("=" * 60)
    typer.echo(f"messages={len(final_messages)} tool_calls={tool_calls}")


def _parse_tools(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if value == "":
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


_FALSY = {"0", "false", "no", "off", "n", "f"}


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse a tristate env var as bool. Unset → ``default``. Truthy unless
    the value matches a small set of conventional falsy strings.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSY


def _build_provider(provider: str, model: str) -> tuple[str, dict[str, Any]]:
    """Map ``--provider`` + ``--model`` into ``(module_path, config_dict)``.

    Each branch reads its own env-var conventions so users can pick provider
    without re-typing endpoint config on every invocation:

    - ``anthropic`` (default): respects ``ANTHROPIC_BASE_URL``.
    - ``openai``: respects ``OPENAI_BASE_URL``, ``WARPGATE_TICKET`` (folded
      into ``default_query={"warpgate-ticket": ...}`` for self-signed
      Warpgate gateways), and ``OPENAI_VERIFY_SSL`` (default ``true``;
      set ``false``/``0``/``no``/``off`` to skip cert verification when
      hitting a self-signed proxy).
    """

    if provider == "anthropic":
        cfg: dict[str, Any] = {"model": model}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            cfg["base_url"] = base_url
        return ("agentm.llm.anthropic", cfg)

    if provider == "openai":
        cfg = {"model": model}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            cfg["base_url"] = base_url
        ticket = os.environ.get("WARPGATE_TICKET")
        if ticket:
            cfg["default_query"] = {"warpgate-ticket": ticket}
        if not _env_bool("OPENAI_VERIFY_SSL", default=True):
            cfg["verify_ssl"] = False
        return ("agentm.llm.openai", cfg)

    raise typer.BadParameter(
        f"unknown --provider {provider!r}; expected 'anthropic' or 'openai'"
    )


async def _run(
    *,
    prompt: str,
    scenario: str | None,
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
    quiet: bool,
) -> int:
    from agentm.harness import AgentSession, AgentSessionConfig

    error_seen = False

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        nonlocal error_seen
        prefix = {
            "info": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
        }.get(event.level, event.level.upper())
        typer.echo(f"{prefix}: [{event.source}] {event.message}", err=True)
        if event.level == "error":
            error_seen = True

    def _on_extension_install(event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        typer.echo(
            f"WARNING: [extension_install] {event.module_path}: {event.error}",
            err=True,
        )

    bus = EventBus()
    bus.on(DiagnosticEvent.CHANNEL, _on_diagnostic)
    bus.on(ExtensionInstallEvent.CHANNEL, _on_extension_install)

    config = AgentSessionConfig(
        cwd=cwd,
        provider=_build_provider(provider, model),
        scenario=scenario,
        no_extensions=no_extensions,
        no_skills=no_skills,
        no_prompt_templates=no_prompt_templates,
        tool_allowlist=tool_allowlist,
        bus=bus,
    )

    session = await AgentSession.create(config)
    try:
        final = await session.prompt(prompt)
    finally:
        await session.shutdown()

    if not quiet:
        _print_final(final)
    return 1 if error_seen else 0


async def _run_interactive(
    *,
    scenario: str | None,
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
) -> int:
    """Build a session config and hand off to the Textual TUI runner."""

    from agentm.harness import AgentSessionConfig
    from agentm.modes.textual_app import run as run_textual_tui

    bus = EventBus()

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        prefix = {
            "info": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
        }.get(event.level, event.level.upper())
        typer.echo(f"{prefix}: [{event.source}] {event.message}", err=True)

    def _on_extension_install(event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        typer.echo(
            f"WARNING: [extension_install] {event.module_path}: {event.error}",
            err=True,
        )

    bus.on(DiagnosticEvent.CHANNEL, _on_diagnostic)
    bus.on(ExtensionInstallEvent.CHANNEL, _on_extension_install)

    config = AgentSessionConfig(
        cwd=cwd,
        provider=_build_provider(provider, model),
        scenario=scenario,
        no_extensions=no_extensions,
        no_skills=no_skills,
        no_prompt_templates=no_prompt_templates,
        tool_allowlist=tool_allowlist,
        bus=bus,
    )
    return await run_textual_tui(config)


def run_cmd(
    prompt: Annotated[
        str,
        typer.Argument(
            help=(
                "User prompt to send to the agent. Pass empty string with "
                "--interactive for the multi-turn TUI."
            ),
        ),
    ] = "",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            help=(
                "Opt-in curated extension list. Bare name resolves under "
                "<cwd>/scenarios/<name>/manifest.yaml. An absolute path is "
                "also accepted. When unset, auto-discovers every builtin atom."
            ),
        ),
    ] = None,
    no_extensions: Annotated[
        bool,
        typer.Option(
            "--no-extensions",
            help="Skip atom discovery and loading; agent runs with no tools.",
        ),
    ] = False,
    no_skills: Annotated[
        bool,
        typer.Option(
            "--no-skills",
            help="Skip skill discovery in the resource loader.",
        ),
    ] = False,
    no_prompt_templates: Annotated[
        bool,
        typer.Option(
            "--no-prompt-templates",
            help="Skip prompt-template discovery in the resource loader.",
        ),
    ] = False,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            help="Comma-separated allowlist applied to atom-registered tools.",
        ),
    ] = None,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help=(
                "LLM provider to register: 'anthropic' (default; respects "
                "ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY) or 'openai' (respects "
                "OPENAI_BASE_URL/OPENAI_API_KEY plus WARPGATE_TICKET and "
                "OPENAI_VERIFY_SSL for self-signed gateways like Warpgate)."
            ),
        ),
    ] = os.environ.get("AGENTM_PROVIDER", "anthropic"),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Model id passed to the active provider. Default fits the "
                "anthropic provider; override when --provider openai (e.g. "
                "'gpt-4o', 'Kimi-K2', 'deepseek-chat')."
            ),
        ),
    ] = os.environ.get("AGENTM_MODEL", "claude-sonnet-4-6"),
    cwd: Annotated[
        str,
        typer.Option(
            "--cwd",
            help="Working directory exposed to extensions.",
        ),
    ] = os.getcwd(),
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress final-output printout."),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "-i",
            "--interactive",
            help="Open the Textual multi-turn TUI instead of running a single prompt.",
        ),
    ] = False,
) -> None:
    """Send a single prompt and print the agent's final text."""

    if interactive:
        rc = asyncio.run(
            _run_interactive(
                scenario=scenario,
                no_extensions=no_extensions,
                no_skills=no_skills,
                no_prompt_templates=no_prompt_templates,
                tool_allowlist=_parse_tools(tools),
                provider=provider,
                model=model,
                cwd=cwd,
            )
        )
        raise typer.Exit(code=rc)

    if not prompt:
        typer.echo("ERROR: prompt is required (or pass --interactive)", err=True)
        raise typer.Exit(code=2)

    rc = asyncio.run(
        _run(
            prompt=prompt,
            scenario=scenario,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=_parse_tools(tools),
            provider=provider,
            model=model,
            cwd=cwd,
            quiet=quiet,
        )
    )
    raise typer.Exit(code=rc)


def main() -> None:
    """Entry point referenced by the ``agentm`` console script."""

    typer.run(run_cmd)
