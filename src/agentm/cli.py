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
from typing import Annotated, Any, cast

import typer
from dotenv import load_dotenv

from agentm.ai import DEFAULT_PROVIDER_REGISTRY, ProviderRegistry
from agentm.core.abi.events import DiagnosticEvent, EventBus
from agentm.core.abi.session_store import SessionState, SessionStore
from agentm.core.lib.render import final_summary
from agentm.harness.events import ExtensionInstallEvent


# Walk up from cwd to find the nearest .env. Existing env vars win
# (override=False), so explicit ``KEY=... agentm`` still beats file values.
_cur = Path.cwd().resolve()
for _candidate in (_cur, *_cur.parents):
    _env_path = _candidate / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path, override=False)
        break


def _print_final(
    final_messages: list[Any],
    *,
    cost_service: Any | None = None,
    provider: str | None = None,
) -> None:
    report = final_summary(final_messages)
    typer.echo("\n" + "=" * 60)
    typer.echo("AGENT FINAL OUTPUT")
    typer.echo("=" * 60)
    typer.echo(report.text if report.text else "<no text output>")
    typer.echo("=" * 60)
    typer.echo(f"messages={report.message_count} tool_calls={report.tool_calls}")
    usage = report.usage
    if usage.assistant_turns:
        estimate = getattr(cost_service, "estimate", None)
        cost = estimate(usage, provider=provider) if callable(estimate) else None
        suffix = f" cost={cost.currency} {cost.amount:.6f}" if cost is not None else ""
        turns = "turn" if usage.assistant_turns == 1 else "turns"
        typer.echo(
            f"tokens: in={usage.input_tokens} out={usage.output_tokens} "
            f"cache_r={usage.cache_read} cache_w={usage.cache_write} "
            f"(over {usage.assistant_turns} {turns}){suffix}"
        )

def _parse_tools(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if value == "":
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


def _parse_extensions(values: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    """Parse repeated ``--extension MODULE[:JSON]`` flags.

    Each entry is either a bare dotted module path (config = {}) or
    ``module.path:{"key":value}`` where the JSON object is parsed verbatim.
    The first colon splits module from config so module paths themselves
    cannot contain a colon (they are dotted Python imports — colons would
    already be invalid).
    """

    if not values:
        return []
    import json

    out: list[tuple[str, dict[str, Any]]] = []
    for raw in values:
        spec = raw.strip()
        if not spec:
            continue
        if ":" in spec:
            module, _, cfg_raw = spec.partition(":")
            module = module.strip()
            try:
                cfg = json.loads(cfg_raw)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(
                    f"--extension {raw!r}: config after ':' must be valid JSON ({exc})"
                ) from exc
            if not isinstance(cfg, dict):
                raise typer.BadParameter(
                    f"--extension {raw!r}: config must be a JSON object"
                )
        else:
            module = spec
            cfg = {}
        if not module:
            raise typer.BadParameter(f"--extension {raw!r}: module path is empty")
        out.append((module, cfg))
    return out


def _provider_default() -> str:
    return os.environ.get(
        "AGENTM_PROVIDER", DEFAULT_PROVIDER_REGISTRY.default_provider().id
    )


def _model_default() -> str:
    provider = _provider_default()
    return os.environ.get(
        "AGENTM_MODEL", DEFAULT_PROVIDER_REGISTRY.default_model(provider)
    )


def _make_default_session_store(cwd: str) -> SessionStore:
    from agentm.harness.session_manager import JsonlSessionStore

    return JsonlSessionStore(cwd=Path(cwd))


def _resolve_session_state(
    *,
    cwd: str,
    resume: str | None,
    continue_recent: bool,
    session_store: SessionStore,
) -> SessionState:
    if resume:
        try:
            return session_store.open(resume)
        except FileNotFoundError as exc:
            raise typer.BadParameter(
                f"--resume {resume!r}: no session found for cwd {cwd!r}"
            ) from exc
    if continue_recent:
        state = session_store.most_recent(Path(cwd))
        if state is not None:
            return state
    return session_store.create(Path(cwd))


def _make_install_warner() -> Any:
    def _on_install(event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        typer.echo(
            f"WARNING: [extension_install] {event.module_path}: {event.error}",
            err=True,
        )

    return _on_install


def _attach_default_diagnostics(bus: EventBus) -> dict[str, bool]:
    state = {"error_seen": False}

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        prefix = {
            "info": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
        }.get(event.level, event.level.upper())
        typer.echo(f"{prefix}: [{event.source}] {event.message}", err=True)
        if event.level == "error":
            state["error_seen"] = True

    bus.on(DiagnosticEvent.CHANNEL, _on_diagnostic)
    bus.on(ExtensionInstallEvent.CHANNEL, _make_install_warner())
    return state


def _build_session_config(
    *,
    scenario: str | None,
    extra_extensions: list[tuple[str, dict[str, Any]]],
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
    bus: EventBus,
    resume: str | None = None,
    continue_recent: bool = False,
    session_store: SessionStore | None = None,
    provider_registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
) -> tuple[Any, SessionState]:
    from agentm.harness import AgentSessionConfig

    store = session_store or _make_default_session_store(cwd)
    session_state = _resolve_session_state(
        cwd=cwd,
        resume=resume,
        continue_recent=continue_recent,
        session_store=store,
    )
    try:
        provider_spec = provider_registry.build(provider, {"model": model})
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return (
        AgentSessionConfig(
            cwd=cwd,
            provider=provider_spec,
            scenario=scenario,
            extra_extensions=extra_extensions,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=tool_allowlist,
            session_manager=cast(Any, session_state),
            bus=bus,
        ),
        session_state,
    )


async def _run(
    *,
    prompt: str,
    scenario: str | None,
    extra_extensions: list[tuple[str, dict[str, Any]]],
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
    quiet: bool,
    resume: str | None,
    continue_recent: bool,
) -> int:
    from agentm.harness import AgentSession

    bus = EventBus()
    diagnostic_state = _attach_default_diagnostics(bus)

    config, session_manager = _build_session_config(
        scenario=scenario,
        extra_extensions=extra_extensions,
        no_extensions=no_extensions,
        no_skills=no_skills,
        no_prompt_templates=no_prompt_templates,
        tool_allowlist=tool_allowlist,
        provider=provider,
        model=model,
        cwd=cwd,
        bus=bus,
        resume=resume,
        continue_recent=continue_recent,
    )
    if not quiet and session_manager.session_file is not None:
        typer.echo(f"INFO: session log: {session_manager.session_file}", err=True)
        typer.echo(f"INFO: session id: {session_manager.get_session_id()}", err=True)

    session = await AgentSession.create(config)
    try:
        final = await session.prompt(prompt)
        cost_service = session.get_service("cost_query")
        provider_name = session.model.provider if session.model is not None else None
        if not quiet:
            _print_final(final, cost_service=cost_service, provider=provider_name)
    finally:
        await session.shutdown()

    if not quiet:
        sid = session_manager.get_session_id()
        if sid:
            typer.echo(
                f'session_id={sid}  (resume with: agentm --resume {sid} "<prompt>")'
            )
    return 1 if diagnostic_state["error_seen"] else 0


async def _run_interactive(
    *,
    scenario: str | None,
    extra_extensions: list[tuple[str, dict[str, Any]]],
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
) -> int:
    """Build a session config and hand off to the Textual TUI runner."""

    from agentm.modes.textual_app import run as run_textual_tui

    bus = EventBus()
    _attach_default_diagnostics(bus)
    config, session_manager = _build_session_config(
        scenario=scenario,
        extra_extensions=extra_extensions,
        no_extensions=no_extensions,
        no_skills=no_skills,
        no_prompt_templates=no_prompt_templates,
        tool_allowlist=tool_allowlist,
        provider=provider,
        model=model,
        cwd=cwd,
        bus=bus,
    )
    if session_manager.session_file is not None:
        typer.echo(f"INFO: session log: {session_manager.session_file}", err=True)

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
                "<cwd>/contrib/scenarios/<name>/manifest.yaml. An absolute "
                "path is also accepted. When unset, auto-discovers every "
                "builtin atom."
            ),
        ),
    ] = None,
    extension: Annotated[
        list[str] | None,
        typer.Option(
            "--extension",
            help=(
                "Mount an extra atom on top of --scenario / auto-discovery. "
                "Repeatable. Form: 'dotted.module.path' or "
                '\'dotted.module.path:{"key":"value"}\' for inline JSON '
                "config. Example: --extension llmharness.adapters.agentm "
                "--extension some.atom:'{\"k\":3}'."
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
                "LLM provider to register. Defaults to the provider registry. Builtins include 'anthropic' (respects "
                "ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY) or 'openai' (respects "
                "OPENAI_BASE_URL/OPENAI_API_KEY plus WARPGATE_TICKET and "
                "OPENAI_VERIFY_SSL for self-signed gateways like Warpgate)."
            ),
        ),
    ] = _provider_default(),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Model id passed to the active provider. Defaults to the "
                "selected provider registry descriptor; override for alternates (e.g. "
                "'gpt-4o', 'Kimi-K2', 'deepseek-chat')."
            ),
        ),
    ] = _model_default(),
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
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help=(
                "Resume an existing session. Accepts a session id (hex, as "
                "printed at the end of a previous run) or an explicit path "
                "to a .jsonl session log. The new --prompt is appended to "
                "that session's history, enabling multi-turn from the CLI."
            ),
        ),
    ] = None,
    continue_recent: Annotated[
        bool,
        typer.Option(
            "--continue",
            help=(
                "Resume the most recent session for the current --cwd. "
                "Convenient shortcut for --resume <id> when you just ran "
                "agentm in this directory."
            ),
        ),
    ] = False,
) -> None:
    """Send a single prompt and print the agent's final text."""

    extra_extensions = _parse_extensions(extension)

    if interactive:
        rc = asyncio.run(
            _run_interactive(
                scenario=scenario,
                extra_extensions=extra_extensions,
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
            extra_extensions=extra_extensions,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=_parse_tools(tools),
            provider=provider,
            model=model,
            cwd=cwd,
            quiet=quiet,
            resume=resume,
            continue_recent=continue_recent,
        )
    )
    raise typer.Exit(code=rc)


def main() -> None:
    """Entry point referenced by the ``agentm`` console script."""

    typer.run(run_cmd)
