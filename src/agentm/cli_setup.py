"""Root ``agentm setup`` and ``agentm onboard`` command registration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def register_setup_commands(app: typer.Typer) -> None:
    @app.command(name="setup")
    def setup_cmd(
        profile: Annotated[
            str | None,
            typer.Option(
                "--profile",
                help="config.toml profile name to create or update.",
            ),
        ] = None,
        provider: Annotated[
            str | None,
            typer.Option(
                "--provider",
                help="Provider id, e.g. openai or anthropic. Defaults from env/config.",
            ),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option(
                "--model",
                help="Model id or existing config.toml profile to use.",
            ),
        ] = None,
        api_key: Annotated[
            str | None,
            typer.Option(
                "--api-key",
                help="Provider API key to store in config.toml (chmod 0600).",
            ),
        ] = None,
        base_url: Annotated[
            str | None,
            typer.Option(
                "--base-url",
                help="Optional OpenAI/Anthropic-compatible base URL.",
            ),
        ] = None,
        context_window: Annotated[
            int | None,
            typer.Option(
                "--context-window",
                help="Optional context window recorded on the model profile.",
            ),
        ] = None,
        reasoning_effort: Annotated[
            str | None,
            typer.Option(
                "--reasoning-effort",
                help="Optional reasoning-effort hint recorded on the model profile.",
            ),
        ] = None,
        workspace: Annotated[
            Path | None,
            typer.Option(
                "--workspace",
                "--cwd",
                help="Workspace to initialize. Defaults to the current directory.",
            ),
        ] = None,
        bot_name: Annotated[
            str,
            typer.Option("--name", help="Persona name for seeded IDENTITY.md."),
        ] = "Assistant",
        voice: Annotated[
            str,
            typer.Option("--voice", help="Optional extra voice/tone line for SOUL.md."),
        ] = "",
        quick: Annotated[
            bool,
            typer.Option(
                "--quick",
                help=(
                    "Non-interactive setup. Fails if required model credentials "
                    "are missing."
                ),
            ),
        ] = False,
        check: Annotated[
            bool,
            typer.Option(
                "--check",
                help="Show setup status without writing files.",
            ),
        ] = False,
        test_model: Annotated[
            bool,
            typer.Option(
                "--test",
                help="Run one real model request to verify credentials and model access.",
            ),
        ] = False,
        test_prompt: Annotated[
            str,
            typer.Option(
                "--test-prompt",
                help="Prompt used by --test.",
            ),
        ] = "Reply with exactly: agentm-ok",
        no_contrib: Annotated[
            bool,
            typer.Option(
                "--no-contrib",
                help="Skip installing demo scenarios into AGENTM_HOME.",
            ),
        ] = False,
        no_skills: Annotated[
            bool,
            typer.Option(
                "--no-skills",
                help="Skip installing bundled SKILL.md resources into AGENTM_HOME.",
            ),
        ] = False,
        no_persona: Annotated[
            bool,
            typer.Option(
                "--no-persona",
                help="Skip seeding SOUL.md / IDENTITY.md / USER.md in the workspace.",
            ),
        ] = False,
        force_model: Annotated[
            bool,
            typer.Option(
                "--force-model",
                help=(
                    "Overwrite/create the model profile even when a default "
                    "already exists."
                ),
            ),
        ] = False,
    ) -> None:
        """Set up AgentM for first use with the fewest required choices."""
        from agentm.onboard import SetupConfig, run_setup

        raise typer.Exit(
            code=run_setup(
                SetupConfig(
                    profile=profile,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    context_window=context_window,
                    reasoning_effort=reasoning_effort,
                    workspace=workspace,
                    bot_name=bot_name,
                    voice=voice,
                    quick=quick,
                    check=check,
                    test_model=test_model,
                    test_prompt=test_prompt,
                    install_demo_scenarios=not no_contrib,
                    install_skills=not no_skills,
                    seed_persona_files=not no_persona,
                    force_model=force_model,
                )
            )
        )

    @app.command(name="onboard")
    def onboard_cmd() -> None:
        """Interactively bootstrap a fresh install."""
        from agentm.onboard import run_onboard

        run_onboard()
