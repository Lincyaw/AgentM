"""Thin AgentM CLI with registry-backed provider resolution and OAuth login."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import webbrowser

from agentm.ai.oauth import get_oauth_provider, get_oauth_providers
from agentm.extensions.loader import (
    ScenarioLoadError,
    load_scenario_definition,
)
from agentm.harness import AgentSession, AgentSessionConfig
from agentm.harness.auth_storage import AuthStorage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentm", description="AgentM CLI")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run one prompt against a scenario.")
    run.add_argument("prompt", help="User prompt to send to the agent.")
    run.add_argument(
        "--scenario",
        default="general_purpose",
        help="Scenario recipe name under agentm.extensions.scenarios.",
    )
    run.add_argument(
        "--model",
        default=None,
        help="Override the scenario's default model id.",
    )
    run.add_argument(
        "--provider",
        default=None,
        help="Override the scenario's provider registry key.",
    )
    run.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory exposed to extensions (default: cwd).",
    )

    auth = subparsers.add_parser("auth", help="Manage stored OAuth/API credentials.")
    auth_subparsers = auth.add_subparsers(dest="auth_command")

    login = auth_subparsers.add_parser("login", help="Run an OAuth login flow.")
    login.add_argument("provider", help="OAuth provider id, e.g. anthropic")

    status = auth_subparsers.add_parser("status", help="Show provider auth state.")
    status.add_argument("provider", nargs="?", default=None)

    auth_subparsers.add_parser("providers", help="List OAuth-capable providers.")

    parser.add_argument("prompt", nargs="?", help=argparse.SUPPRESS)
    return parser


async def _run(prompt: str, scenario_name: str, model: str | None, provider: str | None, cwd: str) -> int:
    scenario = load_scenario_definition(scenario_name)
    config = AgentSessionConfig(
        cwd=cwd,
        extensions=scenario.extensions,
        provider=provider or scenario.provider,
        model=model or scenario.model,
        provider_config=scenario.provider_config,
    )
    session = await AgentSession.create(config)
    try:
        await session.prompt(prompt)
    finally:
        await session.shutdown()
    return 0


async def _auth_login(provider_id: str) -> int:
    provider = get_oauth_provider(provider_id)
    if provider is None:
        print(f"agentm: unknown OAuth provider: {provider_id}", file=sys.stderr)
        return 2

    storage = AuthStorage.create()

    class Callbacks:
        def on_auth(self, info: object) -> None:
            from agentm.ai.types import OAuthAuthInfo

            assert isinstance(info, OAuthAuthInfo)
            print(info.url)
            if info.instructions:
                print(info.instructions)
            try:
                webbrowser.open(info.url)
            except Exception:
                pass

        async def on_prompt(self, prompt: object) -> str:
            from agentm.ai.types import OAuthPrompt

            assert isinstance(prompt, OAuthPrompt)
            return input(f"{prompt.message}\n> ")

        def on_progress(self, message: str) -> None:
            print(message)

        async def on_manual_code_input(self) -> str:
            return ""

    credentials = await provider.login(Callbacks())
    storage.store_oauth(provider_id, credentials)
    print(f"Stored OAuth credentials for {provider_id} at {storage.path()}")
    return 0


def _auth_status(provider_id: str | None) -> int:
    storage = AuthStorage.create()
    if provider_id is not None:
        status = storage.get_status(provider_id)
        label = status.label or "not configured"
        print(f"{provider_id}: configured={status.configured} source={status.source} label={label}")
        return 0
    for provider in get_oauth_providers():
        status = storage.get_status(provider.id)
        label = status.label or "not configured"
        print(f"{provider.id}: configured={status.configured} source={status.source} label={label}")
    return 0


def _auth_providers() -> int:
    for provider in get_oauth_providers():
        print(f"{provider.id}\t{provider.name}")
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "auth":
            if args.auth_command == "login":
                rc = asyncio.run(_auth_login(args.provider))
            elif args.auth_command == "status":
                rc = _auth_status(args.provider)
            elif args.auth_command == "providers":
                rc = _auth_providers()
            else:
                _build_parser().error("missing auth subcommand")
                return
        else:
            prompt = args.prompt
            if args.command == "run":
                prompt = args.prompt
            if not prompt:
                _build_parser().error("missing prompt")
                return
            rc = asyncio.run(
                _run(
                    prompt,
                    getattr(args, "scenario", "general_purpose"),
                    getattr(args, "model", None),
                    getattr(args, "provider", None),
                    getattr(args, "cwd", os.getcwd()),
                )
            )
    except ScenarioLoadError as exc:
        print(f"agentm: scenario load failed: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)
