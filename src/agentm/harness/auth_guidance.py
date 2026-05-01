from __future__ import annotations


def get_provider_login_help() -> str:
    return (
        "Authenticate with `agentm auth login <provider>` or set the provider's "
        "API-key environment variable."
    )


def format_no_models_available_message() -> str:
    return f"No models available. {get_provider_login_help()}"


def format_no_model_selected_message() -> str:
    return (
        "No model selected. "
        f"{get_provider_login_help()} Then choose a model with --model or the scenario default."
    )


def format_no_api_key_found_message(provider: str) -> str:
    target = provider if provider else "the selected provider"
    return f"No API key found for {target}. {get_provider_login_help()}"
