from __future__ import annotations

from agentm.ai.api_registry import clear_api_providers, register_api_provider
from agentm.ai.providers.amazon_bedrock import PROVIDER as AMAZON_BEDROCK_PROVIDER
from agentm.ai.providers.anthropic import PROVIDER as ANTHROPIC_PROVIDER
from agentm.ai.providers.azure_openai_responses import (
    PROVIDER as AZURE_OPENAI_RESPONSES_PROVIDER,
)
from agentm.ai.providers.cerebras import PROVIDER as CEREBRAS_PROVIDER
from agentm.ai.providers.cloudflare_workers_ai import (
    PROVIDER as CLOUDFLARE_WORKERS_AI_PROVIDER,
)
from agentm.ai.providers.deepseek import PROVIDER as DEEPSEEK_PROVIDER
from agentm.ai.providers.fireworks import PROVIDER as FIREWORKS_PROVIDER
from agentm.ai.providers.github_copilot import PROVIDER as GITHUB_COPILOT_PROVIDER
from agentm.ai.providers.google import PROVIDER as GOOGLE_PROVIDER
from agentm.ai.providers.google_antigravity import (
    PROVIDER as GOOGLE_ANTIGRAVITY_PROVIDER,
)
from agentm.ai.providers.google_gemini_cli import (
    PROVIDER as GOOGLE_GEMINI_CLI_PROVIDER,
)
from agentm.ai.providers.google_vertex import PROVIDER as GOOGLE_VERTEX_PROVIDER
from agentm.ai.providers.groq import PROVIDER as GROQ_PROVIDER
from agentm.ai.providers.huggingface import PROVIDER as HUGGINGFACE_PROVIDER
from agentm.ai.providers.kimi_coding import PROVIDER as KIMI_CODING_PROVIDER
from agentm.ai.providers.mistral import PROVIDER as MISTRAL_PROVIDER
from agentm.ai.providers.minimax import PROVIDER as MINIMAX_PROVIDER
from agentm.ai.providers.minimax_cn import PROVIDER as MINIMAX_CN_PROVIDER
from agentm.ai.providers.opencode import PROVIDER as OPENCODE_PROVIDER
from agentm.ai.providers.opencode_go import PROVIDER as OPENCODE_GO_PROVIDER
from agentm.ai.providers.openai import PROVIDER as OPENAI_PROVIDER
from agentm.ai.providers.openai_codex import PROVIDER as OPENAI_CODEX_PROVIDER
from agentm.ai.providers.openrouter import PROVIDER as OPENROUTER_PROVIDER
from agentm.ai.providers.vercel_ai_gateway import (
    PROVIDER as VERCEL_AI_GATEWAY_PROVIDER,
)
from agentm.ai.providers.xai import PROVIDER as XAI_PROVIDER
from agentm.ai.providers.zai import PROVIDER as ZAI_PROVIDER


def register_builtin_providers() -> None:
    clear_api_providers()
    for provider in [
        AMAZON_BEDROCK_PROVIDER,
        ANTHROPIC_PROVIDER,
        AZURE_OPENAI_RESPONSES_PROVIDER,
        CEREBRAS_PROVIDER,
        CLOUDFLARE_WORKERS_AI_PROVIDER,
        DEEPSEEK_PROVIDER,
        FIREWORKS_PROVIDER,
        GITHUB_COPILOT_PROVIDER,
        GOOGLE_ANTIGRAVITY_PROVIDER,
        GOOGLE_GEMINI_CLI_PROVIDER,
        GOOGLE_PROVIDER,
        GOOGLE_VERTEX_PROVIDER,
        GROQ_PROVIDER,
        HUGGINGFACE_PROVIDER,
        KIMI_CODING_PROVIDER,
        MINIMAX_CN_PROVIDER,
        MINIMAX_PROVIDER,
        MISTRAL_PROVIDER,
        OPENCODE_GO_PROVIDER,
        OPENCODE_PROVIDER,
        OPENAI_CODEX_PROVIDER,
        OPENAI_PROVIDER,
        OPENROUTER_PROVIDER,
        VERCEL_AI_GATEWAY_PROVIDER,
        XAI_PROVIDER,
        ZAI_PROVIDER,
    ]:
        register_api_provider(provider)
