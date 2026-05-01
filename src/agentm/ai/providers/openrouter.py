from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='openrouter', display_name='OpenRouter', api='openai-completions', default_model='moonshotai/kimi-k2.6', env_vars=('OPENROUTER_API_KEY',))
