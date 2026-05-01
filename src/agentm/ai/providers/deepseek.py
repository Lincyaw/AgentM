from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='deepseek', display_name='DeepSeek', api='openai-completions', default_model='deepseek-v4-pro', env_vars=('DEEPSEEK_API_KEY',))
