from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='kimi-coding', display_name='Kimi Coding', api='openai-completions', default_model='kimi-for-coding', env_vars=('KIMI_API_KEY',))
