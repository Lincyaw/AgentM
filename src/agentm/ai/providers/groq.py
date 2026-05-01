from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='groq', display_name='Groq', api='openai-completions', default_model='openai/gpt-oss-120b', env_vars=('GROQ_API_KEY',))
