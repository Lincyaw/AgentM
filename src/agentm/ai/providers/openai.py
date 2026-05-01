from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='openai', display_name='OpenAI', api='openai-responses', default_model='gpt-5.4', env_vars=('OPENAI_API_KEY',))
