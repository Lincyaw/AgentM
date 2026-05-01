from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='xai', display_name='xAI', api='openai-completions', default_model='grok-4.20-0309-reasoning', env_vars=('XAI_API_KEY',))
