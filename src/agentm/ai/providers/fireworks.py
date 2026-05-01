from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='fireworks', display_name='Fireworks', api='openai-completions', default_model='accounts/fireworks/models/kimi-k2p6', env_vars=('FIREWORKS_API_KEY',))
