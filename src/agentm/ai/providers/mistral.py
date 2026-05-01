from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='mistral', display_name='Mistral', api='mistral-conversations', default_model='devstral-medium-latest', env_vars=('MISTRAL_API_KEY',))
