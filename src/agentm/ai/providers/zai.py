from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='zai', display_name='Z.ai', api='openai-completions', default_model='glm-5.1', env_vars=('ZAI_API_KEY',))
