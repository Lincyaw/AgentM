from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='cerebras', display_name='Cerebras', api='openai-completions', default_model='zai-glm-4.7', env_vars=('CEREBRAS_API_KEY',))
