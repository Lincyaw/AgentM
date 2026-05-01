from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='vercel-ai-gateway', display_name='Vercel AI Gateway', api='openai-completions', default_model='zai/glm-5.1', env_vars=('AI_GATEWAY_API_KEY',))
