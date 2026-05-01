from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='opencode-go', display_name='OpenCode Go', api='openai-completions', default_model='kimi-k2.6', env_vars=('OPENCODE_API_KEY',))
