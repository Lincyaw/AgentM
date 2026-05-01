from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='minimax', display_name='MiniMax', api='openai-completions', default_model='MiniMax-M2.7', env_vars=('MINIMAX_API_KEY',))
