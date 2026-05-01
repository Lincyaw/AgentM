from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='minimax-cn', display_name='MiniMax CN', api='openai-completions', default_model='MiniMax-M2.7', env_vars=('MINIMAX_CN_API_KEY',))
