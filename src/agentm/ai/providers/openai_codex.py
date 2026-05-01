from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='openai-codex', display_name='OpenAI Codex', api='openai-codex-responses', default_model='gpt-5.5', env_vars=())
