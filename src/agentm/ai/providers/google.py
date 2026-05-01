from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='google', display_name='Google Gemini', api='google-generative-ai', default_model='gemini-3.1-pro-preview', env_vars=('GEMINI_API_KEY',))
