from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='google-vertex', display_name='Google Vertex', api='google-vertex', default_model='gemini-3.1-pro-preview', env_vars=('GOOGLE_CLOUD_API_KEY',))
