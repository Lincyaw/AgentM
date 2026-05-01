from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='azure-openai-responses', display_name='Azure OpenAI Responses', api='azure-openai-responses', default_model='gpt-5.4', env_vars=('AZURE_OPENAI_API_KEY',))
