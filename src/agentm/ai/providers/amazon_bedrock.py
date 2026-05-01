from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='amazon-bedrock', display_name='Amazon Bedrock', api='bedrock-converse-stream', default_model='us.anthropic.claude-opus-4-6-v1', env_vars=())
