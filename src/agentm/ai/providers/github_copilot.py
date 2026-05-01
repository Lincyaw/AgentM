from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='github-copilot', display_name='GitHub Copilot', api='openai-completions', default_model='gpt-5.4', env_vars=('COPILOT_GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN'))
