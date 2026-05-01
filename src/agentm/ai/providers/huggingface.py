from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='huggingface', display_name='Hugging Face', api='openai-completions', default_model='moonshotai/Kimi-K2.6', env_vars=('HF_TOKEN',))
