from __future__ import annotations

from agentm.ai.providers._shared import make_stub_provider

PROVIDER = make_stub_provider(provider_id='cloudflare-workers-ai', display_name='Cloudflare Workers AI', api='openai-completions', default_model='@cf/moonshotai/kimi-k2.6', env_vars=('CLOUDFLARE_API_KEY',))
