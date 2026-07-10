"""
Provider adapters. Only the OpenAI provider is included in this tutorial —
heterogeneous multi-provider access flows through OpenRouter instead (see
toolkit.config.DEBATE_MODEL_ROSTER).

Import provider modules directly so consumers only pay for the SDKs they use:

    from toolkit.providers import openai_provider
"""

from toolkit.providers._keys import PROVIDER_ENV

__all__ = ["PROVIDER_ENV"]
