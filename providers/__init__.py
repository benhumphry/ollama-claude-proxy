"""
LLM Providers package.

This module loads provider configurations from YAML files and registers
them with the global registry. To add a new provider:

1. Add the provider definition to config/providers.yml
2. Create config/models/{provider_name}.yml with model definitions

OpenAI-compatible providers work automatically. Only providers with custom
SDK requirements (like Anthropic) need a Python class.
"""

import logging

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ModelInfo, OpenAICompatibleProvider, get_api_key
from .loader import (
    get_all_provider_names,
    get_default_config,
    get_provider_config,
    load_models_for_provider,
)
from .registry import registry

logger = logging.getLogger(__name__)

# Only providers with custom SDK requirements need a class
# OpenAI-compatible providers are created directly from config
CUSTOM_PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
}


def _create_provider(provider_name: str, config: dict) -> LLMProvider | None:
    """Create a provider instance from config.

    Args:
        provider_name: Name of the provider
        config: Provider configuration dict

    Returns:
        Provider instance or None if creation failed
    """
    provider_type = config.get("type", provider_name)

    # Load models and aliases from config
    models, aliases = load_models_for_provider(provider_name)

    if not models:
        logger.warning(f"No models loaded for provider '{provider_name}', skipping")
        return None

    # Check if this provider needs a custom class (non-OpenAI SDK)
    if provider_name in CUSTOM_PROVIDER_CLASSES:
        provider_class = CUSTOM_PROVIDER_CLASSES[provider_name]
        return provider_class(models=models, aliases=aliases)

    # OpenAI-compatible providers are created directly from config
    if provider_type == "openai-compatible":
        base_url = config.get("base_url")
        api_key_env = config.get("api_key_env")

        if not base_url or not api_key_env:
            logger.error(f"Provider '{provider_name}' missing base_url or api_key_env")
            return None

        return OpenAICompatibleProvider(
            name=provider_name,
            base_url=base_url,
            api_key_env=api_key_env,
            models=models,
            aliases=aliases,
        )

    logger.error(f"Unknown provider type '{provider_type}' for '{provider_name}'")
    return None


def _register_providers():
    """Load and register all providers from config."""
    provider_names = get_all_provider_names()

    for provider_name in provider_names:
        config = get_provider_config(provider_name)
        if not config:
            continue

        provider = _create_provider(provider_name, config)
        if provider:
            registry.register(provider)
            logger.info(f"Registered provider: {provider_name}")

    # Set default provider/model from config
    default_provider, default_model = get_default_config()
    registry.set_default(default_provider, default_model)
    logger.info(f"Default: {default_provider}/{default_model}")


# Register all providers on module import
_register_providers()

__all__ = [
    "registry",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ModelInfo",
    "get_api_key",
    "AnthropicProvider",
]
