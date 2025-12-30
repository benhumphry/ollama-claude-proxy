"""
Google Gemini provider.

Uses Google's OpenAI-compatible API endpoint, so we can leverage the
OpenAI SDK with a different base URL.
"""

from .base import ModelInfo, OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    """Provider for Google Gemini models."""

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env = "GOOGLE_API_KEY"

    models: dict[str, ModelInfo] = {
        # Gemini 3 family (latest)
        "gemini-3-flash": ModelInfo(
            family="gemini-3",
            description="Gemini 3 Flash - Latest fast and versatile model",
            context_length=1000000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
        "gemini-3-pro": ModelInfo(
            family="gemini-3",
            description="Gemini 3 Pro - Most capable Gemini model",
            context_length=1000000,
            capabilities=["vision", "analysis", "coding", "writing", "reasoning"],
        ),
        # Gemini 2.5 family
        "gemini-2.5-flash": ModelInfo(
            family="gemini-2.5",
            description="Gemini 2.5 Flash - Fast and versatile",
            context_length=1000000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
        "gemini-2.5-pro": ModelInfo(
            family="gemini-2.5",
            description="Gemini 2.5 Pro - Previous flagship model",
            context_length=1000000,
            capabilities=["vision", "analysis", "coding", "writing", "reasoning"],
        ),
        # Gemini 2.0 family
        "gemini-2.0-flash": ModelInfo(
            family="gemini-2.0",
            description="Gemini 2.0 Flash - Previous generation fast model",
            context_length=1000000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
        "gemini-2.0-flash-lite": ModelInfo(
            family="gemini-2.0",
            description="Gemini 2.0 Flash Lite - Lightweight fast model",
            context_length=1000000,
            capabilities=["analysis", "coding", "writing", "fast"],
        ),
        # Gemini 1.5 family (legacy)
        "gemini-1.5-pro": ModelInfo(
            family="gemini-1.5",
            description="Gemini 1.5 Pro - Legacy pro model",
            context_length=2000000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        "gemini-1.5-flash": ModelInfo(
            family="gemini-1.5",
            description="Gemini 1.5 Flash - Legacy fast model",
            context_length=1000000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
    }

    aliases: dict[str, str] = {
        # Gemini 3 aliases (latest)
        "gemini-flash": "gemini-3-flash",
        "gemini-pro": "gemini-3-pro",
        "gemini-3": "gemini-3-flash",
        # Gemini 2.5 aliases
        "gemini-2.5": "gemini-2.5-flash",
        # Gemini 2.0 aliases
        "gemini-2.0": "gemini-2.0-flash",
        "gemini-flash-lite": "gemini-2.0-flash-lite",
        # Gemini 1.5 aliases
        "gemini-1.5": "gemini-1.5-flash",
        # Generic alias (points to latest)
        "gemini": "gemini-3-flash",
    }
