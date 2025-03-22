"""
LLM integration for Enterprise AI.

This package provides a unified interface for interacting with various
large language model providers, including OpenAI, Anthropic, and Ollama.
"""

from enterprise_ai.llm.base import LLMProvider, ConversationManager
from enterprise_ai.llm.exceptions import (
    TokenCountError,
    ProviderNotSupportedError,
    ParameterError,
    ContextWindowExceededError,
    ImageProcessingError,
    ModelCapabilityError,
)
from enterprise_ai.llm.utils import (
    TokenCounter,
    encode_image_file,
    encode_image_bytes,
    encode_image_from_url,
    retry_with_exponential_backoff,
)
from enterprise_ai.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from enterprise_ai.llm.service import (
    LLMService,
    default_llm_service,
    complete,
    complete_stream,
    acomplete,
    acomplete_stream,
    get_conversation_manager,
    clear_cache,
    MemoryCache,
    DiskCache,
    RetryConfig,
)

# Re-export commonly used classes and functions
__all__ = [
    # Base classes
    "LLMProvider",
    "ConversationManager",
    # Service class and helpers
    "LLMService",
    "default_llm_service",
    "complete",
    "complete_stream",
    "acomplete",
    "acomplete_stream",
    "get_conversation_manager",
    "clear_cache",
    "MemoryCache",
    "DiskCache",
    "RetryConfig",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    # Exceptions
    "TokenCountError",
    "ProviderNotSupportedError",
    "ParameterError",
    "ContextWindowExceededError",
    "ImageProcessingError",
    "ModelCapabilityError",
    # Utilities
    "TokenCounter",
    "encode_image_file",
    "encode_image_bytes",
    "encode_image_from_url",
    "retry_with_exponential_backoff",
]
