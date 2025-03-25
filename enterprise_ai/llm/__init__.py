"""
Enterprise AI LLM interface.

This module provides interfaces for working with language models through
different providers, with a unified service layer and utilities.
"""

# Base classes
from enterprise_ai.llm.base import LLMProvider, ConversationManager

# Schema classes
from enterprise_ai.schema import Message, Role, ToolCall, Function

# Provider implementations
from enterprise_ai.llm.providers import AnthropicProvider, OpenAIProvider, OllamaProvider

# Retry functionality
from enterprise_ai.llm.retry import (
    RetryConfig,
    with_retry,
    DEFAULT_RETRY_CONFIG,
    BackoffStrategy,
    RetryableException,
)

# Utility functions for LLM handling
from enterprise_ai.llm.utils import (
    TokenCounter,
    retry_with_exponential_backoff,
)

# Import service components (import from service subpackage, not service.py)
from enterprise_ai.llm.service import (
    # Core service
    LLMService,
    # Default service instance and proxy
    get_default_llm_service,
    default_llm_service,
    # Helper functions for common operations
    complete,
    complete_stream,
    acomplete,
    acomplete_stream,
    get_conversation_manager,
    clear_cache,
    change_model,
    change_provider,
    get_available_models,
    get_similar_models,
    batch_complete,
    get_metrics,
    reset_metrics,
    shutdown,
    # Configuration classes
    CacheConfig,
    RequestTimeouts,
    ModelSelectionStrategy,
    OrchestratorConfig,
    LLMServiceConfig,
)

# Exception classes
from enterprise_ai.llm.exceptions import (
    TokenCountError,
    ProviderNotSupportedError,
    ParameterError,
    ContextWindowExceededError,
    ImageProcessingError,
    ModelCapabilityError,
    OllamaError,
    OllamaModelUnavailableError,
    OllamaConnectionError,
)

# Image processing utilities
from enterprise_ai.llm.image import ImageHandler

__all__ = [
    # Base classes
    "LLMProvider",
    "ConversationManager",
    # Schema classes
    "Message",
    "Role",
    "ToolCall",
    "Function",
    # Provider implementations
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    # Retry functionality
    "RetryConfig",
    "with_retry",
    "DEFAULT_RETRY_CONFIG",
    "BackoffStrategy",
    "RetryableException",
    # Utility functions
    "TokenCounter",
    "retry_with_exponential_backoff",
    # Core service and default instance
    "LLMService",
    "get_default_llm_service",
    "default_llm_service",
    # Helper functions
    "complete",
    "complete_stream",
    "acomplete",
    "acomplete_stream",
    "get_conversation_manager",
    "clear_cache",
    "change_model",
    "change_provider",
    "get_available_models",
    "get_similar_models",
    "batch_complete",
    "get_metrics",
    "reset_metrics",
    "shutdown",
    # Configuration classes
    "CacheConfig",
    "RequestTimeouts",
    "ModelSelectionStrategy",
    "OrchestratorConfig",
    "LLMServiceConfig",
    # Exception classes
    "TokenCountError",
    "ProviderNotSupportedError",
    "ParameterError",
    "ContextWindowExceededError",
    "ImageProcessingError",
    "ModelCapabilityError",
    "OllamaError",
    "OllamaModelUnavailableError",
    "OllamaConnectionError",
    # Image processing utilities
    "ImageHandler",
]
