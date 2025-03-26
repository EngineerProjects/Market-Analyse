"""
Default LLM service implementation.

This module provides the default LLM service instance and helper functions
for common LLM operations using the default service.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union, cast

from enterprise_ai.config import config as default_config
from enterprise_ai.llm.base import ConversationManager
from enterprise_ai.llm.retry import DEFAULT_RETRY_CONFIG
from enterprise_ai.schema import Message

from enterprise_ai.llm.service.config import (
    CacheConfig,
    RequestTimeouts,
    ModelSelectionStrategy,
    OrchestratorConfig,
    LLMServiceConfig,
)
from enterprise_ai.llm.service.core import LLMService
from enterprise_ai.llm.service.orchestration import RequestPriority


# Global variable to store the default service instance
_default_llm_service: Optional[LLMService] = None


class DefaultServiceProxy:
    """Proxy for the default LLM service that lazily initializes it."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the real service.

        Args:
            name: Attribute name to access

        Returns:
            The requested attribute from the default service
        """
        return getattr(get_default_llm_service(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> LLMService:
        """Return the actual LLMService instance when called.

        This allows code to use default_llm_service() to get the real service.

        Returns:
            The default LLM service instance
        """
        return get_default_llm_service()


# For backward compatibility with existing imports
default_llm_service = DefaultServiceProxy()


def get_default_llm_service() -> LLMService:
    """Get or create the default LLM service.

    This function lazily initializes the default LLM service when first needed.

    Returns:
        The default LLM service instance
    """
    global _default_llm_service
    if _default_llm_service is None:
        # Create default service directly from global config
        _default_llm_service = LLMService.from_global_config()

        # Configure cache for better defaults
        _default_llm_service.set_config_value("cache_config.cache_type", "hybrid")
        _default_llm_service.set_config_value("cache_config.synchronize_writes", False)

        # Enable provider pooling for better performance
        _default_llm_service.set_config_value("enable_provider_pooling", True)
        _default_llm_service.set_config_value("provider_pool_size", (2, 5))

    return _default_llm_service


# Helper functions to use the default service
def complete(messages: List[Message], **kwargs: Any) -> Message:
    """Generate a completion using the default LLM service.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Generated completion message
    """
    return cast(Message, get_default_llm_service().complete(messages, **kwargs))


def complete_stream(messages: List[Message], **kwargs: Any) -> Generator[Message, None, None]:
    """Generate a streaming completion using the default LLM service.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Generator yielding partial completion messages
    """
    return cast(
        Generator[Message, None, None],
        get_default_llm_service().complete_stream(messages, **kwargs),
    )


async def acomplete(
    messages: List[Message],
    priority: Union[int, RequestPriority] = RequestPriority.NORMAL,
    **kwargs: Any,
) -> Message:
    """Generate a completion using the default LLM service asynchronously.

    Args:
        messages: List of messages in the conversation
        priority: Request priority (lower values = higher priority)
        **kwargs: Additional completion options

    Returns:
        Generated completion message
    """
    return await get_default_llm_service().acomplete(messages, priority=priority, **kwargs)


async def acomplete_stream(
    messages: List[Message],
    priority: Union[int, RequestPriority] = RequestPriority.NORMAL,
    **kwargs: Any,
) -> AsyncGenerator[Message, None]:
    """Generate a streaming completion using the default LLM service asynchronously.

    Args:
        messages: List of messages in the conversation
        priority: Request priority (lower values = higher priority)
        **kwargs: Additional completion options

    Yields:
        Partial completion messages
    """
    async for message in get_default_llm_service().acomplete_stream(
        messages, priority=priority, **kwargs
    ):
        yield message


def get_conversation_manager(
    system_message: Optional[str] = None, max_tokens: Optional[int] = None
) -> ConversationManager:
    """Get a conversation manager using the default LLM service.

    Args:
        system_message: Optional system message
        max_tokens: Maximum token limit (default: provider's limit)

    Returns:
        ConversationManager instance
    """
    return get_default_llm_service().get_conversation_manager(
        system_message=system_message, max_tokens=max_tokens
    )


def clear_cache() -> None:
    """Clear the default LLM service cache."""
    get_default_llm_service().clear_cache()


def change_model(model_name: str, validate: bool = True) -> bool:
    """Change the model for the default LLM service.

    Args:
        model_name: New model name to use
        validate: Whether to validate the model exists

    Returns:
        True if successful, False if model validation fails
    """
    return get_default_llm_service().change_model(model_name, validate)


def change_provider(
    provider_name: str, model_name: Optional[str] = None, validate: bool = True
) -> bool:
    """Change the provider for the default LLM service.

    Args:
        provider_name: New provider name to use
        model_name: New model name to use (uses current if None)
        validate: Whether to validate the model exists

    Returns:
        True if successful, False if provider or model validation fails
    """
    return get_default_llm_service().change_provider(provider_name, model_name, validate)


def get_available_models() -> List[str]:
    """Get available models from the current provider.

    Returns:
        List of available model names
    """
    return get_default_llm_service().get_available_models()


def get_similar_models(model_name: str, max_suggestions: int = 3) -> List[str]:
    """Get similar models to the provided model name.

    Args:
        model_name: Model name to find similar models for
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of similar model names
    """
    return get_default_llm_service().get_similar_models(model_name, max_suggestions)


async def batch_complete(
    message_batches: List[List[Message]],
    priorities: Optional[List[Union[int, RequestPriority]]] = None,
    deduplicate: bool = True,
    **kwargs: Any,
) -> List[Message]:
    """Complete multiple message batches using the default LLM service.

    Args:
        message_batches: List of message batches to complete
        priorities: Priority level for each batch
        deduplicate: Whether to deduplicate identical requests
        **kwargs: Additional completion options

    Returns:
        List of completion messages
    """
    return await get_default_llm_service().batch_complete(
        message_batches, priorities, deduplicate, **kwargs
    )


def get_metrics() -> Dict[str, Any]:
    """Get metrics from the default LLM service.

    Returns:
        Dictionary of metrics
    """
    return get_default_llm_service().get_metrics()


def reset_metrics() -> None:
    """Reset metrics for the default LLM service."""
    get_default_llm_service().reset_metrics()


async def shutdown() -> None:
    """Shutdown the default LLM service."""
    global _default_llm_service
    if _default_llm_service is not None:
        await _default_llm_service.shutdown()
        _default_llm_service = None
