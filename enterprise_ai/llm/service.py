"""
LLM service for Enterprise AI.

This module provides a high-level interface for working with language models
in the Enterprise AI platform, abstracting away provider-specific details.
"""

import os
from typing import Any, Dict, List, Optional, Union, Generator, AsyncGenerator, Type, cast

from enterprise_ai.schema import Message
from enterprise_ai.config import config
from enterprise_ai.exceptions import LLMError, ModelNotAvailable
from enterprise_ai.logger import get_logger, trace_execution

from enterprise_ai.llm.base import LLMProvider, ConversationManager
from enterprise_ai.llm.exceptions import ProviderNotSupportedError
from enterprise_ai.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from enterprise_ai.llm.utils import encode_image_file, encode_image_bytes, encode_image_from_url
from enterprise_ai.llm.cache import MemoryCache, DiskCache, generate_cache_key, default_cache
from enterprise_ai.llm.retry import with_retry, RetryConfig, DEFAULT_RETRY_CONFIG

# Initialize logger
logger = get_logger("llm")


class LLMService:
    """High-level service for LLM interactions."""

    # Map provider types to implementation classes
    PROVIDER_MAP = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        cache: Optional[Union[MemoryCache, DiskCache]] = None,
        retry_config: Optional[RetryConfig] = None,
        use_caching: bool = True,
    ):
        """Initialize the LLM service.

        Args:
            provider_name: Provider to use (default: from config)
            model_name: Model to use (default: from config)
            cache: Cache to use (default: module default cache)
            retry_config: Retry configuration (default: module default config)
            use_caching: Whether to use caching (default: True)
        """
        # Get provider and model from config if not specified
        if provider_name is None or model_name is None:
            # Use default LLM configuration
            llm_config = config.llm.get("default")
            if not llm_config:
                raise LLMError("No default LLM configuration found")

            provider_name = provider_name or llm_config.api_type
            model_name = model_name or llm_config.model

        self.provider_name = provider_name.lower()
        self.model_name = model_name

        # Initialize the provider
        self.provider = self._create_provider(self.provider_name, self.model_name)

        # Set caching options
        self.use_caching = use_caching
        self.cache = cache or default_cache

        # Set retry configuration
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG

        logger.info(
            f"Initialized LLM service with provider '{provider_name}' and model '{model_name}'"
        )

    def _create_provider(self, provider_name: str, model_name: str) -> LLMProvider:
        """Create a provider instance based on provider name.

        Args:
            provider_name: Provider name
            model_name: Model name

        Returns:
            Provider instance

        Raises:
            ProviderNotSupportedError: If the provider is not supported
        """
        provider_class = self.PROVIDER_MAP.get(provider_name.lower())

        if not provider_class:
            raise ProviderNotSupportedError(provider_name)

        # Get provider-specific configuration
        provider_config = config.llm.get(provider_name, config.llm.get("default"))

        # Create provider configuration with safe access
        provider_config_dict: Dict[str, Any] = {}
        if provider_config is not None:
            provider_config_dict = {
                "api_key": getattr(provider_config, "api_key", None),
                "api_base": getattr(provider_config, "base_url", None),
                "temperature": getattr(provider_config, "temperature", 0.7),
                "max_tokens": getattr(provider_config, "max_tokens", None),
            }

        # Create provider instance with appropriate configuration
        if provider_name == "openai":
            return OpenAIProvider(model_name=model_name, **provider_config_dict)
        elif provider_name == "anthropic":
            return AnthropicProvider(model_name=model_name, **provider_config_dict)
        elif provider_name == "ollama":
            return OllamaProvider(model_name=model_name, **provider_config_dict)
        else:
            # This should never happen due to the earlier check
            raise ProviderNotSupportedError(provider_name)

    def get_conversation_manager(
        self, system_message: Optional[str] = None, max_tokens: Optional[int] = None
    ) -> ConversationManager:
        """Get a conversation manager for this LLM service.

        Args:
            system_message: Optional system message
            max_tokens: Maximum token limit (default: provider's limit)

        Returns:
            ConversationManager instance
        """
        return ConversationManager(
            provider=self.provider, system_message=system_message, max_tokens=max_tokens
        )

    @trace_execution()
    @with_retry()
    def complete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generated completion message
        """
        # Skip cache if streaming or if caching is disabled
        if kwargs.get("stream", False) or not self.use_caching:
            return self.provider.complete(messages, **kwargs)

        # Generate cache key
        cache_key = generate_cache_key(self.model_name, messages, kwargs)

        # Check cache
        cached_response = self.cache.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for {cache_key[:8]}...")

            # Convert cached response to Message
            content = cached_response.get("content")
            tool_calls = cached_response.get("tool_calls")

            if tool_calls:
                from enterprise_ai.schema import Function, ToolCall

                # Convert raw tool calls to ToolCall objects
                tc_objects = []
                for tc in tool_calls:
                    function = Function(
                        name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                    )
                    tc_objects.append(ToolCall(id=tc["id"], type=tc["type"], function=function))

                # Create message with tool calls
                message = Message.from_tool_calls(tc_objects, content or "")
                message.metadata = cached_response.get("metadata", {})
                return message
            else:
                # Create standard message
                message = Message.assistant_message(content)
                message.metadata = cached_response.get("metadata", {})
                return message

        # Cache miss, call the provider
        logger.debug(f"Cache miss for {cache_key[:8]}...")
        response = self.provider.complete(messages, **kwargs)

        # Store in cache
        cached_data: Dict[str, Any] = {"content": response.content, "metadata": response.metadata}

        # Add tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_data = []
            for tc in response.tool_calls:
                tool_calls_data.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                )
            cached_data["tool_calls"] = tool_calls_data

        self.cache.set(cache_key, cached_data)

        return response

    @trace_execution()
    @with_retry()
    def complete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion for the given messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generator yielding partial completion messages
        """
        # Streaming always skips cache
        kwargs["stream"] = True
        return self.provider.complete_stream(messages, **kwargs)

    @trace_execution()
    @with_retry()
    async def acomplete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generated completion message
        """
        # Skip cache if streaming or if caching is disabled
        if kwargs.get("stream", False) or not self.use_caching:
            return await self.provider.acomplete(messages, **kwargs)

        # Generate cache key
        cache_key = generate_cache_key(self.model_name, messages, kwargs)

        # Check cache
        cached_response = self.cache.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for {cache_key[:8]}...")

            # Convert cached response to Message
            content = cached_response.get("content")
            tool_calls = cached_response.get("tool_calls")

            if tool_calls:
                from enterprise_ai.schema import Function, ToolCall

                # Convert raw tool calls to ToolCall objects
                tc_objects = []
                for tc in tool_calls:
                    function = Function(
                        name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                    )
                    tc_objects.append(ToolCall(id=tc["id"], type=tc["type"], function=function))

                # Create message with tool calls
                message = Message.from_tool_calls(tc_objects, content or "")
                message.metadata = cached_response.get("metadata", {})
                return message
            else:
                # Create standard message
                message = Message.assistant_message(content)
                message.metadata = cached_response.get("metadata", {})
                return message

        # Cache miss, call the provider
        logger.debug(f"Cache miss for {cache_key[:8]}...")
        response = await self.provider.acomplete(messages, **kwargs)

        # Store in cache
        cached_data: Dict[str, Any] = {"content": response.content, "metadata": response.metadata}

        # Add tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_data = []
            for tc in response.tool_calls:
                tool_calls_data.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                )
            cached_data["tool_calls"] = tool_calls_data

        self.cache.set(cache_key, cached_data)

        return response

    @trace_execution()
    @with_retry()
    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion for the given messages asynchronously."""
        # Streaming always skips cache
        kwargs["stream"] = True

        # Use an explicit type cast to help the type checker
        coroutine_result = self.provider.acomplete_stream(messages, **kwargs)
        generator = cast(AsyncGenerator[Message, None], await coroutine_result)

        # Now iterate over the properly typed generator
        async for message in generator:
            yield message

    def count_tokens(self, messages: List[Message]) -> int:
        """Count the number of tokens in the given messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """
        return self.provider.count_tokens(messages)

    @property
    def max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model.

        Returns:
            Maximum token count
        """
        return self.provider.get_max_tokens()

    @property
    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        return self.provider.supports_vision()

    @property
    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        return self.provider.supports_tools()

    def clear_cache(self) -> None:
        """Clear the service's cache."""
        if self.cache:
            self.cache.clear()
            logger.info("LLM service cache cleared")


# Create project workspace cache directory
workspace_cache_dir = config.workspace_root / "cache" / "llm"
workspace_cache_dir.mkdir(parents=True, exist_ok=True)

# Create disk cache for persistent storage
disk_cache = DiskCache(
    cache_dir=workspace_cache_dir,
    ttl=86400,  # 24 hours
    max_size_mb=500,  # 500 MB
)

# Default LLM service instance
default_llm_service = LLMService(
    cache=disk_cache, use_caching=True, retry_config=DEFAULT_RETRY_CONFIG
)


# Helper functions to use the default service
def complete(messages: List[Message], **kwargs: Any) -> Message:
    """Generate a completion using the default LLM service.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Generated completion message
    """
    return cast(Message, default_llm_service.complete(messages, **kwargs))


def complete_stream(messages: List[Message], **kwargs: Any) -> Generator[Message, None, None]:
    """Generate a streaming completion using the default LLM service.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Generator yielding partial completion messages
    """
    return cast(
        Generator[Message, None, None], default_llm_service.complete_stream(messages, **kwargs)
    )


async def acomplete(messages: List[Message], **kwargs: Any) -> Message:
    """Generate a completion using the default LLM service asynchronously.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Generated completion message
    """
    return cast(Message, await default_llm_service.acomplete(messages, **kwargs))


async def acomplete_stream(messages: List[Message], **kwargs: Any) -> AsyncGenerator[Message, None]:
    """Generate a streaming completion using the default LLM service asynchronously.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Async generator yielding partial completion messages
    """
    async for message in default_llm_service.acomplete_stream(messages, **kwargs):
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
    return default_llm_service.get_conversation_manager(
        system_message=system_message, max_tokens=max_tokens
    )


def clear_cache() -> None:
    """Clear the default LLM service cache."""
    default_llm_service.clear_cache()
