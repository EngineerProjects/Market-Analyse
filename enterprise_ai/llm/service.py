"""
LLM service for Enterprise AI.

This module provides a high-level interface for working with language models
in the Enterprise AI platform, abstracting away provider-specific details.
It integrates with multiple LLM providers (OpenAI, Anthropic, Ollama) and
supports advanced features like model validation, capability detection, and
intelligent fallbacks.
"""

import os
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Generator,
    AsyncGenerator,
    Type,
    Set,
    cast,
    Tuple,
)

from enterprise_ai.schema import Message, Role
from enterprise_ai.config import Config, config as default_config
from enterprise_ai.exceptions import LLMError, ModelNotAvailable
from enterprise_ai.logger import get_logger, trace_execution

from enterprise_ai.llm.base import LLMProvider, ConversationManager
from enterprise_ai.llm.exceptions import (
    ProviderNotSupportedError,
    ModelCapabilityError,
    ParameterError,
    ContextWindowExceededError,
)
from enterprise_ai.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from enterprise_ai.llm.utils import encode_image_file, encode_image_bytes, encode_image_from_url
from enterprise_ai.llm.cache import MemoryCache, DiskCache, generate_cache_key, default_cache
from enterprise_ai.llm.retry import (
    with_retry,
    RetryConfig,
    DEFAULT_RETRY_CONFIG,
    RATE_LIMIT_RETRY_CONFIG,
)

# Initialize logger
logger = get_logger("llm")


class LLMService:
    """High-level service for LLM interactions.

    This service provides a unified interface to multiple LLM providers,
    with support for model validation, capability detection, caching,
    and automatic retries.
    """

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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        organization: Optional[str] = None,
        config_path: Optional[str] = None,
        cache: Optional[Union[MemoryCache, DiskCache]] = None,
        retry_config: Optional[RetryConfig] = None,
        use_caching: bool = True,
        validate_model: bool = False,  # Changed from True to False
        strict_validation: bool = False,
        fallback_models: Optional[Dict[str, str]] = None,
        model_capabilities: Optional[Dict[str, bool]] = None,  # New parameter
    ):
        """Initialize the LLM service.

        Args:
            provider_name: Provider to use (default: from config)
            model_name: Model to use (default: from config)
            api_key: API key to use (overrides config)
            api_base: API base URL to use (overrides config)
            api_version: API version to use (overrides config)
            temperature: Model temperature (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            organization: Organization ID (for OpenAI, overrides config)
            config_path: Path to a custom config file
            cache: Cache to use (default: module default cache)
            retry_config: Retry configuration (default: module default config)
            use_caching: Whether to use caching (default: True)
            validate_model: Whether to validate the model with the API (default: False)
            strict_validation: Whether to raise an exception if model doesn't exist (default: False)
            fallback_models: Dictionary mapping of provider -> fallback model if main model unavailable
            model_capabilities: Dictionary of model capabilities (vision, tools, etc.)
        """
        # Initialize custom config if needed
        if config_path:
            config_instance = Config(config_path=config_path)
        else:
            config_instance = default_config

        # Get provider and model from config if not specified
        if provider_name is None or model_name is None:
            # Use default LLM configuration
            llm_config = config_instance.llm.get("default")
            if not llm_config:
                raise LLMError("No default LLM configuration found")

            provider_name = provider_name or llm_config.api_type
            model_name = model_name or llm_config.model

        self.provider_name = provider_name.lower()
        self.model_name = model_name
        self.use_caching = use_caching
        self.cache = cache or default_cache
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self.fallback_models = fallback_models or {}
        self.validate_model = validate_model
        self.strict_validation = strict_validation
        self.model_capabilities = model_capabilities

        # Store the direct parameters for provider creation
        self.direct_params = {
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "organization": organization,
        }

        # Remove None values to allow fallback to config values
        self.direct_params = {k: v for k, v in self.direct_params.items() if v is not None}

        # Track model validation status
        self._model_validated = False
        self._available_models: Optional[List[str]] = None
        self._model_capabilities: Optional[Dict[str, bool]] = None
        self._config_instance = config_instance

        # Initialize the provider
        self.provider = self._create_provider(self.provider_name, self.model_name)

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
        provider_config = self._config_instance.llm.get(
            provider_name, self._config_instance.llm.get("default")
        )

        # Create provider configuration with safe access
        provider_config_dict: Dict[str, Any] = {}
        if provider_config is not None:
            provider_config_dict = {
                "api_key": getattr(provider_config, "api_key", None),
                "api_base": getattr(provider_config, "base_url", None),
                "temperature": getattr(provider_config, "temperature", 0.7),
                "max_tokens": getattr(provider_config, "max_tokens", None),
            }

            # Add organization for OpenAI if available
            if provider_name == "openai" and hasattr(provider_config, "organization"):
                provider_config_dict["organization"] = provider_config.organization

            # Add API version if available
            if hasattr(provider_config, "api_version"):
                provider_config_dict["api_version"] = provider_config.api_version

        # Override config values with direct parameters
        provider_config_dict.update(self.direct_params)

        # Add validation parameters
        provider_config_dict["validate_model"] = self.validate_model
        provider_config_dict["strict_validation"] = self.strict_validation

        # Add model capabilities if specified for Ollama
        if provider_name == "ollama" and self.model_capabilities is not None:
            provider_config_dict["known_capabilities"] = self.model_capabilities

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

    # The rest of the methods remain the same...

    def get_available_models(self) -> List[str]:
        """Get a list of available models from the current provider.

        Returns:
            List of available model names
        """
        result: List[str] = []

        if hasattr(self.provider, "get_available_models"):
            try:
                # Use the provider's method if available
                provider_models = self.provider.get_available_models()
                # Ensure we return a List[str] - convert any non-compliant result
                if isinstance(provider_models, list):
                    result = [str(model) for model in provider_models]
                else:
                    logger.warning(f"Provider returned non-list models: {provider_models}")
            except Exception as e:
                logger.warning(f"Error getting available models: {e}")

        return result

    def validate_current_model(self) -> bool:
        """Validate if the current model exists and is available.

        Returns:
            True if model is valid, False otherwise
        """
        # If already validated, return cached result
        if self._model_validated:
            return True

        # Check if provider has a validate_model method
        if hasattr(self.provider, "validate_model"):
            try:
                validation_result = self.provider.validate_model()
                # Ensure we return a bool
                is_valid = bool(validation_result)
                self._model_validated = is_valid
                return is_valid
            except Exception as e:
                logger.warning(f"Error validating model: {e}")
                return False

        # Otherwise, assume model is valid (can't validate)
        return True

    def get_similar_models(self, model_name: str, max_suggestions: int = 3) -> List[str]:
        """Get similar models to the provided model name.

        Args:
            model_name: Model name to find similar models for
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of similar model names
        """
        available_models = self.get_available_models()
        similar_models: List[str] = []

        if not available_models:
            return similar_models

        # Check if provider has a model suggestion method
        if hasattr(self.provider, "_suggest_models"):
            try:
                # Use the provider's method if available
                suggestions = self.provider._suggest_models(
                    model_name, available_models, max_suggestions
                )
                if isinstance(suggestions, list):
                    similar_models = [str(model) for model in suggestions]
                else:
                    logger.warning(f"Provider returned non-list suggestions: {suggestions}")
            except Exception as e:
                logger.warning(f"Error getting model suggestions: {e}")

        # If we couldn't get suggestions or there was an error, fall back to simple matching
        if not similar_models:
            # Simple matching based on model name prefix
            model_prefix = model_name.split("-")[0] if "-" in model_name else model_name
            similar_models = [m for m in available_models if m.startswith(model_prefix)]

            # If no matches, try matching anywhere in the name
            if not similar_models:
                similar_models = [m for m in available_models if model_prefix.lower() in m.lower()]

        # Limit to max_suggestions
        return similar_models[:max_suggestions]

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

    def _handle_model_not_available(self, error: ModelNotAvailable) -> LLMProvider:
        """Handle a model not available error by trying fallback models.

        Args:
            error: The model not available error

        Returns:
            A provider instance for the fallback model

        Raises:
            ModelNotAvailable: If no fallback model is available
        """
        # Check if a fallback model exists for this provider
        fallback_model = self.fallback_models.get(self.provider_name)

        if not fallback_model:
            # No fallback model, re-raise the error
            raise error

        logger.warning(
            f"Model '{self.model_name}' not available. Falling back to '{fallback_model}'"
        )

        # Create a new provider with the fallback model
        return self._create_provider(self.provider_name, fallback_model)

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
            try:
                return self.provider.complete(messages, **kwargs)
            except ModelNotAvailable as e:
                if self.fallback_models:
                    # Try to use a fallback model
                    self.provider = self._handle_model_not_available(e)
                    return self.provider.complete(messages, **kwargs)
                raise

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
        try:
            response = self.provider.complete(messages, **kwargs)
        except ModelNotAvailable as e:
            if self.fallback_models:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                response = self.provider.complete(messages, **kwargs)
            else:
                raise

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
        try:
            return self.provider.complete_stream(messages, **kwargs)
        except ModelNotAvailable as e:
            if self.fallback_models:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                return self.provider.complete_stream(messages, **kwargs)
            raise

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
            try:
                return await self.provider.acomplete(messages, **kwargs)
            except ModelNotAvailable as e:
                if self.fallback_models:
                    # Try to use a fallback model
                    self.provider = self._handle_model_not_available(e)
                    return await self.provider.acomplete(messages, **kwargs)
                raise

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
        try:
            response = await self.provider.acomplete(messages, **kwargs)
        except ModelNotAvailable as e:
            if self.fallback_models:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                response = await self.provider.acomplete(messages, **kwargs)
            else:
                raise

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
        """Generate a streaming completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Async generator yielding partial completion messages
        """
        # Streaming always skips cache
        kwargs["stream"] = True

        try:
            # Use an explicit type cast to help the type checker
            coroutine_result = self.provider.acomplete_stream(messages, **kwargs)
            generator = cast(AsyncGenerator[Message, None], await coroutine_result)
        except ModelNotAvailable as e:
            if self.fallback_models:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                coroutine_result = self.provider.acomplete_stream(messages, **kwargs)
                generator = cast(AsyncGenerator[Message, None], await coroutine_result)
            else:
                raise

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
        return bool(self.provider.supports_vision())

    @property
    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        return bool(self.provider.supports_tools())

    @property
    def supports_audio(self) -> bool:
        """Check if the provider/model supports audio processing.

        Returns:
            True if audio is supported, False otherwise
        """
        # Check if the provider has audio support method
        if hasattr(self.provider, "supports_audio"):
            try:
                return bool(self.provider.supports_audio())
            except Exception:
                return False
        return False

    def clear_cache(self) -> None:
        """Clear the service's cache."""
        if self.cache:
            self.cache.clear()
            logger.info("LLM service cache cleared")

    def change_model(self, model_name: str, validate: bool = True) -> bool:
        """Change the current model while keeping the same provider.

        Args:
            model_name: New model name to use
            validate: Whether to validate the model exists

        Returns:
            True if successful, False if model validation fails

        Raises:
            ModelNotAvailable: If strict validation is enabled and model doesn't exist
        """
        # Don't do anything if it's the same model
        if model_name == self.model_name:
            return True

        try:
            # Create a new provider instance with the new model
            new_provider = self._create_provider(self.provider_name, model_name)

            # Validate the model if requested
            is_valid = True
            if validate:
                is_valid = False

                if hasattr(new_provider, "validate_model"):
                    try:
                        validation_result = new_provider.validate_model()
                        is_valid = bool(validation_result)
                    except Exception:
                        is_valid = False
                else:
                    # Can't validate, assume valid
                    is_valid = True

                if not is_valid and self.strict_validation:
                    raise ModelNotAvailable(model_name, f"Model {model_name} not available")

                if not is_valid:
                    return False

            # Update the provider and model name
            self.provider = new_provider
            self.model_name = model_name
            self._model_validated = True if not validate else is_valid
            return True

        except ModelNotAvailable:
            if self.strict_validation:
                raise
            return False
        except Exception:
            return False

    def change_provider(
        self, provider_name: str, model_name: Optional[str] = None, validate: bool = True
    ) -> bool:
        """Change the current provider and optionally the model.

        Args:
            provider_name: New provider name to use
            model_name: New model name to use (uses current if None)
            validate: Whether to validate the model exists

        Returns:
            True if successful, False if provider or model validation fails

        Raises:
            ProviderNotSupportedError: If the provider is not supported
            ModelNotAvailable: If strict validation is enabled and model doesn't exist
        """
        # Check if provider is supported
        if provider_name.lower() not in self.PROVIDER_MAP:
            raise ProviderNotSupportedError(provider_name)

        # Use current model if not specified
        new_model = model_name or self.model_name

        try:
            # Create a new provider instance
            new_provider = self._create_provider(provider_name.lower(), new_model)

            # Validate the model if requested
            is_valid = True
            if validate:
                is_valid = False

                if hasattr(new_provider, "validate_model"):
                    try:
                        validation_result = new_provider.validate_model()
                        is_valid = bool(validation_result)
                    except Exception:
                        is_valid = False
                else:
                    # Can't validate, assume valid
                    is_valid = True

                if not is_valid and self.strict_validation:
                    raise ModelNotAvailable(new_model, f"Model {new_model} not available")

                if not is_valid:
                    return False

            # Update the provider and model name
            self.provider = new_provider
            self.provider_name = provider_name.lower()
            self.model_name = new_model
            self._model_validated = True if not validate else is_valid
            self._model_capabilities = None  # Reset capabilities cache
            return True

        except ModelNotAvailable:
            if self.strict_validation:
                raise
            return False
        except Exception:
            return False


# Global variable to store the default service instance
_default_llm_service: Optional[LLMService] = None

# Define fallback models for each provider
DEFAULT_FALLBACK_MODELS = {
    "openai": "gpt-3.5-turbo",  # Fallback to 3.5 if requested model is unavailable
    "anthropic": "claude-3-haiku",  # Fallback to smallest Claude 3 model
    "ollama": "llama3",  # Fallback to Llama 3 base model
}


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
        # Create workspace cache directory
        workspace_cache_dir = default_config.workspace_root / "cache" / "llm"
        workspace_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create disk cache for persistent storage
        disk_cache = DiskCache(
            cache_dir=workspace_cache_dir,
            ttl=86400,  # 24 hours
            max_size_mb=500,  # 500 MB
        )

        # Create default service
        _default_llm_service = LLMService(
            cache=disk_cache,
            use_caching=True,
            retry_config=DEFAULT_RETRY_CONFIG,
            fallback_models=DEFAULT_FALLBACK_MODELS,
        )

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


async def acomplete(messages: List[Message], **kwargs: Any) -> Message:
    """Generate a completion using the default LLM service asynchronously.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Generated completion message
    """
    return cast(Message, await get_default_llm_service().acomplete(messages, **kwargs))


async def acomplete_stream(messages: List[Message], **kwargs: Any) -> AsyncGenerator[Message, None]:
    """Generate a streaming completion using the default LLM service asynchronously.

    Args:
        messages: List of messages in the conversation
        **kwargs: Additional completion options

    Returns:
        Async generator yielding partial completion messages
    """
    async for message in get_default_llm_service().acomplete_stream(messages, **kwargs):
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
