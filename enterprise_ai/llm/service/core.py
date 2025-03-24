"""
Core LLM service implementation.

This module provides the main LLMService class for interacting with
language models through a unified interface with advanced capabilities
including caching, fallbacks, and provider management.
"""

import re
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

# Import cache classes at the top level

if TYPE_CHECKING:
    pass

from enterprise_ai.llm.cache import MemoryCache, DiskCache
from enterprise_ai.config import Config, config as default_config
from enterprise_ai.exceptions import LLMError, ModelNotAvailable
from enterprise_ai.llm.base import LLMProvider, ConversationManager
from enterprise_ai.llm.exceptions import (
    ProviderNotSupportedError,
    ModelCapabilityError,
    ParameterError,
    ContextWindowExceededError,
)
from enterprise_ai.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from enterprise_ai.llm.retry import with_retry, RetryConfig, DEFAULT_RETRY_CONFIG
from enterprise_ai.llm.service.config import LLMServiceConfig
from enterprise_ai.logger import get_logger, trace_execution
from enterprise_ai.schema import Message, Role

from enterprise_ai.llm.service.caching import HybridCache, generate_cache_key

from enterprise_ai.llm.service.metrics import ServiceMetrics
from enterprise_ai.llm.service.orchestration import RequestOrchestrator, RequestPriority
from enterprise_ai.llm.service.pools import ProviderPool
from enterprise_ai.llm.service.registry import ProviderRegistration

# Initialize logger
logger = get_logger("llm.service")

# Type variables
P = TypeVar("P", bound=LLMProvider)


class LLMService:
    """High-level service for LLM interactions.

    This service provides a unified interface to multiple LLM providers,
    with support for model validation, capability detection, caching,
    automatic retries, and performance optimization.
    """

    # Registry of provider implementations
    _PROVIDER_REGISTRY: Dict[str, ProviderRegistration] = {}

    # Initialize registry with built-in providers
    @classmethod
    def initialize_registry(cls) -> None:
        """Initialize the provider registry with built-in providers."""
        if not cls._PROVIDER_REGISTRY:
            cls._PROVIDER_REGISTRY = {
                "openai": ProviderRegistration(
                    name="openai",
                    provider_class=OpenAIProvider,
                    default_model="gpt-4o",
                    base_url="https://api.openai.com/v1",
                    capabilities={"vision": True, "tools": True},
                ),
                "anthropic": ProviderRegistration(
                    name="anthropic",
                    provider_class=AnthropicProvider,
                    default_model="claude-3-opus-20240229",
                    base_url="https://api.anthropic.com/v1",
                    capabilities={"vision": True, "tools": True},
                ),
                "ollama": ProviderRegistration(
                    name="ollama",
                    provider_class=OllamaProvider,
                    default_model="llama3",
                    base_url="http://localhost:11434",
                    capabilities={"vision": True, "tools": True},
                ),
            }

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[LLMProvider],
        default_model: str,
        base_url: Optional[str] = None,
        capabilities: Optional[Dict[str, bool]] = None,
        create_func: Optional[Any] = None,
    ) -> None:
        """Register a new provider with the service.

        Args:
            name: Provider name (lowercase)
            provider_class: Provider class
            default_model: Default model name for this provider
            base_url: Default API base URL
            capabilities: Default capabilities dictionary
            create_func: Custom factory function for provider creation
        """
        cls.initialize_registry()
        cls._PROVIDER_REGISTRY[name.lower()] = ProviderRegistration(
            name=name.lower(),
            provider_class=provider_class,
            default_model=default_model,
            base_url=base_url,
            capabilities=capabilities,
            create_func=create_func,
        )
        logger.info(f"Registered LLM provider: {name}")

    def __init__(self, config: Optional[LLMServiceConfig] = None):
        """Initialize the LLM service.

        Args:
            config: Service configuration (uses defaults if not provided)
        """
        # Initialize the provider registry
        self.__class__.initialize_registry()

        # Set configuration
        self.config = config or LLMServiceConfig()

        # Initialize config-related variables
        self._config_instance = self._load_config_instance()

        # Get provider and model from config if not specified
        provider_name, model_name = self._resolve_provider_and_model()
        self.provider_name = provider_name.lower()
        self.model_name = model_name

        # Initialize caching
        self._init_cache()

        # Initialize metrics
        self.metrics = ServiceMetrics() if self.config.enable_metrics else None

        # Setup retry config
        self.retry_config = self.config.retry_config or DEFAULT_RETRY_CONFIG

        # Track model validation status
        self._model_validated = False
        self._available_models: Optional[List[str]] = None
        self._model_capabilities: Optional[Dict[str, bool]] = None

        # Create fallback models map
        self.fallback_models = self._setup_fallback_models()

        # Initialize request orchestrator
        self.orchestrator = self._init_orchestrator()

        # Setup provider pool if enabled
        self.provider_pool = (
            self._init_provider_pool() if self.config.enable_provider_pooling else None
        )

        # Initialize the provider
        self.provider = self._create_provider(self.provider_name, self.model_name)

        logger.info(
            f"Initialized LLM service with provider '{provider_name}' and model '{model_name}'"
        )

    def _load_config_instance(self) -> Config:
        """Load the configuration instance.

        Returns:
            Config instance
        """
        if self.config.config_path:
            return Config(config_path=self.config.config_path)
        return default_config

    def _resolve_provider_and_model(self) -> Tuple[str, str]:
        """Resolve provider and model from configuration.

        Returns:
            Tuple of (provider_name, model_name)
        """
        provider_name = self.config.provider_name
        model_name = self.config.model_name

        if not (provider_name and model_name):
            # Use default LLM configuration
            llm_config = self._config_instance.llm.get("default")
            if not llm_config:
                raise LLMError("No default LLM configuration found")

            provider_name = provider_name or llm_config.api_type
            model_name = model_name or llm_config.model

        # Apply model selection strategy if provided
        if self.config.model_selection and not self.config.model_name:
            model_name = self.config.model_selection.preferred_model

        return provider_name, model_name

    def _init_cache(self) -> None:
        """Initialize the cache based on configuration."""
        cache_config = self.config.cache_config

        if not cache_config.use_cache:
            self.cache: Optional[Union["MemoryCache", "DiskCache", "HybridCache"]] = None
            self.use_caching = False
            return

        self.use_caching = True

        if cache_config.cache_type == "memory":
            # Import here to avoid circular imports
            from enterprise_ai.llm.cache import MemoryCache

            self.cache = cast(
                Union[MemoryCache, DiskCache, HybridCache],
                MemoryCache(ttl=cache_config.ttl, max_entries=cache_config.max_entries),
            )

        elif cache_config.cache_type == "hybrid":
            # Use the enhanced hybrid cache
            cache_dir = cache_config.cache_dir
            if not cache_dir:
                cache_dir = self._config_instance.workspace_root / "cache" / "llm"
                cache_dir.mkdir(parents=True, exist_ok=True)

            self.cache = HybridCache(
                memory_ttl=cache_config.ttl,
                disk_ttl=cache_config.ttl * 2 if cache_config.ttl else None,
                memory_max_entries=cache_config.max_entries,
                disk_cache_dir=cache_dir,
                disk_max_size_mb=cache_config.max_size_mb,
                promotion_policy=cache_config.promotion_policy,
                synchronize_writes=cache_config.synchronize_writes,
            )
        else:
            # Default to memory cache
            from enterprise_ai.llm.cache import MemoryCache

            self.cache = MemoryCache(ttl=cache_config.ttl)

    def _setup_fallback_models(self) -> Dict[str, List[str]]:
        """Set up fallback models for each provider.

        Returns:
            Dictionary mapping providers to fallback model lists
        """
        fallbacks: Dict[str, List[str]] = {
            "openai": ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
            "anthropic": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
            "ollama": ["llama3", "llama2", "mistral"],
        }

        # Apply custom fallbacks from model selection strategy
        if self.config.model_selection and self.config.model_selection.fallback_models:
            strategy = self.config.model_selection

            # Only override fallbacks for the current provider
            if self.provider_name in fallbacks:
                fallbacks[self.provider_name] = strategy.fallback_models

        return fallbacks

    def _init_orchestrator(self) -> RequestOrchestrator:
        """Initialize the request orchestrator.

        Returns:
            RequestOrchestrator instance
        """
        import asyncio

        orch_config = self.config.orchestrator_config

        # Define rate limits if not specified
        rate_limits = orch_config.rate_limits or {
            "openai": 100.0,  # Higher limit for OpenAI
            "anthropic": 20.0,  # Lower limit for Anthropic
            "ollama": 10.0,  # Lowest for local Ollama
        }

        orchestrator = RequestOrchestrator(
            max_concurrent_requests=orch_config.max_concurrent_requests,
            max_queue_size=orch_config.max_queue_size,
            rate_limits=rate_limits,
            priority_levels=orch_config.priority_levels,
            adaptive_scaling=orch_config.adaptive_scaling,
            max_retries=orch_config.max_retries,
        )

        # Start the orchestrator
        asyncio.create_task(orchestrator.start())

        return orchestrator

    def _init_provider_pool(self) -> ProviderPool:
        """Initialize the provider pool.

        Returns:
            ProviderPool instance
        """
        min_size, max_size = self.config.provider_pool_size

        def create_provider() -> LLMProvider:
            """Factory function to create new provider instances."""
            return self._create_provider(self.provider_name, self.model_name)

        return ProviderPool(
            provider_factory=create_provider,
            min_size=min_size,
            max_size=max_size,
            idle_timeout=300.0,  # 5 minutes
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
        # Get provider registration
        provider_reg = self._PROVIDER_REGISTRY.get(provider_name.lower())

        if not provider_reg:
            raise ProviderNotSupportedError(provider_name)

        # Get provider-specific configuration
        provider_config = self._config_instance.llm.get(
            provider_name, self._config_instance.llm.get("default")
        )

        # Create provider configuration with appropriate timeouts
        provider_config_dict: Dict[str, Any] = {}
        if provider_config is not None:
            provider_config_dict = {
                "api_key": getattr(provider_config, "api_key", None),
                "api_base": getattr(provider_config, "base_url", None),
                "temperature": getattr(provider_config, "temperature", 0.7),
                "max_tokens": getattr(provider_config, "max_tokens", None),
                "request_timeout": self.config.timeouts.default_timeout,
            }

            # Add organization for OpenAI if available
            if provider_name == "openai" and hasattr(provider_config, "organization"):
                provider_config_dict["organization"] = provider_config.organization

            # Add API version if available
            if hasattr(provider_config, "api_version"):
                provider_config_dict["api_version"] = provider_config.api_version

        # Override with direct parameters from service config
        if self.config.api_key:
            provider_config_dict["api_key"] = self.config.api_key

        if self.config.api_base:
            provider_config_dict["api_base"] = self.config.api_base

        if self.config.api_version:
            provider_config_dict["api_version"] = self.config.api_version

        if self.config.temperature is not None:
            provider_config_dict["temperature"] = self.config.temperature

        if self.config.max_tokens is not None:
            provider_config_dict["max_tokens"] = self.config.max_tokens

        if self.config.organization:
            provider_config_dict["organization"] = self.config.organization

        # Add validation parameters
        provider_config_dict["validate_model"] = self.config.validate_model
        provider_config_dict["strict_validation"] = self.config.strict_validation
        provider_config_dict["connection_pool_size"] = self.config.connection_pool_size

        # Use custom creation function if available
        if provider_reg.create_func:
            return cast(
                LLMProvider, provider_reg.create_func(model_name=model_name, **provider_config_dict)
            )

        # Otherwise, create provider with standard constructor
        return cast(
            LLMProvider, provider_reg.provider_class(model_name=model_name, **provider_config_dict)
        )

    def get_available_models(self) -> List[str]:
        """Get a list of available models from the current provider.

        Returns:
            List of available model names
        """
        if self._available_models is not None:
            return self._available_models

        result: List[str] = []

        if hasattr(self.provider, "get_available_models"):
            try:
                # Use the provider's method if available
                provider_models = self.provider.get_available_models()
                # Ensure we return a List[str]
                if isinstance(provider_models, list):
                    result = [str(model) for model in provider_models]
                else:
                    logger.warning(f"Provider returned non-list models: {provider_models}")
            except Exception as e:
                logger.warning(f"Error getting available models: {e}")

        self._available_models = result
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
                start_time = time.time()
                validation_result = self.provider.validate_model()

                # Record validation time if metrics enabled
                if self.metrics:
                    validation_time = time.time() - start_time
                    self.metrics.record_response_time(validation_time)

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
            similar_models = self._find_similar_models(
                model_name, available_models, max_suggestions
            )

        # Limit to max_suggestions
        return similar_models[:max_suggestions]

    def _find_similar_models(
        self, model_name: str, available_models: List[str], max_suggestions: int = 3
    ) -> List[str]:
        """Find similar models using various matching strategies.

        Args:
            model_name: Model name to match
            available_models: List of available models
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of similar model names
        """
        similar_models: List[str] = []

        # Check for exact prefix match
        model_prefix = model_name.split("-")[0] if "-" in model_name else model_name
        prefix_matches = [m for m in available_models if m.startswith(model_prefix)]
        similar_models.extend(prefix_matches)

        # If no prefix matches, try substring match
        if not similar_models:
            substring_matches = [
                m
                for m in available_models
                if model_prefix.lower() in m.lower() and m not in similar_models
            ]
            similar_models.extend(substring_matches)

        # If still no matches, use model family matching
        if not similar_models:
            family_pattern = r"([a-zA-Z]+)(\d*)"
            match = re.match(family_pattern, model_name)
            if match:
                family = match.group(1).lower() if match else ""
                family_matches = [
                    m
                    for m in available_models
                    if (match_m := re.match(family_pattern, m))
                    and match_m.group(1).lower() == family
                    and m not in similar_models
                ]
                similar_models.extend(family_matches)

        # Limit results
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
        # Use provider from pool if pooling is enabled
        if self.provider_pool:
            provider = self.provider_pool.acquire()
            try:
                manager = ConversationManager(
                    provider=provider, system_message=system_message, max_tokens=max_tokens
                )
                # Return manager wrapped to release provider on __del__
                return self._wrap_conversation_manager(manager, provider)
            except Exception:
                # Release provider on error
                self.provider_pool.release(provider)
                raise

        # Use default provider otherwise
        return ConversationManager(
            provider=self.provider, system_message=system_message, max_tokens=max_tokens
        )

    def _wrap_conversation_manager(
        self, manager: ConversationManager, provider: LLMProvider
    ) -> ConversationManager:
        """Wrap a conversation manager to release provider when done.

        Args:
            manager: Conversation manager
            provider: Provider to release

        Returns:
            Wrapped conversation manager
        """
        # Store original __del__ method
        original_del = getattr(manager, "__del__", lambda: None)

        # Capture service instance (self) in a variable for the closure
        service = self

        # Define new __del__ method
        def new_del(self_cm: ConversationManager) -> None:
            # Call original __del__
            original_del()
            # Release provider using the captured service instance
            if service.provider_pool:
                service.provider_pool.release(provider)

        # Attach method to manager
        setattr(manager, "__del__", new_del.__get__(manager))

        return manager

    def _handle_model_not_available(self, error: ModelNotAvailable) -> LLMProvider:
        """Handle a model not available error by trying fallback models.

        Args:
            error: The model not available error

        Returns:
            A provider instance for the fallback model

        Raises:
            ModelNotAvailable: If no fallback model is available
        """
        # Get fallback models for this provider
        fallback_models = self.fallback_models.get(self.provider_name, [])

        if not fallback_models:
            # No fallback models, re-raise the error
            raise error

        # Try each fallback model
        original_model = self.model_name
        last_error = error

        for fallback_model in fallback_models:
            try:
                logger.warning(
                    f"Model '{original_model}' not available. Trying fallback '{fallback_model}'"
                )

                # Update model name
                self.model_name = fallback_model

                # Create a new provider with the fallback model
                provider = self._create_provider(self.provider_name, fallback_model)

                # Validate the model if required
                if self.config.validate_model:
                    if hasattr(provider, "validate_model"):
                        if not provider.validate_model():
                            logger.warning(f"Fallback model '{fallback_model}' validation failed")
                            continue
                    else:
                        # Can't validate, assume valid
                        pass

                return provider

            except Exception as e:
                logger.warning(f"Error with fallback model '{fallback_model}': {e}")
                last_error = ModelNotAvailable(fallback_model, f"Fallback model not available: {e}")

        # Reset to original model
        self.model_name = original_model

        # All fallbacks failed
        raise last_error

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
        # Start timing
        start_time = time.time()
        is_cached = False

        try:
            # Skip cache if streaming or if caching is disabled
            if kwargs.get("stream", False) or not self.use_caching or not self.cache:
                try:
                    # Use provider pool if enabled
                    if self.provider_pool:
                        provider = self.provider_pool.acquire()
                        try:
                            response = provider.complete(messages, **kwargs)
                        finally:
                            self.provider_pool.release(provider)
                    else:
                        response = self.provider.complete(messages, **kwargs)

                    # Record metrics if enabled
                    if self.metrics:
                        self.metrics.record_request(self.provider_name, cached=False)
                        self.metrics.record_tokens(self.count_tokens(messages))

                    return response
                except ModelNotAvailable as e:
                    # Try to use a fallback model
                    self.provider = self._handle_model_not_available(e)
                    response = self.provider.complete(messages, **kwargs)

                    # Record metrics if enabled
                    if self.metrics:
                        self.metrics.record_request(self.provider_name, cached=False)
                        self.metrics.record_tokens(self.count_tokens(messages))

                    return response

            # Generate cache key
            cache_key = generate_cache_key(self.model_name, messages, kwargs)

            # Check cache
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                is_cached = True

                # Record metrics if enabled
                if self.metrics:
                    self.metrics.record_request(self.provider_name, cached=True)

                # Convert cached response to Message
                return self._convert_cached_response_to_message(cached_response)

            # Cache miss, call the provider
            logger.debug(f"Cache miss for {cache_key[:8]}...")

            try:
                # Use provider pool if enabled
                if self.provider_pool:
                    provider = self.provider_pool.acquire()
                    try:
                        response = provider.complete(messages, **kwargs)
                    finally:
                        self.provider_pool.release(provider)
                else:
                    response = self.provider.complete(messages, **kwargs)
            except ModelNotAvailable as e:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                response = self.provider.complete(messages, **kwargs)

            # Store in cache
            cached_data = self._prepare_response_for_cache(response)
            self.cache.set(cache_key, cached_data)

            return response

        except Exception:
            # Record error if metrics enabled
            if self.metrics:
                self.metrics.record_error(self.provider_name)
            raise
        finally:
            # Record metrics if enabled
            if self.metrics:
                completion_time = time.time() - start_time
                self.metrics.record_response_time(completion_time)

                if not is_cached:
                    self.metrics.record_tokens(self.count_tokens(messages))

    def _convert_cached_response_to_message(self, cached_response: Dict[str, Any]) -> Message:
        """Convert a cached response to a Message object.

        Args:
            cached_response: Cached response data

        Returns:
            Message object
        """
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

    def _prepare_response_for_cache(self, response: Message) -> Dict[str, Any]:
        """Prepare a response for caching.

        Args:
            response: Response message

        Returns:
            Dictionary for caching
        """
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

        return cached_data

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
        # Start timing
        start_time = time.time()

        try:
            # Streaming always skips cache
            kwargs["stream"] = True

            # Apply streaming timeout if configured
            if self.config.timeouts.streaming_timeout != self.config.timeouts.default_timeout:
                # Try to update the timeout in the provider's client
                if hasattr(self.provider, "client") and hasattr(self.provider.client, "timeout"):
                    original_timeout = self.provider.client.timeout
                    self.provider.client.timeout = self.config.timeouts.streaming_timeout
            else:
                original_timeout = None

            try:
                # Use provider pool if enabled
                if self.provider_pool:
                    provider = self.provider_pool.acquire()
                    try:
                        for chunk in provider.complete_stream(messages, **kwargs):
                            yield chunk
                    finally:
                        self.provider_pool.release(provider)
                else:
                    for chunk in self.provider.complete_stream(messages, **kwargs):
                        yield chunk
            except ModelNotAvailable as e:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                for chunk in self.provider.complete_stream(messages, **kwargs):
                    yield chunk

        except Exception:
            # Record error if metrics enabled
            if self.metrics:
                self.metrics.record_error(self.provider_name)
            raise
        finally:
            # Restore original timeout if it was changed
            if original_timeout is not None:
                if hasattr(self.provider, "client") and hasattr(self.provider.client, "timeout"):
                    self.provider.client.timeout = original_timeout

            # Record metrics if enabled
            if self.metrics:
                self.metrics.record_request(self.provider_name, cached=False)
                completion_time = time.time() - start_time
                self.metrics.record_response_time(completion_time)
                self.metrics.record_tokens(self.count_tokens(messages))

    async def acomplete(
        self,
        messages: List[Message],
        priority: Union[int, RequestPriority] = RequestPriority.NORMAL,
        **kwargs: Any,
    ) -> Message:
        """Generate a completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            priority: Request priority
            **kwargs: Additional completion options

        Returns:
            Generated completion message
        """
        # Start timing
        start_time = time.time()
        is_cached = False

        try:
            # Skip cache if streaming or if caching is disabled
            if kwargs.get("stream", False) or not self.use_caching or not self.cache:
                # Use the orchestrator for request processing
                # This is the line that needs fixing - directly await the result
                return await self._orchestrated_complete(messages, priority, **kwargs)

            # Generate cache key
            cache_key = generate_cache_key(self.model_name, messages, kwargs)

            # Check cache
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                is_cached = True

                # Record metrics if enabled
                if self.metrics:
                    self.metrics.record_request(self.provider_name, cached=True)

                # Convert cached response to Message
                return self._convert_cached_response_to_message(cached_response)

            # Cache miss, use orchestrator
            logger.debug(f"Cache miss for {cache_key[:8]}...")
            # Fixed: Directly await the orchestrated complete coroutine
            response = await self._orchestrated_complete(messages, priority, **kwargs)

            # Store in cache
            cached_data = self._prepare_response_for_cache(response)
            self.cache.set(cache_key, cached_data)

            return response

        except Exception:
            # Record error if metrics enabled
            if self.metrics:
                self.metrics.record_error(self.provider_name)
            raise
        finally:
            # Record metrics if enabled
            if self.metrics and self.metrics:
                completion_time = time.time() - start_time
                self.metrics.record_response_time(completion_time)

                if not is_cached:
                    self.metrics.record_tokens(self.count_tokens(messages))

    async def _orchestrated_complete(
        self, messages: List[Message], priority: Union[int, RequestPriority], **kwargs: Any
    ) -> Message:
        """Process a completion request through the orchestrator."""

        # Define function to be executed
        async def _complete_func() -> Message:
            try:
                # Use provider pool if enabled
                if self.provider_pool:
                    provider = self.provider_pool.acquire()
                    try:
                        return await provider.acomplete(messages, **kwargs)
                    finally:
                        self.provider_pool.release(provider)
                else:
                    return await self.provider.acomplete(messages, **kwargs)
            except ModelNotAvailable as e:
                # Try to use a fallback model
                self.provider = self._handle_model_not_available(e)
                return await self.provider.acomplete(messages, **kwargs)

        # Submit to orchestrator - this returns a future we need to await
        future = await self.orchestrator.submit(
            provider=self.provider_name,
            fn=_complete_func,
            priority=priority,
            deduplicate=self.config.orchestrator_config.enable_deduplication,
        )

        # Explicitly await and ensure we return a Message rather than a coroutine
        result = await future

        return cast(Message, result)

    async def acomplete_stream(
        self,
        messages: List[Message],
        priority: Union[int, RequestPriority] = RequestPriority.NORMAL,
        **kwargs: Any,
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion for the given messages asynchronously."""

        # Start timing
        start_time = time.time()

        try:
            # Streaming always skips cache
            kwargs["stream"] = True

            # Apply streaming timeout if configured
            original_timeout = None
            if self.config.timeouts.streaming_timeout != self.config.timeouts.default_timeout:
                if hasattr(self.provider, "async_client") and hasattr(
                    self.provider.async_client, "timeout"
                ):
                    original_timeout = self.provider.async_client.timeout
                    self.provider.async_client.timeout = self.config.timeouts.streaming_timeout

            try:
                # Use provider pool if enabled
                if self.provider_pool:
                    provider = self.provider_pool.acquire()
                    try:
                        # First get the coroutine
                        stream_coro = provider.acomplete_stream(messages, **kwargs)
                        # Then await it to get the AsyncGenerator
                        stream_generator = await stream_coro
                        # Now we can iterate over the generator
                        async for chunk in cast(AsyncGenerator[Message, None], stream_generator):
                            yield chunk
                    finally:
                        self.provider_pool.release(provider)
                else:
                    # Same pattern for the main provider
                    stream_coro = self.provider.acomplete_stream(messages, **kwargs)
                    stream_generator = await stream_coro
                    async for chunk in cast(AsyncGenerator[Message, None], stream_generator):
                        yield chunk
            except ModelNotAvailable as e:
                # Handle fallback
                self.provider = self._handle_model_not_available(e)
                # Same pattern for the fallback provider
                stream_coro = self.provider.acomplete_stream(messages, **kwargs)
                stream_generator = await stream_coro
                async for chunk in cast(AsyncGenerator[Message, None], stream_generator):
                    yield chunk

        except Exception:
            # Record error if metrics enabled
            if self.metrics:
                self.metrics.record_error(self.provider_name)
            raise

        finally:
            # Restore original timeout if it was changed
            if original_timeout is not None:
                if hasattr(self.provider, "async_client") and hasattr(
                    self.provider.async_client, "timeout"
                ):
                    self.provider.async_client.timeout = original_timeout

            # Record metrics if enabled
            if self.metrics:
                self.metrics.record_request(self.provider_name, cached=False)
                completion_time = time.time() - start_time
                self.metrics.record_response_time(completion_time)
                self.metrics.record_tokens(self.count_tokens(messages))

    async def batch_complete(
        self,
        message_batches: List[List[Message]],
        priorities: Optional[List[Union[int, RequestPriority]]] = None,
        deduplicate: bool = True,
        **kwargs: Any,
    ) -> List[Message]:
        """Complete multiple message batches with advanced orchestration.

        Args:
            message_batches: List of message batches to complete
            priorities: Priority level for each batch
            deduplicate: Whether to deduplicate identical requests
            **kwargs: Additional completion options

        Returns:
            List of completion messages
        """
        if not message_batches:
            return []

        # Default priorities if not specified
        if not priorities:
            priorities = [RequestPriority.NORMAL] * len(message_batches)
        elif len(priorities) < len(message_batches):
            # Extend with default priority
            priorities.extend([RequestPriority.NORMAL] * (len(message_batches) - len(priorities)))

        # Submit all requests to orchestrator and collect futures
        request_futures = []
        for messages, priority in zip(message_batches, priorities):
            # Get a future for the completion
            completion_future = self.acomplete(
                messages=messages, priority=priority, deduplicate=deduplicate, **kwargs
            )
            request_futures.append(completion_future)

        # Gather all results by awaiting each future
        results = []
        for future in request_futures:
            result = await future
            results.append(result)

        return results

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

                if not is_valid and self.config.strict_validation:
                    raise ModelNotAvailable(model_name, f"Model {model_name} not available")

                if not is_valid:
                    return False

            # Update the provider and model name
            self.provider = new_provider
            self.model_name = model_name
            self._model_validated = True if not validate else is_valid

            # Update provider pool if enabled
            if self.provider_pool:
                # Reset the pool with new model
                min_size, max_size = self.config.provider_pool_size
                self.provider_pool = ProviderPool(
                    provider_factory=lambda: self._create_provider(
                        self.provider_name, self.model_name
                    ),
                    min_size=min_size,
                    max_size=max_size,
                )

            # Return success
            return True

        except ModelNotAvailable:
            if self.config.strict_validation:
                raise
            return False
        except Exception as e:
            logger.error(f"Error changing model: {e}")
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
        if provider_name.lower() not in self._PROVIDER_REGISTRY:
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

                if not is_valid and self.config.strict_validation:
                    raise ModelNotAvailable(new_model, f"Model {new_model} not available")

                if not is_valid:
                    return False

            # Update the provider and model name
            self.provider = new_provider
            self.provider_name = provider_name.lower()
            self.model_name = new_model
            self._model_validated = True if not validate else is_valid
            self._model_capabilities = None  # Reset capabilities cache

            # Update fallback models
            self.fallback_models = self._setup_fallback_models()

            # Update provider pool if enabled
            if self.provider_pool:
                # Reset the pool with new provider/model
                min_size, max_size = self.config.provider_pool_size
                self.provider_pool = ProviderPool(
                    provider_factory=lambda: self._create_provider(
                        self.provider_name, self.model_name
                    ),
                    min_size=min_size,
                    max_size=max_size,
                )

            return True

        except ModelNotAvailable:
            if self.config.strict_validation:
                raise
            return False
        except Exception as e:
            logger.error(f"Error changing provider: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics.

        Returns:
            Dictionary of metrics or empty dict if metrics disabled
        """
        metrics = {}

        # Get service-level metrics
        if self.metrics:
            metrics.update(self.metrics.get_metrics())

        # Get orchestrator metrics
        if hasattr(self.orchestrator, "stats"):
            orchestrator_metrics = {}
            for key, value in self.orchestrator.stats.items():
                orchestrator_metrics[f"orchestrator_{key}"] = value
            metrics["orchestrator"] = orchestrator_metrics

        # Get cache metrics if available
        if self.cache and isinstance(self.cache, HybridCache):
            cache_stats = self.cache.get_stats()
            metrics["cache"] = cache_stats
        elif self.cache:
            # Basic metrics for other cache types
            metrics["cache"] = {
                "type": self.config.cache_config.cache_type,
                "enabled": self.use_caching,
                "ttl": self.config.cache_config.ttl,
            }

        # Add provider information
        metrics["provider"] = {
            "name": self.provider_name,
            "model": self.model_name,
            "pooling_enabled": self.provider_pool is not None,
        }

        # Add configuration summary
        metrics["config"] = {
            "validate_model": self.config.validate_model,
            "connection_pool_size": self.config.connection_pool_size,
            "timeout": self.config.timeouts.default_timeout,
        }

        return metrics

    def reset_metrics(self) -> None:
        """Reset all service metrics."""
        if self.metrics:
            self.metrics.reset()

        # Reset orchestrator metrics
        if hasattr(self.orchestrator, "stats"):
            for key in self.orchestrator.stats:
                self.orchestrator.stats[key] = (
                    0 if isinstance(self.orchestrator.stats[key], int) else 0.0
                )

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        # Shutdown the orchestrator
        if self.orchestrator:
            await self.orchestrator.shutdown()

        # Clear the cache if needed
        if self.cache:
            self.clear_cache()

        logger.info("LLM service shut down")
