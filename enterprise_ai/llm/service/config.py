"""
Configuration classes for the LLM service.

This module provides configuration classes for various aspects of the LLM service,
including caching, timeouts, model selection, and orchestration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal

from enterprise_ai.llm.retry import RetryConfig, DEFAULT_RETRY_CONFIG


class CacheConfig:
    """Configuration for the LLM service cache."""

    def __init__(
        self,
        use_cache: bool = True,
        cache_type: Literal["memory", "disk", "hybrid"] = "memory",
        ttl: Optional[int] = 3600,
        max_size_mb: int = 500,
        cache_dir: Optional[Union[str, Path]] = None,
        max_entries: int = 1000,
        promotion_policy: Literal["read", "write", "both"] = "both",
        synchronize_writes: bool = True,
    ):
        """Initialize cache configuration.

        Args:
            use_cache: Whether to use caching
            cache_type: Type of cache to use
            ttl: Time-to-live in seconds (None for no expiration)
            max_size_mb: Maximum cache size in MB (for disk cache)
            cache_dir: Directory for cache files (for disk cache)
            max_entries: Maximum number of cache entries (for memory cache)
            promotion_policy: When to promote items from disk to memory in hybrid cache
            synchronize_writes: Whether to wait for disk writes to complete
        """
        self.use_cache = use_cache
        self.cache_type = cache_type
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_entries = max_entries
        self.promotion_policy = promotion_policy
        self.synchronize_writes = synchronize_writes


class RequestTimeouts:
    """Timeout configuration for the LLM service."""

    def __init__(
        self,
        default_timeout: float = 60.0,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
        streaming_timeout: Optional[float] = None,
        async_timeout: Optional[float] = None,
    ):
        """Initialize timeout configuration.

        Args:
            default_timeout: Default timeout for all requests
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            streaming_timeout: Timeout for streaming requests
            async_timeout: Timeout for async requests
        """
        self.default_timeout = default_timeout
        self.connect_timeout = connect_timeout or default_timeout
        self.read_timeout = read_timeout or default_timeout
        self.streaming_timeout = streaming_timeout or default_timeout * 2
        self.async_timeout = async_timeout or default_timeout

    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary for httpx.

        Returns:
            Timeout dictionary
        """
        return {
            "default": self.default_timeout,
            "connect": self.connect_timeout,
            "read": self.read_timeout,
        }


class ModelSelectionStrategy:
    """Strategy for model selection and fallback."""

    def __init__(
        self,
        preferred_model: str,
        fallback_models: Optional[List[str]] = None,
        auto_fallback: bool = True,
        capability_requirements: Optional[Dict[str, bool]] = None,
        max_cost_tier: Optional[int] = None,
        fallback_across_providers: bool = False,
        provider_preferences: Optional[List[str]] = None,
    ):
        """Initialize model selection strategy.

        Args:
            preferred_model: Preferred model name
            fallback_models: List of fallback models in order of preference
            auto_fallback: Whether to automatically suggest fallbacks
            capability_requirements: Required capabilities (vision, tools, etc.)
            max_cost_tier: Maximum cost tier (1-5, None for no limit)
            fallback_across_providers: Whether to fallback to different providers
            provider_preferences: List of preferred providers in order
        """
        self.preferred_model = preferred_model
        self.fallback_models = fallback_models or []
        self.auto_fallback = auto_fallback
        self.capability_requirements = capability_requirements or {}
        self.max_cost_tier = max_cost_tier
        self.fallback_across_providers = fallback_across_providers
        self.provider_preferences = provider_preferences or []


class OrchestratorConfig:
    """Configuration for request orchestration."""

    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_queue_size: int = 100,
        rate_limits: Optional[Dict[str, float]] = None,
        priority_levels: int = 4,
        adaptive_scaling: bool = True,
        max_retries: int = 3,
        enable_deduplication: bool = True,
        deduplication_ttl: float = 5.0,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ):
        """Initialize orchestrator configuration.

        Args:
            max_concurrent_requests: Maximum concurrent requests
            max_queue_size: Maximum size of the request queue
            rate_limits: Rate limits by provider
            priority_levels: Number of priority levels
            adaptive_scaling: Whether to dynamically adjust concurrency
            max_retries: Maximum number of retries for failed requests
            enable_deduplication: Whether to enable request deduplication
            deduplication_ttl: Time-to-live for deduplication entries
            enable_circuit_breaker: Whether to enable circuit breakers
            circuit_breaker_threshold: Failure threshold for circuit breakers
            circuit_breaker_timeout: Recovery timeout for circuit breakers
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.rate_limits = rate_limits or {}
        self.priority_levels = priority_levels
        self.adaptive_scaling = adaptive_scaling
        self.max_retries = max_retries
        self.enable_deduplication = enable_deduplication
        self.deduplication_ttl = deduplication_ttl
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout


class LLMServiceConfig:
    """Configuration for the LLM service."""

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
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        timeouts: Optional[RequestTimeouts] = None,
        validate_model: bool = False,
        strict_validation: bool = False,
        model_selection: Optional[ModelSelectionStrategy] = None,
        connection_pool_size: int = 10,
        enable_metrics: bool = True,
        log_level: str = "INFO",
        orchestrator_config: Optional[OrchestratorConfig] = None,
        enable_provider_pooling: bool = False,
        provider_pool_size: Tuple[int, int] = (1, 5),  # (min_size, max_size)
    ):
        """Initialize LLM service configuration.

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
            cache_config: Cache configuration
            retry_config: Retry configuration
            timeouts: Request timeout configuration
            validate_model: Whether to validate the model with the API
            strict_validation: Whether to raise an exception if model doesn't exist
            model_selection: Model selection and fallback strategy
            connection_pool_size: Size of the connection pool
            enable_metrics: Whether to collect metrics
            log_level: Logging level
            orchestrator_config: Request orchestration configuration
            enable_provider_pooling: Whether to enable provider pooling
            provider_pool_size: Min and max provider pool size
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.organization = organization
        self.config_path = config_path
        self.cache_config = cache_config or CacheConfig()
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self.timeouts = timeouts or RequestTimeouts()
        self.validate_model = validate_model
        self.strict_validation = strict_validation
        self.model_selection = model_selection
        self.connection_pool_size = connection_pool_size
        self.enable_metrics = enable_metrics
        self.log_level = log_level
        self.orchestrator_config = orchestrator_config or OrchestratorConfig()
        self.enable_provider_pooling = enable_provider_pooling
        self.provider_pool_size = provider_pool_size
