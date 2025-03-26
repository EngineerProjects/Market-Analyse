"""
Configuration classes for the LLM service.

This module provides configuration classes for various aspects of the LLM service,
including caching, timeouts, model selection, and orchestration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal, cast

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
        retention: Optional[str] = None,
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
            retention: How long to retain cache files
        """
        self.use_cache = use_cache
        self.cache_type = cache_type
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_entries = max_entries
        self.promotion_policy = promotion_policy
        self.synchronize_writes = synchronize_writes
        self.retention = retention

    def update(self, new_config: "CacheConfig") -> None:
        """Update configuration with values from new_config.

        Args:
            new_config: New configuration values
        """
        for attr_name in [
            "use_cache",
            "cache_type",
            "ttl",
            "max_size_mb",
            "cache_dir",
            "max_entries",
            "promotion_policy",
            "synchronize_writes",
            "retention",
        ]:
            new_value = getattr(new_config, attr_name)
            # Only update if the new value is not None
            if new_value is not None:
                setattr(self, attr_name, new_value)


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

    def update(self, new_config: "RequestTimeouts") -> None:
        """Update configuration with values from new_config.

        Args:
            new_config: New configuration values
        """
        for attr_name in [
            "default_timeout",
            "connect_timeout",
            "read_timeout",
            "streaming_timeout",
            "async_timeout",
        ]:
            new_value = getattr(new_config, attr_name)
            # Only update if the new value is not None
            if new_value is not None:
                setattr(self, attr_name, new_value)


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

    def update(self, new_config: "ModelSelectionStrategy") -> None:
        """Update configuration with values from new_config.

        Args:
            new_config: New configuration values
        """
        for attr_name in [
            "preferred_model",
            "fallback_models",
            "auto_fallback",
            "capability_requirements",
            "max_cost_tier",
            "fallback_across_providers",
            "provider_preferences",
        ]:
            new_value = getattr(new_config, attr_name)
            # Only update non-empty or non-None values
            if new_value is not None:
                if isinstance(new_value, list) and not new_value:
                    continue  # Skip empty lists
                if isinstance(new_value, dict) and not new_value:
                    continue  # Skip empty dicts
                setattr(self, attr_name, new_value)


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

    def update(self, new_config: "OrchestratorConfig") -> None:
        """Update configuration with values from new_config.

        Args:
            new_config: New configuration values
        """
        for attr_name in [
            "max_concurrent_requests",
            "max_queue_size",
            "rate_limits",
            "priority_levels",
            "adaptive_scaling",
            "max_retries",
            "enable_deduplication",
            "deduplication_ttl",
            "enable_circuit_breaker",
            "circuit_breaker_threshold",
            "circuit_breaker_timeout",
        ]:
            new_value = getattr(new_config, attr_name)
            # Skip empty dictionaries
            if attr_name == "rate_limits" and isinstance(new_value, dict) and not new_value:
                continue
            # Only update if the new value is not None
            if new_value is not None:
                setattr(self, attr_name, new_value)


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

    @classmethod
    def from_file(cls, file_path: str) -> "LLMServiceConfig":
        """Create configuration from a file path (YAML or TOML).

        Args:
            file_path: Path to configuration file

        Returns:
            LLMServiceConfig instance

        Raises:
            ValueError: If file format is unsupported or file doesn't exist
        """
        config_path = Path(file_path)
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {file_path}")

        # Load based on file extension
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML configuration files. Install with 'pip install PyYAML'"
                )

            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".toml":
            try:
                import tomli
            except ImportError:
                raise ImportError(
                    "tomli is required for TOML configuration files. Install with 'pip install tomli'"
                )

            with open(config_path, "rb") as f:
                config_dict = tomli.load(f)
        else:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMServiceConfig":
        """Create configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            LLMServiceConfig instance
        """
        # Extract main config and nested components
        service_config = config_dict.get("llm_service", {})

        # Create nested config objects
        cache_config = cls._extract_cache_config(config_dict.get("cache", {}))
        timeouts = cls._extract_timeout_config(config_dict.get("timeouts", {}))
        model_selection = cls._extract_model_selection(config_dict.get("model_selection", {}))
        orchestrator_config = cls._extract_orchestrator_config(config_dict.get("orchestrator", {}))
        retry_config = cls._extract_retry_config(config_dict.get("retry", {}))

        # Create config instance with extracted values
        return cls(
            provider_name=service_config.get("provider_name"),
            model_name=service_config.get("model_name"),
            api_key=service_config.get("api_key"),
            api_base=service_config.get("api_base"),
            api_version=service_config.get("api_version"),
            temperature=service_config.get("temperature"),
            max_tokens=service_config.get("max_tokens"),
            organization=service_config.get("organization"),
            cache_config=cache_config,
            timeouts=timeouts,
            retry_config=retry_config,
            validate_model=service_config.get("validate_model", False),
            strict_validation=service_config.get("strict_validation", False),
            model_selection=model_selection,
            connection_pool_size=service_config.get("connection_pool_size", 10),
            enable_metrics=service_config.get("enable_metrics", True),
            log_level=service_config.get("log_level", "INFO"),
            orchestrator_config=orchestrator_config,
            enable_provider_pooling=service_config.get("enable_provider_pooling", False),
            provider_pool_size=service_config.get("provider_pool_size", (1, 5)),
        )

    @classmethod
    def from_global_config(cls, config_instance: Optional[Any] = None) -> "LLMServiceConfig":
        """Create configuration from the global config.

        Args:
            config_instance: Optional Config instance (uses default if None)

        Returns:
            LLMServiceConfig instance
        """
        from enterprise_ai.config import config as default_config

        # Use provided config instance or default global config
        conf = config_instance or default_config

        # Extract LLM settings from global config
        llm_settings = conf.llm.get("default")

        # Create cache config from global settings
        cache_config = None
        if hasattr(conf, "cache_config"):
            cache_config = CacheConfig(
                use_cache=getattr(conf.cache_config, "use_cache", True),
                cache_type=getattr(conf.cache_config, "cache_type", "memory"),
                ttl=getattr(conf.cache_config, "ttl", 3600),
                max_size_mb=getattr(conf.cache_config, "max_size_mb", 500),
                cache_dir=getattr(conf.cache_config, "cache_dir", None),
                max_entries=getattr(conf.cache_config, "max_entries", 1000),
                promotion_policy=getattr(conf.cache_config, "promotion_policy", "both"),
                synchronize_writes=getattr(conf.cache_config, "synchronize_writes", False),
            )

        # Create timeouts config from global settings
        timeouts = None
        if hasattr(conf, "timeouts"):
            timeouts = RequestTimeouts(
                default_timeout=getattr(conf.timeouts, "default_timeout", 60.0),
                connect_timeout=getattr(conf.timeouts, "connect_timeout", None),
                read_timeout=getattr(conf.timeouts, "read_timeout", None),
                streaming_timeout=getattr(conf.timeouts, "streaming_timeout", None),
            )

        # Create model selection from global settings
        model_selection = None
        if hasattr(conf, "model_selection"):
            model_selection = ModelSelectionStrategy(
                preferred_model=getattr(conf.model_selection, "preferred_model", ""),
                fallback_models=getattr(conf.model_selection, "fallback_models", None),
                auto_fallback=getattr(conf.model_selection, "auto_fallback", True),
                capability_requirements=getattr(
                    conf.model_selection, "capability_requirements", {}
                ),
                fallback_across_providers=getattr(
                    conf.model_selection, "fallback_across_providers", False
                ),
                provider_preferences=getattr(conf.model_selection, "provider_preferences", []),
            )

        # Create orchestrator config from global settings
        orchestrator_config = None
        if hasattr(conf, "orchestrator_config"):
            orchestrator_config = OrchestratorConfig(
                max_concurrent_requests=getattr(
                    conf.orchestrator_config, "max_concurrent_requests", 10
                ),
                max_queue_size=getattr(conf.orchestrator_config, "max_queue_size", 100),
                rate_limits=getattr(conf.orchestrator_config, "rate_limits", {}),
                priority_levels=getattr(conf.orchestrator_config, "priority_levels", 4),
                adaptive_scaling=getattr(conf.orchestrator_config, "adaptive_scaling", True),
                max_retries=getattr(conf.orchestrator_config, "max_retries", 3),
                enable_deduplication=getattr(
                    conf.orchestrator_config, "enable_deduplication", True
                ),
                enable_circuit_breaker=getattr(
                    conf.orchestrator_config, "enable_circuit_breaker", True
                ),
            )

        # Create service config
        return cls(
            provider_name=llm_settings.api_type if llm_settings else None,
            model_name=llm_settings.model if llm_settings else None,
            api_key=llm_settings.api_key if llm_settings else None,
            api_base=llm_settings.base_url if llm_settings else None,
            api_version=llm_settings.api_version if llm_settings else None,
            temperature=llm_settings.temperature if llm_settings else None,
            max_tokens=llm_settings.max_tokens if llm_settings else None,
            cache_config=cache_config,
            timeouts=timeouts,
            model_selection=model_selection,
            orchestrator_config=orchestrator_config,
        )

    def update(self, new_config: Union["LLMServiceConfig", Dict[str, Any]]) -> "LLMServiceConfig":
        """Update configuration with new values.

        Args:
            new_config: New configuration (LLMServiceConfig or dictionary)

        Returns:
            Self for chaining
        """
        if isinstance(new_config, dict):
            new_config = self.from_dict(new_config)

        # Update simple attributes
        for attr in [
            "provider_name",
            "model_name",
            "api_key",
            "api_base",
            "api_version",
            "temperature",
            "max_tokens",
            "organization",
            "config_path",
            "validate_model",
            "strict_validation",
            "connection_pool_size",
            "enable_metrics",
            "log_level",
            "enable_provider_pooling",
            "provider_pool_size",
        ]:
            new_value = getattr(new_config, attr)
            if new_value is not None:
                setattr(self, attr, new_value)

        # Update nested configurations
        if new_config.cache_config:
            self.cache_config.update(new_config.cache_config)

        if new_config.timeouts:
            self.timeouts.update(new_config.timeouts)

        if new_config.model_selection:
            if self.model_selection:
                self.model_selection.update(new_config.model_selection)
            else:
                self.model_selection = new_config.model_selection

        if new_config.orchestrator_config:
            self.orchestrator_config.update(new_config.orchestrator_config)

        # Retry config is replaced since it doesn't have an update method
        if new_config.retry_config:
            self.retry_config = new_config.retry_config

        return self

    @staticmethod
    def _extract_cache_config(cache_dict: Dict[str, Any]) -> Optional[CacheConfig]:
        """Extract cache configuration from dictionary.

        Args:
            cache_dict: Cache configuration dictionary

        Returns:
            CacheConfig instance or None if dictionary is empty
        """
        if not cache_dict:
            return None

        # Convert string path to Path object if present
        cache_dir = cache_dict.get("cache_dir")
        if cache_dir and isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        return CacheConfig(
            use_cache=cache_dict.get("use_cache", True),
            cache_type=cache_dict.get("cache_type", "memory"),
            ttl=cache_dict.get("ttl", 3600),
            max_size_mb=cache_dict.get("max_size_mb", 500),
            cache_dir=cache_dir,
            max_entries=cache_dict.get("max_entries", 1000),
            promotion_policy=cache_dict.get("promotion_policy", "both"),
            synchronize_writes=cache_dict.get("synchronize_writes", True),
            retention=cache_dict.get("retention"),
        )

    @staticmethod
    def _extract_timeout_config(timeout_dict: Dict[str, Any]) -> Optional[RequestTimeouts]:
        """Extract timeout configuration from dictionary.

        Args:
            timeout_dict: Timeout configuration dictionary

        Returns:
            RequestTimeouts instance or None if dictionary is empty
        """
        if not timeout_dict:
            return None

        return RequestTimeouts(
            default_timeout=timeout_dict.get("default_timeout", 60.0),
            connect_timeout=timeout_dict.get("connect_timeout"),
            read_timeout=timeout_dict.get("read_timeout"),
            streaming_timeout=timeout_dict.get("streaming_timeout"),
            async_timeout=timeout_dict.get("async_timeout"),
        )

    @staticmethod
    def _extract_model_selection(model_dict: Dict[str, Any]) -> Optional[ModelSelectionStrategy]:
        """Extract model selection strategy from dictionary.

        Args:
            model_dict: Model selection configuration dictionary

        Returns:
            ModelSelectionStrategy instance or None if dictionary is empty
        """
        if not model_dict:
            return None

        return ModelSelectionStrategy(
            preferred_model=model_dict.get("preferred_model", ""),
            fallback_models=model_dict.get("fallback_models"),
            auto_fallback=model_dict.get("auto_fallback", True),
            capability_requirements=model_dict.get("capability_requirements", {}),
            max_cost_tier=model_dict.get("max_cost_tier"),
            fallback_across_providers=model_dict.get("fallback_across_providers", False),
            provider_preferences=model_dict.get("provider_preferences", []),
        )

    @staticmethod
    def _extract_orchestrator_config(orch_dict: Dict[str, Any]) -> Optional[OrchestratorConfig]:
        """Extract orchestrator configuration from dictionary.

        Args:
            orch_dict: Orchestrator configuration dictionary

        Returns:
            OrchestratorConfig instance or None if dictionary is empty
        """
        if not orch_dict:
            return None

        return OrchestratorConfig(
            max_concurrent_requests=orch_dict.get("max_concurrent_requests", 10),
            max_queue_size=orch_dict.get("max_queue_size", 100),
            rate_limits=orch_dict.get("rate_limits", {}),
            priority_levels=orch_dict.get("priority_levels", 4),
            adaptive_scaling=orch_dict.get("adaptive_scaling", True),
            max_retries=orch_dict.get("max_retries", 3),
            enable_deduplication=orch_dict.get("enable_deduplication", True),
            deduplication_ttl=orch_dict.get("deduplication_ttl", 5.0),
            enable_circuit_breaker=orch_dict.get("enable_circuit_breaker", True),
            circuit_breaker_threshold=orch_dict.get("circuit_breaker_threshold", 5),
            circuit_breaker_timeout=orch_dict.get("circuit_breaker_timeout", 30.0),
        )

    @staticmethod
    def _extract_retry_config(retry_dict: Dict[str, Any]) -> Optional[RetryConfig]:
        """Extract retry configuration from dictionary."""
        if not retry_dict:
            return None

        # Import here to avoid circular imports
        from enterprise_ai.llm.retry import RetryConfig, BackoffStrategy

        # Convert string backoff strategy to enum if needed
        backoff_str = retry_dict.get("backoff_strategy")
        # Initialize with a default value instead of None
        backoff_strategy = BackoffStrategy.EXPONENTIAL  # Default to EXPONENTIAL
        if backoff_str and isinstance(backoff_str, str):
            try:
                backoff_strategy = BackoffStrategy[backoff_str.upper()]
            except KeyError:
                pass  # Keep the default value if invalid

        return RetryConfig(
            max_retries=retry_dict.get("max_retries", 3),
            initial_delay=retry_dict.get("initial_delay", 1.0),
            max_delay=retry_dict.get("max_delay", 60.0),
            backoff_strategy=backoff_strategy,  # Now always a valid BackoffStrategy
            jitter_factor=retry_dict.get("jitter_factor", 0.25),
            retry_on_timeout=retry_dict.get("retry_on_timeout", True),
            retry_on_connection_error=retry_dict.get("retry_on_connection_error", True),
            retry_on_server_error=retry_dict.get("retry_on_server_error", True),
            retry_on_rate_limit=retry_dict.get("retry_on_rate_limit", True),
        )
