"""
Configuration management system for Enterprise AI.

This module provides a central configuration system for all components,
including LLM settings, sandbox environments, workspace paths, and more.
The configuration can be loaded from files (TOML or YAML) or environment variables.
"""

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, cast

# Import constants directly from constants.py
from enterprise_ai.constants import (
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    DEFAULT_CONFIGS_DIR,
    DEFAULT_CONFIG_TOML,
    DEFAULT_CONFIG_YAML,
)

# For type checking
if TYPE_CHECKING:
    import tomli
    import yaml

# Runtime imports
try:
    import tomli
except ImportError:
    tomli = None  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from pydantic import BaseModel, Field, field_validator


class LLMSettings(BaseModel):
    """Configuration for Language Model interactions."""

    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None, description="Maximum input tokens to use across all requests (None for unlimited)"
    )
    temperature: float = Field(0.7, description="Sampling temperature")
    api_type: str = Field(..., description="OpenAI, Azure, Anthropic, etc.")
    api_version: Optional[str] = Field(None, description="API version if applicable")


class SandboxSettings(BaseModel):
    """Configuration for execution sandboxes."""

    use_sandbox: bool = Field(True, description="Whether to use the sandbox")
    image: str = Field("python:3.12-slim", description="Base image")
    work_dir: str = Field("/workspace", description="Container working directory")
    memory_limit: str = Field("512m", description="Memory limit")
    cpu_limit: float = Field(1.0, description="CPU limit")
    timeout: int = Field(300, description="Default command timeout (seconds)")
    network_enabled: bool = Field(False, description="Whether network access is allowed")


class TeamSettings(BaseModel):
    """Configuration for team hierarchies and collaboration."""

    max_team_size: int = Field(10, description="Maximum number of agents in a team")
    default_roles: List[str] = Field(
        default_factory=lambda: ["manager", "developer", "researcher", "analyst"],
        description="Default available roles in teams",
    )
    role_templates_dir: Optional[Path] = Field(None, description="Directory for role templates")

    @field_validator("role_templates_dir", mode="before")
    @classmethod
    def set_default_templates_dir(cls, v: Any) -> Path:
        """Set default templates directory if not provided."""
        if v is None:
            return Path(__file__).parent / "team" / "templates"
        # Ensure return value is a Path
        if isinstance(v, Path):
            return v
        return Path(str(v))


class OllamaConfig(BaseModel):
    """Configuration specific to Ollama provider."""

    auto_pull: bool = Field(
        True, description="Whether to automatically pull models if not available"
    )
    timeout: float = Field(900.0, description="Timeout for Ollama operations in seconds")
    fallback_model: str = Field(
        "llama3", description="Fallback model if requested model is unavailable"
    )
    model_cache_size: int = Field(3, description="Maximum number of models to keep loaded")
    connection_pool_size: int = Field(
        10, description="Size of the connection pool for HTTP requests"
    )
    keep_alive: bool = Field(True, description="Whether to keep models loaded in memory")
    strict_validation: bool = Field(
        False, description="Whether to raise an exception if model validation fails"
    )
    host: str = Field("localhost", description="Ollama server host")
    port: int = Field(11434, description="Ollama server port")
    secure: bool = Field(False, description="Whether to use HTTPS for connections")


class AppConfig(BaseModel):
    """Main application configuration."""

    llm: Dict[str, LLMSettings] = Field(..., description="LLM configurations")
    llm_service: Optional[Any] = Field(None, description="LLM service configuration")
    cache_config: Optional[Any] = Field(None, description="Cache configuration")
    timeouts: Optional[Any] = Field(None, description="Request timeout configuration")
    model_selection: Optional[Any] = Field(None, description="Model selection strategy")
    orchestrator_config: Optional[Any] = Field(
        None, description="Request orchestration configuration"
    )
    ollama_config: Optional[OllamaConfig] = Field(None, description="Ollama-specific configuration")
    sandbox: Optional[SandboxSettings] = Field(None, description="Sandbox configuration")
    browser_config: Optional[Any] = Field(None, description="Browser configuration")
    search_config: Optional[Any] = Field(None, description="Search configuration")
    team_config: Optional[TeamSettings] = Field(None, description="Team configuration")
    workspace_root: Path = Field(default=WORKSPACE_ROOT, description="Workspace root directory")

    @field_validator("workspace_root")
    @classmethod
    def ensure_workspace_exists(cls, v: Path) -> Path:
        """Ensure the workspace directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = {"arbitrary_types_allowed": True}


class Config:
    """Singleton configuration manager."""

    _instance: Optional["Config"] = None
    _lock = threading.Lock()
    _initialized: bool = False
    _config: Optional[AppConfig] = None
    _config_path: Optional[str] = None
    _config_dir: Optional[str] = None
    _config_format: Optional[str] = None
    _init_params: Dict[str, Optional[str]] = {}

    def __new__(
        cls,
        config_path: Optional[str] = None,
        config_dir: Optional[str] = None,
        config_format: Optional[str] = None,
    ) -> "Config":
        """Create a new Config instance, or return the existing one."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_params = {
                        "config_path": config_path,
                        "config_dir": config_dir,
                        "config_format": config_format,
                    }
        return cls._instance

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dir: Optional[str] = None,
        config_format: Optional[str] = None,
    ) -> None:
        """Initialize the configuration manager.

        Args:
            config_path: Direct path to configuration file (overrides other options)
            config_dir: Directory to search for configuration files
            config_format: Preferred format ('toml' or 'yaml')
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    # Use the parameters stored during __new__
                    self._config_path = self._init_params["config_path"]
                    self._config_dir = self._init_params["config_dir"]
                    self._config_format = self._init_params["config_format"]
                    self._load_initial_config()
                    self._initialized = True

    def _get_config_path(self) -> Path:
        """Find the appropriate configuration file.

        Returns:
            Path to the configuration file to use
        """
        # 1. If a specific config path is provided, use it
        if self._config_path:
            path = Path(self._config_path)
            if path.exists():
                return path
            else:
                print(f"Warning: Specified config path {path} does not exist.")

        # 2. Look in the specified config directory
        if self._config_dir:
            config_dir = Path(self._config_dir)
            if not config_dir.exists():
                print(f"Warning: Specified config directory {config_dir} does not exist.")
                config_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default config directory
            config_dir = PROJECT_ROOT / "config"
            config_dir.mkdir(parents=True, exist_ok=True)

        # 3. Look for existing config files in the config directory based on format preference
        extensions = []

        # Add extensions based on format preference
        if self._config_format == "toml":
            extensions = [".toml"]
        elif self._config_format in ["yaml", "yml"]:
            extensions = [".yaml", ".yml"]
        else:
            # No preference, try all formats
            extensions = [".toml", ".yaml", ".yml"]

        # Look for existing config files
        for ext in extensions:
            config_file = config_dir / f"config{ext}"
            if config_file.exists():
                return config_file

        # 4. No existing config found, copy from default_configs
        return self._create_default_config(config_dir)

    def _create_default_config(self, config_dir: Path) -> Path:
        """Create a default configuration file.

        Args:
            config_dir: Directory where the config file should be created

        Returns:
            Path to the created config file
        """
        # Handle format preference directly without intermediate variables
        use_toml = False

        # Check if we should use TOML
        if self._config_format == "toml":
            use_toml = True
        elif self._config_format not in ["yaml", "yml"] and tomli is not None and yaml is None:
            # Default to TOML if that's the only parser available
            use_toml = True

        # If no parsers are available, raise an error
        if not use_toml and yaml is None:
            raise ImportError("Neither PyYAML nor tomli is installed. Please install at least one.")
        if use_toml and tomli is None:
            raise ImportError(
                "tomli is required to parse TOML files. Install with 'pip install tomli'"
            )
        # Set file extension and content based on format decision
        if use_toml:
            ext = ".toml"
            source_file = DEFAULT_CONFIGS_DIR / "config.example.toml"
            fallback_content = DEFAULT_CONFIG_TOML
        else:
            ext = ".yaml"
            source_file = DEFAULT_CONFIGS_DIR / "config.example.yaml"
            fallback_content = DEFAULT_CONFIG_YAML

        # Path for the new config file
        target_file = config_dir / f"config{ext}"

        # Copy from default_configs if available, otherwise use inline templates
        if source_file.exists():
            shutil.copy(source_file, target_file)
        else:
            # Fall back to inline templates
            target_file.write_text(fallback_content)

        return target_file

    def _load_config_from_file(self, config_path: Path) -> Dict[Any, Any]:
        """Load configuration based on file extension.

        Args:
            config_path: Path to the configuration file

        Returns:
            Parsed configuration as dictionary
        """
        with open(config_path, "rb") as f:
            # Choose parser based on file extension
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required to parse YAML files. Install with 'pip install PyYAML'"
                    )
                content = f.read().decode("utf-8")
                result = yaml.safe_load(content)
                return cast(Dict[Any, Any], result or {})
            elif config_path.suffix.lower() == ".toml":
                if tomli is None:
                    raise ImportError(
                        "tomli is required to parse TOML files. Install with 'pip install tomli'"
                    )
                return cast(Dict[Any, Any], tomli.load(f))
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    def _load_config(self) -> Dict[Any, Any]:
        """Load configuration from file.

        Returns:
            Dictionary with configuration values
        """
        config_path = self._get_config_path()
        result = self._load_config_from_file(config_path)
        if not isinstance(result, dict):
            return {}
        return cast(Dict[Any, Any], result)

    def _load_initial_config(self) -> None:
        """Initialize configuration from file or defaults."""
        try:
            raw_config = self._load_config()

            # Process LLM configuration
            base_llm = raw_config.get("llm", {})
            llm_overrides = {
                k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
            }

            default_settings = {
                "model": base_llm.get("model", "gpt-4-1106-preview"),
                "base_url": base_llm.get("base_url", "https://api.openai.com/v1"),
                "api_key": base_llm.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
                "max_tokens": base_llm.get("max_tokens", 4096),
                "max_input_tokens": base_llm.get("max_input_tokens"),
                "temperature": base_llm.get("temperature", 0.7),
                "api_type": base_llm.get("api_type", "openai"),
                "api_version": base_llm.get("api_version", ""),
            }

            # Process LLM service configuration
            llm_service_config = raw_config.get("llm_service", {})
            llm_service_settings = None
            if llm_service_config:
                # Importing here to avoid circular imports
                from enterprise_ai.llm.service.config import LLMServiceConfig

                llm_service_settings = LLMServiceConfig(
                    provider_name=llm_service_config.get("default_provider", "ollama"),
                    model_name=llm_service_config.get("default_model", "llama3"),
                    api_key=llm_service_config.get("api_key"),
                    api_base=llm_service_config.get("api_base_url"),
                    api_version=llm_service_config.get("api_version"),
                    temperature=llm_service_config.get("temperature"),
                    max_tokens=llm_service_config.get("max_tokens"),
                    organization=llm_service_config.get("organization"),
                    config_path=None,
                    cache_config=None,  # Will be set after creating cache_settings
                    retry_config=None,
                    timeouts=None,  # Will be set after creating timeout_settings
                    validate_model=llm_service_config.get("validate_model", False),
                    strict_validation=llm_service_config.get("strict_validation", False),
                    model_selection=None,  # Will be set after creating model_selection_settings
                    connection_pool_size=llm_service_config.get("connection_pool_size", 20),
                    enable_metrics=llm_service_config.get("enable_metrics", True),
                    log_level=llm_service_config.get("log_level", "INFO"),
                    orchestrator_config=None,  # Will be set after creating orchestrator_settings
                    enable_provider_pooling=llm_service_config.get("enable_provider_pooling", True),
                    provider_pool_size=llm_service_config.get("provider_pool_size", (2, 5)),
                )

            # Process cache configuration
            cache_config = raw_config.get("cache", {})
            cache_settings = None
            if cache_config:
                # Import here to avoid circular imports
                from enterprise_ai.llm.service.config import CacheConfig

                # Create cache directory if specified
                cache_dir = None
                if "cache_dir" in cache_config and cache_config["cache_dir"]:
                    cache_dir = Path(cache_config["cache_dir"])
                    cache_dir.mkdir(parents=True, exist_ok=True)

                cache_settings = CacheConfig(
                    use_cache=cache_config.get("use_cache", True),
                    cache_type=cache_config.get("cache_type", "hybrid"),
                    ttl=cache_config.get("ttl", 86400),
                    max_size_mb=cache_config.get("max_size_mb", 500),
                    cache_dir=cache_dir,
                    max_entries=cache_config.get("max_entries", 1000),
                    promotion_policy=cache_config.get("promotion_policy", "both"),
                    synchronize_writes=cache_config.get("synchronize_writes", False),
                )

            # Process timeout configuration
            timeout_config = raw_config.get("timeouts", {})
            timeout_settings = None
            if timeout_config:
                # Import here to avoid circular imports
                from enterprise_ai.llm.service.config import RequestTimeouts

                timeout_settings = RequestTimeouts(
                    default_timeout=timeout_config.get("default_timeout", 60.0),
                    connect_timeout=timeout_config.get("connect_timeout", 30.0),
                    read_timeout=timeout_config.get("read_timeout", 90.0),
                    streaming_timeout=timeout_config.get("streaming_timeout", 300.0),
                    async_timeout=timeout_config.get("async_timeout", 60.0),
                )

            # Process model selection configuration
            model_selection_config = raw_config.get("model_selection", {})
            model_selection_settings = None
            if model_selection_config:
                # Import here to avoid circular imports
                from enterprise_ai.llm.service.config import ModelSelectionStrategy

                model_selection_settings = ModelSelectionStrategy(
                    preferred_model=model_selection_config.get("preferred_model", ""),
                    fallback_models=model_selection_config.get("fallback_models"),
                    auto_fallback=model_selection_config.get("auto_fallback", True),
                    fallback_across_providers=model_selection_config.get(
                        "fallback_across_providers", True
                    ),
                    provider_preferences=model_selection_config.get(
                        "provider_preferences", ["openai", "anthropic", "ollama"]
                    ),
                    capability_requirements=model_selection_config.get(
                        "capability_requirements", {}
                    ),
                    max_cost_tier=model_selection_config.get("max_cost_tier"),
                )

            # Process orchestrator configuration
            orchestrator_config = raw_config.get("orchestrator", {})
            orchestrator_settings = None
            if orchestrator_config:
                # Import here to avoid circular imports
                from enterprise_ai.llm.service.config import OrchestratorConfig

                orchestrator_settings = OrchestratorConfig(
                    max_concurrent_requests=orchestrator_config.get("max_concurrent_requests", 20),
                    max_queue_size=orchestrator_config.get("max_queue_size", 100),
                    rate_limits=orchestrator_config.get("rate_limits", {}),
                    priority_levels=orchestrator_config.get("priority_levels", 4),
                    adaptive_scaling=orchestrator_config.get("adaptive_scaling", True),
                    max_retries=orchestrator_config.get("max_retries", 3),
                    enable_deduplication=orchestrator_config.get("enable_deduplication", True),
                    deduplication_ttl=orchestrator_config.get("deduplication_ttl", 5.0),
                    enable_circuit_breaker=orchestrator_config.get("enable_circuit_breaker", True),
                    circuit_breaker_threshold=orchestrator_config.get(
                        "circuit_breaker_threshold", 5
                    ),
                    circuit_breaker_timeout=orchestrator_config.get(
                        "circuit_breaker_reset_timeout", 300
                    ),
                )

            # Process Ollama-specific configuration
            ollama_config = raw_config.get("ollama", {})
            ollama_settings = None
            if ollama_config:
                ollama_settings = OllamaConfig(
                    auto_pull=ollama_config.get("auto_pull", True),
                    timeout=ollama_config.get("timeout", 900.0),
                    fallback_model=ollama_config.get("fallback_model", "llama3"),
                    model_cache_size=ollama_config.get("model_cache_size", 3),
                    connection_pool_size=ollama_config.get("connection_pool_size", 10),
                    keep_alive=ollama_config.get("keep_alive", True),
                    strict_validation=ollama_config.get("strict_validation", False),
                    host=ollama_config.get("host", "localhost"),
                    port=ollama_config.get("port", 11434),
                    secure=ollama_config.get("secure", False),
                )

            # Process sandbox configuration
            sandbox_config = raw_config.get("sandbox", {})
            sandbox_settings = None
            if sandbox_config:
                sandbox_settings = SandboxSettings(
                    use_sandbox=sandbox_config.get("use_sandbox", True),
                    image=sandbox_config.get("image", "python:3.12-slim"),
                    work_dir=sandbox_config.get("work_dir", "/workspace"),
                    memory_limit=sandbox_config.get("memory_limit", "512m"),
                    cpu_limit=sandbox_config.get("cpu_limit", 1.0),
                    timeout=sandbox_config.get("timeout", 300),
                    network_enabled=sandbox_config.get("network_enabled", False),
                )
            else:
                # Explicitly use default values
                sandbox_settings = SandboxSettings(
                    use_sandbox=True,
                    image="python:3.12-slim",
                    work_dir="/workspace",
                    memory_limit="512m",
                    cpu_limit=1.0,
                    timeout=300,
                    network_enabled=False,
                )

            # Process team configuration
            team_config = raw_config.get("team", {})
            team_settings = None
            if team_config:
                # Get role templates directory
                role_templates_dir = None
                if "role_templates_dir" in team_config:
                    templates_path = team_config["role_templates_dir"]
                    if templates_path:
                        role_templates_dir = Path(templates_path)

                team_settings = TeamSettings(
                    max_team_size=team_config.get("max_team_size", 10),
                    default_roles=team_config.get(
                        "default_roles", ["manager", "developer", "researcher", "analyst"]
                    ),
                    role_templates_dir=role_templates_dir,
                )
            else:
                # Explicitly use default values
                team_settings = TeamSettings(
                    max_team_size=10,
                    default_roles=["manager", "developer", "researcher", "analyst"],
                    role_templates_dir=None,
                )

            # Update connections between objects
            if llm_service_settings:
                llm_service_settings.cache_config = (
                    cache_settings or CacheConfig()
                )  # Provide default if None
                llm_service_settings.timeouts = (
                    timeout_settings or RequestTimeouts()
                )  # Provide default if None
                llm_service_settings.model_selection = (
                    model_selection_settings or ModelSelectionStrategy(preferred_model="")
                )  # Provide default if None
                llm_service_settings.orchestrator_config = (
                    orchestrator_settings or OrchestratorConfig()
                )  # Provide default if None

            # Build final configuration
            llm_dict = {}
            llm_dict["default"] = LLMSettings(**default_settings)
            for name, override_config in llm_overrides.items():
                config = {**default_settings, **override_config}
                llm_dict[name] = LLMSettings(**config)

            config_dict = {
                "llm": llm_dict,
                "llm_service": llm_service_settings,
                "cache_config": cache_settings,
                "timeouts": timeout_settings,
                "model_selection": model_selection_settings,
                "orchestrator_config": orchestrator_settings,
                "ollama_config": ollama_settings,
                "sandbox": sandbox_settings,
                "browser_config": None,  # Not processed in this example
                "search_config": None,  # Not processed in this example
                "team_config": team_settings,
                "workspace_root": Path(raw_config.get("workspace_root", WORKSPACE_ROOT)),
            }

            self._config = AppConfig(**config_dict)  # type: ignore

        except Exception as e:
            print(f"Error loading configuration: {e}. Using defaults.")
            # Create minimal default configuration with new sections
            # Import here to avoid circular imports
            from enterprise_ai.llm.service.config import (
                LLMServiceConfig,
                CacheConfig,
                RequestTimeouts,
                ModelSelectionStrategy,
                OrchestratorConfig,
            )

            self._config = AppConfig(
                llm={
                    "default": LLMSettings(
                        model="gpt-4-1106-preview",
                        base_url="https://api.openai.com/v1",
                        api_key=os.environ.get("OPENAI_API_KEY", ""),
                        api_type="openai",
                        max_tokens=4096,
                        temperature=0.7,
                        api_version=None,
                        max_input_tokens=None,
                    )
                },
                llm_service=LLMServiceConfig(
                    provider_name="ollama",
                    model_name="llama3",
                    api_key=None,
                    api_base=None,
                    api_version=None,
                    temperature=None,
                    max_tokens=None,
                    organization=None,
                    config_path=None,
                    cache_config=None,
                    retry_config=None,
                    timeouts=None,
                    validate_model=False,
                    strict_validation=False,
                    model_selection=None,
                    connection_pool_size=20,
                    enable_metrics=True,
                    log_level="INFO",
                    orchestrator_config=None,
                    enable_provider_pooling=True,
                    provider_pool_size=(2, 5),
                ),
                cache_config=CacheConfig(
                    use_cache=True,
                    cache_type="hybrid",
                    ttl=86400,
                    max_size_mb=500,
                    max_entries=1000,
                    cache_dir=None,
                    promotion_policy="both",
                    synchronize_writes=False,
                ),
                timeouts=RequestTimeouts(
                    default_timeout=60.0,
                    streaming_timeout=300.0,
                    connect_timeout=30.0,
                    read_timeout=90.0,
                ),
                model_selection=ModelSelectionStrategy(
                    preferred_model="",
                    fallback_models=None,
                    auto_fallback=True,
                    fallback_across_providers=True,
                    provider_preferences=["ollama", "openai", "anthropic"],
                    capability_requirements={},
                    max_cost_tier=None,
                ),
                orchestrator_config=OrchestratorConfig(
                    max_concurrent_requests=20,
                    max_queue_size=100,
                    adaptive_scaling=True,
                    max_retries=3,
                    enable_deduplication=True,
                    deduplication_ttl=5.0,
                    enable_circuit_breaker=True,
                    circuit_breaker_threshold=5,
                    circuit_breaker_timeout=300,
                    priority_levels=4,
                    rate_limits={},
                ),
                ollama_config=OllamaConfig(
                    auto_pull=True,
                    timeout=900.0,
                    fallback_model="llama3",
                    model_cache_size=3,
                    connection_pool_size=10,
                    keep_alive=True,
                    strict_validation=False,
                    host="localhost",
                    port=11434,
                    secure=False,
                ),
                sandbox=SandboxSettings(
                    use_sandbox=True,
                    image="python:3.12-slim",
                    work_dir="/workspace",
                    memory_limit="512m",
                    cpu_limit=1.0,
                    timeout=300,
                    network_enabled=False,
                ),
                team_config=TeamSettings(
                    max_team_size=10,
                    default_roles=["manager", "developer", "researcher", "analyst"],
                    role_templates_dir=None,
                ),
                browser_config=None,
                search_config=None,
            )

    def reload_config(self) -> None:
        """Reload configuration from file."""
        with self._lock:
            self._load_initial_config()

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        """Get LLM settings."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        return self._config.llm

    @property
    def llm_service(self) -> Any:
        """Get LLM service configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.llm_service is None:
            # Import here to avoid circular imports
            from enterprise_ai.llm.service.config import LLMServiceConfig

            # Create a default LLMServiceConfig with all required parameters
            self._config.llm_service = LLMServiceConfig(
                provider_name="ollama",
                model_name="llama3",
                api_key=None,
                api_base=None,
                api_version=None,
                temperature=None,
                max_tokens=None,
                organization=None,
                config_path=None,
                cache_config=None,
                retry_config=None,
                timeouts=None,
                validate_model=False,
                strict_validation=False,
                model_selection=None,
                connection_pool_size=20,
                enable_metrics=True,
                log_level="INFO",
                orchestrator_config=None,
                enable_provider_pooling=True,
                provider_pool_size=(2, 5),
            )
        return self._config.llm_service

    @property
    def cache_config(self) -> Any:
        """Get cache configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.cache_config is None:
            # Import here to avoid circular imports
            from enterprise_ai.llm.service.config import CacheConfig

            self._config.cache_config = CacheConfig(
                use_cache=True,
                cache_type="hybrid",
                ttl=86400,
                max_size_mb=500,
                max_entries=1000,
                cache_dir=None,
                promotion_policy="both",
                synchronize_writes=False,
                retention="7 days",
            )
        return self._config.cache_config

    @property
    def timeouts(self) -> Any:
        """Get request timeout configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.timeouts is None:
            # Import here to avoid circular imports
            from enterprise_ai.llm.service.config import RequestTimeouts

            self._config.timeouts = RequestTimeouts(
                default_timeout=60.0,
                streaming_timeout=300.0,
                connect_timeout=30.0,
                read_timeout=90.0,
            )
        return self._config.timeouts

    @property
    def model_selection(self) -> Any:
        """Get model selection strategy."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.model_selection is None:
            # Import here to avoid circular imports
            from enterprise_ai.llm.service.config import ModelSelectionStrategy

            self._config.model_selection = ModelSelectionStrategy(
                preferred_model="",
                fallback_models=None,
                auto_fallback=True,
                fallback_across_providers=True,
                provider_preferences=["ollama", "openai", "anthropic"],
                capability_requirements={},
                max_cost_tier=None,
            )
        return self._config.model_selection

    @property
    def orchestrator_config(self) -> Any:
        """Get request orchestration configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.orchestrator_config is None:
            # Import here to avoid circular imports
            from enterprise_ai.llm.service.config import OrchestratorConfig

            self._config.orchestrator_config = OrchestratorConfig(
                max_concurrent_requests=20,
                max_queue_size=100,
                adaptive_scaling=True,
                max_retries=3,
                enable_deduplication=True,
                deduplication_ttl=5.0,
                enable_circuit_breaker=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=300,
                priority_levels=4,
                rate_limits={},
            )
        return self._config.orchestrator_config

    @property
    def ollama_config(self) -> OllamaConfig:
        """Get Ollama-specific configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.ollama_config is None:
            self._config.ollama_config = OllamaConfig(
                auto_pull=True,
                timeout=900.0,
                fallback_model="llama3",
                model_cache_size=3,
                connection_pool_size=10,
                keep_alive=True,
                strict_validation=False,
                host="localhost",
                port=11434,
                secure=False,
            )
        return self._config.ollama_config

    @property
    def sandbox(self) -> SandboxSettings:
        """Get sandbox settings."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.sandbox is None:
            self._config.sandbox = SandboxSettings(
                use_sandbox=True,
                image="python:3.12-slim",
                work_dir="/workspace",
                memory_limit="512m",
                cpu_limit=1.0,
                timeout=300,
                network_enabled=False,
            )
        return self._config.sandbox

    @property
    def browser_config(self) -> Optional[Any]:
        """Get browser settings."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[Any]:
        """Get search settings."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        return self._config.search_config

    @property
    def team_config(self) -> TeamSettings:
        """Get team settings."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        if self._config.team_config is None:
            self._config.team_config = TeamSettings(
                max_team_size=10,
                default_roles=["manager", "developer", "researcher", "analyst"],
                role_templates_dir=None,
            )
        return self._config.team_config

    @property
    def workspace_root(self) -> Path:
        """Get the workspace root directory."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        return self._config.workspace_root

    @property
    def root_path(self) -> Path:
        """Get the root path of the application."""
        return PROJECT_ROOT

    def create_llm_service_from_config(self) -> Any:
        """Create an LLMService instance from the loaded configuration.

        Returns:
            LLMService instance
        """
        # Import here to avoid circular imports
        from enterprise_ai.llm.service import LLMService

        # Use the existing configuration but ensure it's the proper type
        service_config = self.llm_service

        # Set connections for any missing parts using local variables to handle None case
        cache_config = (
            self.cache_config
            if service_config.cache_config is None
            else service_config.cache_config
        )
        timeouts = self.timeouts if service_config.timeouts is None else service_config.timeouts
        model_selection = (
            self.model_selection
            if service_config.model_selection is None
            else service_config.model_selection
        )
        orchestrator_config = (
            self.orchestrator_config
            if service_config.orchestrator_config is None
            else service_config.orchestrator_config
        )

        # Create a new LLMServiceConfig with all fields properly filled
        service_config = service_config.__class__(
            provider_name=service_config.provider_name or self.llm["default"].api_type,
            model_name=service_config.model_name or self.llm["default"].model,
            api_key=service_config.api_key,
            api_base=service_config.api_base,
            api_version=service_config.api_version,
            temperature=service_config.temperature,
            max_tokens=service_config.max_tokens,
            organization=service_config.organization,
            config_path=service_config.config_path,
            cache_config=cache_config,
            retry_config=service_config.retry_config,
            timeouts=timeouts,
            validate_model=service_config.validate_model,
            strict_validation=service_config.strict_validation,
            model_selection=model_selection,
            connection_pool_size=service_config.connection_pool_size,
            enable_metrics=service_config.enable_metrics,
            log_level=service_config.log_level,
            orchestrator_config=orchestrator_config,
            enable_provider_pooling=service_config.enable_provider_pooling,
            provider_pool_size=service_config.provider_pool_size,
        )

        # If provider_name/model_name not set, get from default LLM config
        if not service_config.provider_name:
            service_config.provider_name = self.llm["default"].api_type

        if not service_config.model_name:
            service_config.model_name = self.llm["default"].model

        # Create and return the service
        return LLMService(service_config)


# Global configuration instance
config = Config()
