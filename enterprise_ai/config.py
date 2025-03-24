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
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING, cast

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


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


# Global constants
PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
DEFAULT_CONFIGS_DIR = Path(__file__).parent / "default_configs"


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


class ProxySettings(BaseModel):
    """Configuration for proxy servers."""

    server: str = Field(..., description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    """Configuration for web search capabilities."""

    engine: str = Field(default="Google", description="Primary search engine")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=60, description="Seconds to wait before retrying all engines after they all fail"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of times to retry all engines when all fail"
    )


class BrowserSettings(BaseModel):
    """Configuration for browser automation."""

    headless: bool = Field(True, description="Whether to run browser in headless mode")
    disable_security: bool = Field(False, description="Disable browser security features")
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    proxy: Optional[ProxySettings] = Field(None, description="Proxy settings for the browser")
    max_content_length: int = Field(
        2000, description="Maximum length for content retrieval operations"
    )


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


class AppConfig(BaseModel):
    """Main application configuration."""

    llm: Dict[str, LLMSettings] = Field(..., description="LLM configurations")
    sandbox: Optional[SandboxSettings] = Field(None, description="Sandbox configuration")
    browser_config: Optional[BrowserSettings] = Field(None, description="Browser configuration")
    search_config: Optional[SearchSettings] = Field(None, description="Search configuration")
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
    _config_format: Optional[str] = None  # Changed from Literal to str
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

            # Process browser configuration
            browser_config = raw_config.get("browser", {})
            browser_settings = None

            if browser_config:
                # Handle proxy settings
                proxy_config = browser_config.get("proxy", {})
                proxy_settings = None

                if proxy_config and proxy_config.get("server"):
                    proxy_settings = ProxySettings(
                        **{
                            k: v
                            for k, v in proxy_config.items()
                            if k in ["server", "username", "password"] and v
                        }
                    )

                # Filter valid browser config parameters
                valid_browser_params = {
                    k: v
                    for k, v in browser_config.items()
                    if k in BrowserSettings.__annotations__ and v is not None
                }

                if proxy_settings:
                    valid_browser_params["proxy"] = proxy_settings

                if valid_browser_params:
                    browser_settings = BrowserSettings(**valid_browser_params)

            # Process search configuration
            search_config = raw_config.get("search", {})
            search_settings = None
            if search_config:
                search_settings = SearchSettings(**search_config)  # type: ignore

            # Process sandbox configuration
            sandbox_config = raw_config.get("sandbox", {})
            if sandbox_config:
                # Extract each field with its default if not present
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
                # Extract needed fields
                team_settings = TeamSettings(
                    max_team_size=team_config.get("max_team_size", 10),
                    default_roles=team_config.get(
                        "default_roles", ["manager", "developer", "researcher", "analyst"]
                    ),
                    role_templates_dir=team_config.get("role_templates_dir", None),
                )
            else:
                # Explicitly use default values
                team_settings = TeamSettings(
                    max_team_size=10,
                    default_roles=["manager", "developer", "researcher", "analyst"],
                    role_templates_dir=None,
                )

            # Build final configuration
            llm_dict = {}
            llm_dict["default"] = LLMSettings(**default_settings)
            for name, override_config in llm_overrides.items():
                config = {**default_settings, **override_config}
                llm_dict[name] = LLMSettings(**config)

            config_dict = {
                "llm": llm_dict,
                "sandbox": sandbox_settings,
                "browser_config": browser_settings,
                "search_config": search_settings,
                "team_config": team_settings,
                "workspace_root": Path(raw_config.get("workspace_root", WORKSPACE_ROOT)),
            }

            self._config = AppConfig(**config_dict)  # type: ignore
        except Exception as e:
            print(f"Error loading configuration: {e}. Using defaults.")
            # Create minimal default configuration
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
    def browser_config(self) -> Optional[BrowserSettings]:
        """Get browser settings."""
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
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


# Default configuration templates
DEFAULT_CONFIG_TOML = """# Enterprise AI Configuration Example (TOML)

# LLM Configuration
[llm]
model = "llama3.2"  # Changed default from gpt-4 to llama3
base_url = "http://localhost:11434"  # Changed default to Ollama URL
api_key = ""  # Not needed for Ollama
max_tokens = 4096
temperature = 0.7
api_type = "ollama"  # Changed default from openai to ollama
api_version = ""

# Example of provider-specific overrides
[llm.ollama]
model = "llama3.2"
auto_pull = true
temperature = 0.7

[llm.gpt4]
model = "gpt-4-1106-preview"
base_url = "https://api.openai.com/v1"
api_key = ""  # Set via environment variable OPENAI_API_KEY
temperature = 0.5
api_type = "openai"

[llm.claude]
model = "claude-3-opus-20240229"
base_url = "https://api.anthropic.com/v1"
api_key = ""  # Set via environment variable ANTHROPIC_API_KEY
api_type = "anthropic"

# Sandbox Configuration
[sandbox]
use_sandbox = true
image = "python:3.12-slim"
work_dir = "/workspace"
memory_limit = "512m"
cpu_limit = 1.0
timeout = 900
network_enabled = false

# Browser Configuration
[browser]
headless = true
disable_security = false
extra_chromium_args = []
max_content_length = 2000

# Search Configuration
[search]
engine = "Google"
fallback_engines = ["DuckDuckGo", "Bing"]
retry_delay = 60
max_retries = 3

# Team Configuration
[team]
max_team_size = 10
default_roles = ["manager", "developer", "researcher", "analyst"]

# Ollama-specific Configuration
[ollama]
auto_pull = true  # Automatically pull models if not available
timeout = 900.0  # 15 minutes timeout for model operations
fallback_model = "llama3"  # Fallback model if requested model is unavailable

# Paths
# workspace_root = "/custom/path"  # Uncomment to override default
"""

DEFAULT_CONFIG_YAML = """# Enterprise AI Configuration Example (YAML)

# LLM Configuration
llm:
  model: "llama3.2"  # Changed default from gpt-4 to llama3
  base_url: "http://localhost:11434"  # Changed default to Ollama URL
  api_key: ""  # Not needed for Ollama
  max_tokens: 4096
  temperature: 0.7
  api_type: "ollama"  # Changed default from openai to ollama
  api_version: ""

  # Example of provider-specific overrides
  ollama:
    model: "llama3.2"
    auto_pull: true
    temperature: 0.7

  gpt4:
    model: "gpt-4-1106-preview"
    base_url: "https://api.openai.com/v1"
    api_key: ""  # Set via environment variable OPENAI_API_KEY
    temperature: 0.5
    api_type: "openai"

  claude:
    model: "claude-3-opus-20240229"
    base_url: "https://api.anthropic.com/v1"
    api_key: ""  # Set via environment variable ANTHROPIC_API_KEY
    api_type: "anthropic"

# Sandbox Configuration
sandbox:
  use_sandbox: true
  image: "python:3.12-slim"
  work_dir: "/workspace"
  memory_limit: "512m"
  cpu_limit: 1.0
  timeout: 900
  network_enabled: false

# Browser Configuration
browser:
  headless: true
  disable_security: false
  extra_chromium_args: []
  max_content_length: 2000

# Search Configuration
search:
  engine: "Google"
  fallback_engines:
    - "DuckDuckGo"
    - "Bing"
  retry_delay: 60
  max_retries: 3

# Team Configuration
team:
  max_team_size: 10
  default_roles:
    - "manager"
    - "developer"
    - "researcher"
    - "analyst"

# Ollama-specific Configuration
ollama:
  auto_pull: true  # Automatically pull models if not available
  timeout: 900.0  # 15 minutes timeout for model operations
  fallback_model: "llama3"  # Fallback model if requested model is unavailable

# Paths
# workspace_root: "/custom/path"  # Uncomment to override default
"""

# Global configuration instance
config = Config()
