"""
Tests for the configuration management system.

These tests verify the configuration loading, validation, and access functionality,
including support for both TOML and YAML file formats and custom paths.
"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Ensure we import the actual implementation
from enterprise_ai.config import (
    Config,
    LLMSettings,
    SandboxSettings,
    TeamSettings,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    DEFAULT_CONFIGS_DIR,
)


# --- Fixtures ---


@pytest.fixture
def reset_config():
    """Reset the Config singleton instance between tests."""
    # Save the original instance
    original_instance = Config._instance

    # Reset the instance for testing
    Config._instance = None
    Config._initialized = False

    yield

    # Restore the original instance after the test
    Config._instance = original_instance
    Config._initialized = True


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def toml_config_file(temp_config_dir):
    """Create a temporary TOML config file."""
    config_content = """
[llm]
model = "test-model-toml"
base_url = "https://test.api"
api_key = "test-key-toml"
api_type = "test-type"

[sandbox]
use_sandbox = false
timeout = 120
"""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def yaml_config_file(temp_config_dir):
    """Create a temporary YAML config file."""
    config_content = """
llm:
  model: "test-model-yaml"
  base_url: "https://test.yaml.api"
  api_key: "test-key-yaml"
  api_type: "test-yaml-type"

sandbox:
  use_sandbox: false
  timeout: 180
"""
    config_file = temp_config_dir / "config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def custom_named_config_file(temp_config_dir):
    """Create a temporary config file with a custom name."""
    config_content = """
llm:
  model: "custom-named-config"
  base_url: "https://custom.api"
  api_key: "custom-key"
  api_type: "custom-type"
"""
    config_file = temp_config_dir / "enterprise_settings.yaml"
    config_file.write_text(config_content)
    return config_file


# --- Tests ---


def test_project_root():
    """Test that PROJECT_ROOT points to the correct directory."""
    # Check that the project root name is correctly identified
    assert PROJECT_ROOT.name == "Enterprise-AI" or PROJECT_ROOT.parent.name == "Enterprise-AI"

    # Check that the enterprise_ai package directory exists in the project root
    assert (PROJECT_ROOT / "enterprise_ai").exists()


def test_workspace_root_creation():
    """Test that WORKSPACE_ROOT is created if it doesn't exist."""
    # Temporarily modify WORKSPACE_ROOT for testing
    with patch("enterprise_ai.config.WORKSPACE_ROOT", new=Path("/tmp/test_workspace")):
        # Create a config instance which should create the workspace directory
        with patch("enterprise_ai.config.Config._load_config", return_value={}):
            test_config = Config()

            # Check that the workspace root was created
            assert test_config.workspace_root.exists()

            # Clean up
            test_config.workspace_root.rmdir()


def test_singleton_pattern(reset_config):
    """Test that Config implements the singleton pattern correctly."""
    config1 = Config()
    config2 = Config()

    # Both instances should be the same object
    assert config1 is config2

    # Even with different parameters, it should return the same instance
    config3 = Config(config_format="toml")
    assert config1 is config3


def test_direct_config_path(reset_config, toml_config_file):
    """Test loading configuration from a specific file path."""
    # Create config with specific path
    test_config = Config(config_path=str(toml_config_file))

    # Check that values were loaded correctly
    assert test_config.llm["default"].model == "test-model-toml"
    assert test_config.llm["default"].api_key == "test-key-toml"
    assert test_config.sandbox.use_sandbox is False
    assert test_config.sandbox.timeout == 120


def test_config_from_directory(reset_config, temp_config_dir, toml_config_file):
    """Test loading configuration from a specified directory."""
    # Create config with specific directory
    test_config = Config(config_dir=str(temp_config_dir))

    # Check that values were loaded correctly
    assert test_config.llm["default"].model == "test-model-toml"
    assert test_config.llm["default"].api_key == "test-key-toml"


def test_config_format_preference(reset_config, temp_config_dir):
    """Test that format preference is respected when both formats are available."""
    # Create separate config files to test format preference
    yaml_file = temp_config_dir / "config.yaml"
    yaml_file.write_text(
        "llm:\n  model: 'yaml-preference-model'\n  api_type: 'test'\n  base_url: 'url'\n  api_key: 'key'"
    )

    toml_file = temp_config_dir / "config.toml"
    toml_file.write_text(
        "[llm]\nmodel = 'toml-preference-model'\napi_type = 'test'\nbase_url = 'url'\napi_key = 'key'"
    )

    # Test YAML format preference by creating a test directory with both formats
    test_dir = temp_config_dir / "both_formats"
    test_dir.mkdir()
    (test_dir / "config.yaml").write_text(yaml_file.read_text())
    (test_dir / "config.toml").write_text(toml_file.read_text())

    # First test with YAML preference
    Config._instance = None  # Force reset before test
    Config._initialized = False
    yaml_config = Config(config_dir=str(test_dir), config_format="yaml")

    # Then test with TOML preference (need complete reset of singleton)
    Config._instance = None  # Force reset before second test
    Config._initialized = False
    toml_config = Config(config_dir=str(test_dir), config_format="toml")

    # Verify correct formats were loaded
    assert "yaml-preference-model" in yaml_config.llm["default"].model
    assert "toml-preference-model" in toml_config.llm["default"].model


def test_custom_named_config(reset_config, custom_named_config_file):
    """Test loading configuration from a file with a custom name."""
    # Create config with specific path to custom-named file
    test_config = Config(config_path=str(custom_named_config_file))

    # Check that values were loaded correctly
    assert test_config.llm["default"].model == "custom-named-config"
    assert test_config.llm["default"].api_key == "custom-key"


def test_default_config_creation(reset_config, temp_config_dir):
    """Test that a default config is created if none exists."""
    # Create a config with a directory that has no config files
    empty_dir = temp_config_dir / "empty"
    empty_dir.mkdir()

    # Mock the DEFAULT_CONFIGS_DIR to use our fixtures
    default_dir = temp_config_dir / "defaults"
    default_dir.mkdir()
    example_file = default_dir / "config.example.yaml"
    example_file.write_text(
        "llm:\n  model: 'example-model'\n  api_type: 'test'\n  base_url: 'url'\n  api_key: 'key'"
    )

    with patch("enterprise_ai.config.DEFAULT_CONFIGS_DIR", new=default_dir):
        test_config = Config(config_dir=str(empty_dir))

        # Check that a config file was created in the empty directory
        assert (empty_dir / "config.yaml").exists()

        # And that it has the default values
        assert test_config.llm["default"].model == "example-model"


def test_env_var_override(reset_config):
    """Test that environment variables can override config values."""
    # Set environment variable
    os.environ["OPENAI_API_KEY"] = "env-api-key"

    # Create a config instance with a minimal mock config
    with patch("enterprise_ai.config.Config._load_config", return_value={"llm": {}}):
        test_config = Config()

    # Check that env var was used
    assert test_config.llm["default"].api_key == "env-api-key"

    # Clean up
    del os.environ["OPENAI_API_KEY"]


def test_default_values(reset_config):
    """Test that default values are set correctly when config is missing."""
    # Create a config instance with an empty mock config
    with patch("enterprise_ai.config.Config._load_config", return_value={}):
        test_config = Config()

    # Check that defaults were applied
    assert test_config.llm["default"].model == "gpt-4-1106-preview"
    assert test_config.llm["default"].max_tokens == 4096
    assert test_config.llm["default"].temperature == 0.7
    assert test_config.sandbox.use_sandbox is True
    assert test_config.team_config.max_team_size == 10
    assert "manager" in test_config.team_config.default_roles


def test_model_specific_overrides(reset_config):
    """Test that model-specific overrides are applied correctly."""
    # Create mock config with model-specific settings
    mock_config = {
        "llm": {
            "model": "base-model",
            "temperature": 0.9,
            "api_type": "openai",
            "base_url": "https://base.api",
            "api_key": "base-key",
            "custom_model": {"model": "custom-model", "temperature": 0.5},
        }
    }

    # Create a config instance with the mock config
    with patch("enterprise_ai.config.Config._load_config", return_value=mock_config):
        test_config = Config()

    # Check base model settings
    assert test_config.llm["default"].model == "base-model"
    assert test_config.llm["default"].temperature == 0.9

    # Check custom model settings (should inherit from base but override specifics)
    assert test_config.llm["custom_model"].model == "custom-model"
    assert test_config.llm["custom_model"].temperature == 0.5
    assert test_config.llm["custom_model"].api_type == "openai"  # Inherited
    assert test_config.llm["custom_model"].base_url == "https://base.api"  # Inherited


def test_config_reload(reset_config):
    """Test that configuration can be reloaded at runtime."""
    # First load with initial config
    with patch(
        "enterprise_ai.config.Config._load_config",
        return_value={
            "llm": {
                "model": "initial-model",
                "api_type": "test",
                "base_url": "url",
                "api_key": "key",
            }
        },
    ):
        test_config = Config()
        assert test_config.llm["default"].model == "initial-model"

    # Reload with updated config
    with patch(
        "enterprise_ai.config.Config._load_config",
        return_value={
            "llm": {
                "model": "updated-model",
                "api_type": "test",
                "base_url": "url",
                "api_key": "key",
            }
        },
    ):
        test_config.reload_config()
        assert test_config.llm["default"].model == "updated-model"


def test_error_handling(reset_config):
    """Test that errors during config loading are handled gracefully."""
    # Mock _load_config to raise an exception
    with patch("enterprise_ai.config.Config._load_config", side_effect=Exception("Test error")):
        # Config should fall back to defaults
        test_config = Config()

        # Check that default values are used
        assert test_config.llm["default"].model == "gpt-4-1106-preview"
        assert test_config.sandbox.use_sandbox is True


def test_file_format_detection(reset_config, temp_config_dir):
    """Test that the correct file format is detected based on extension."""
    # Create real test files instead of mocking
    toml_file = temp_config_dir / "test.toml"
    toml_file.write_text(
        "[llm]\nmodel = 'test-model'\napi_type = 'test'\nbase_url = 'url'\napi_key = 'key'"
    )

    yaml_file = temp_config_dir / "test.yaml"
    yaml_file.write_text(
        "llm:\n  model: 'test-model'\n  api_type: 'test'\n  base_url: 'url'\n  api_key: 'key'"
    )

    # Create a Config instance for testing
    test_config = Config()

    # Test TOML loading
    try:
        result_toml = test_config._load_config_from_file(toml_file)
        assert result_toml.get("llm", {}).get("model") == "test-model"
        assert "toml successfully detected"  # This will be displayed if the test passes
    except ImportError:
        pytest.skip("tomli not installed")

    # Test YAML loading
    try:
        result_yaml = test_config._load_config_from_file(yaml_file)
        assert result_yaml.get("llm", {}).get("model") == "test-model"
        assert "yaml successfully detected"  # This will be displayed if the test passes
    except ImportError:
        pytest.skip("pyyaml not installed")


def test_nonexistent_config_path(reset_config, temp_config_dir):
    """Test behavior when specified config path doesn't exist."""
    nonexistent_path = temp_config_dir / "nonexistent.toml"

    # Test that a warning is printed when nonexistent path is specified
    with patch("builtins.print") as mock_print:
        # This will cause warning but continue with default config creation
        Config(config_path=str(nonexistent_path))

        # Verify a warning was printed
        assert mock_print.called
        # Check warning message content (partial match is safer)
        warning_msg = mock_print.call_args[0][0]
        assert "Warning" in warning_msg
        assert str(nonexistent_path) in warning_msg


def test_nonexistent_config_dir(reset_config):
    """Test behavior when specified config directory doesn't exist."""
    nonexistent_dir = Path("/tmp/nonexistent_dir_for_testing")

    # Mock to prevent actually creating directories
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        # Should try to create the directory
        Config(config_dir=str(nonexistent_dir))
        assert mock_mkdir.called


def test_config_dir_priority(reset_config, temp_config_dir):
    """Test that config_dir is used only if config_path is not specified."""
    # Create both a specific file and a directory with config
    specific_file = temp_config_dir / "specific.yaml"
    specific_file.write_text(
        "llm:\n  model: 'specific-model'\n  api_type: 'test'\n  base_url: 'url'\n  api_key: 'key'"
    )

    config_dir = temp_config_dir / "config_dir"
    config_dir.mkdir()
    dir_file = config_dir / "config.yaml"
    dir_file.write_text(
        "llm:\n  model: 'dir-model'\n  api_type: 'test'\n  base_url: 'url'\n  api_key: 'key'"
    )

    # When both are specified, config_path should take priority
    with patch.object(Config, "_load_config_from_file") as mock_load:
        mock_load.return_value = {
            "llm": {
                "model": "mocked-model",
                "api_type": "test",
                "base_url": "url",
                "api_key": "key",
            }
        }
        _ = Config(config_path=str(specific_file), config_dir=str(config_dir))
        # It should have tried to load the specific file
        assert str(specific_file) in str(mock_load.call_args)
