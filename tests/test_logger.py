"""
Tests for the Enterprise AI logging system.

This module contains tests for the logging system, including configuration,
log levels, contextual logging, and output destinations.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from enterprise_ai.logger import (
    EnterpriseLogger,
    LoggerConfig,
    configure,
    debug,
    error,
    get_agent_logger,
    get_logger,
    get_team_logger,
    info,
    trace_execution,
    warning,
    with_context,
)


@pytest.fixture
def reset_logger_singleton():
    """Fixture to reset the logger singleton between tests."""
    # Store original state
    original_instance = EnterpriseLogger._instance
    original_initialized = EnterpriseLogger._initialized
    original_context_var = (
        EnterpriseLogger._context_var.copy() if EnterpriseLogger._context_var else {}
    )

    # Reset state
    EnterpriseLogger._instance = None
    EnterpriseLogger._initialized = False
    EnterpriseLogger._context_var = {}

    yield

    # Restore original state
    EnterpriseLogger._instance = original_instance
    EnterpriseLogger._initialized = original_initialized
    EnterpriseLogger._context_var = original_context_var


class TestLoggerConfig:
    """Tests for the LoggerConfig class."""

    def test_default_config(self):
        """Test that default configuration works."""
        config = LoggerConfig()
        assert config.console_level == LoggerConfig.INFO
        assert config.file_level == LoggerConfig.DEBUG
        assert config.log_dir.exists()

    def test_custom_config(self):
        """Test that custom configuration works."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            config = LoggerConfig(
                console_level=LoggerConfig.WARNING,
                file_level=LoggerConfig.ERROR,
                log_dir=temp_dir,
                format="custom_format",
                retention="5 days",
                rotation="50 MB",
                enable_context=False,
            )

            assert config.console_level == LoggerConfig.WARNING
            assert config.file_level == LoggerConfig.ERROR
            assert config.log_dir == temp_dir
            assert config.format == "custom_format"
            assert config.retention == "5 days"
            assert config.rotation == "50 MB"
            assert config.enable_context is False
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestEnterpriseLogger:
    """Tests for the EnterpriseLogger class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, reset_logger_singleton):
        """Set up test environment."""
        # Create a temporary directory for log files
        self.temp_dir = Path(tempfile.mkdtemp())

        yield

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_singleton_pattern(self):
        """Test that EnterpriseLogger follows the singleton pattern."""
        logger1 = EnterpriseLogger()
        logger2 = EnterpriseLogger()

        assert logger1 is logger2

        # Test that reconfiguration doesn't create a new instance
        config = LoggerConfig(
            console_level=LoggerConfig.WARNING,
            file_level=LoggerConfig.ERROR,
            log_dir=self.temp_dir,
        )

        logger1.configure(config)
        logger3 = EnterpriseLogger()

        assert logger1 is logger3
        assert logger1._logger_config.console_level == LoggerConfig.WARNING

    def test_configure_logger(self):
        """Test that logger configuration works."""
        # Create a logger with custom configuration
        config = LoggerConfig(
            console_level=LoggerConfig.WARNING,
            file_level=LoggerConfig.ERROR,
            log_dir=self.temp_dir,
        )

        logger = EnterpriseLogger(config)

        # Check that configuration was applied
        assert logger._logger_config.console_level == LoggerConfig.WARNING
        assert logger._logger_config.file_level == LoggerConfig.ERROR
        assert logger._logger_config.log_dir == self.temp_dir

    def test_get_logger(self):
        """Test that get_logger returns a logger with the correct name."""
        logger = EnterpriseLogger(LoggerConfig(log_dir=self.temp_dir))

        component_logger = logger.get_logger("test_component")

        # Check that the logger has the correct context by checking the context property
        assert hasattr(component_logger, "context")
        assert "name" in component_logger.context
        assert component_logger.context["name"] == "test_component"

    def test_get_agent_logger(self):
        """Test that get_agent_logger returns a logger with agent context."""
        logger = EnterpriseLogger(LoggerConfig(log_dir=self.temp_dir))

        agent_logger = logger.get_agent_logger("agent-1", "developer")

        # Check that the logger has agent context by checking the context property
        assert hasattr(agent_logger, "context")
        assert "name" in agent_logger.context
        assert "agent_id" in agent_logger.context
        assert "agent_type" in agent_logger.context
        assert agent_logger.context["name"] == "agent"
        assert agent_logger.context["agent_id"] == "agent-1"
        assert agent_logger.context["agent_type"] == "developer"

    def test_get_team_logger(self):
        """Test that get_team_logger returns a logger with team context."""
        logger = EnterpriseLogger(LoggerConfig(log_dir=self.temp_dir))

        team_logger = logger.get_team_logger("team-1")

        # Check that the logger has team context by checking the context property
        assert hasattr(team_logger, "context")
        assert "name" in team_logger.context
        assert "team_id" in team_logger.context
        assert team_logger.context["name"] == "team"
        assert team_logger.context["team_id"] == "team-1"


class TestLoggingFunctions:
    """Tests for the logging functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, reset_logger_singleton):
        """Set up test environment."""
        # Create a temporary directory for log files
        self.temp_dir = Path(tempfile.mkdtemp())

        # Configure logger
        self.config = LoggerConfig(log_dir=self.temp_dir)
        configure(self.config)

        yield

        # Clean up
        shutil.rmtree(self.temp_dir)

    def test_log_levels(self):
        """Test that log levels work as expected."""
        # Import the logger directly in test scope
        from loguru import logger as test_logger
        import io
        import sys

        # Redirect stderr to capture and suppress console output
        original_stderr = sys.stderr
        stderr_capture = io.StringIO()
        sys.stderr = stderr_capture

        # Use a custom sink to capture log output for verification
        log_output = []

        def custom_sink(message):
            log_output.append(message)

        try:
            # Configure our logger
            configure(self.config)

            # Add our silent custom sink for testing
            handler_id = test_logger.add(custom_sink, level="DEBUG")

            # Create logger
            logger1 = get_logger("test1")

            # Log messages at different levels
            logger1.debug("Debug message")
            logger1.info("Info message")
            logger1.warning("Warning message")
            logger1.error("Error message")

            # Check that all messages were logged to our custom sink
            assert any("Debug message" in str(m) for m in log_output)
            assert any("Info message" in str(m) for m in log_output)
            assert any("Warning message" in str(m) for m in log_output)
            assert any("Error message" in str(m) for m in log_output)

            # Clear log output
            log_output.clear()

            # Configure with a more restrictive level
            test_logger.remove(handler_id)
            handler_id = test_logger.add(custom_sink, level="ERROR")

            # Log messages again
            logger1.debug("Debug message")
            logger1.info("Info message")
            logger1.warning("Warning message")
            logger1.error("Error message")

            # Now only error messages should reach the sink
            assert not any("Debug message" in str(m) for m in log_output)
            assert not any("Info message" in str(m) for m in log_output)
            assert not any("Warning message" in str(m) for m in log_output)
            assert any("Error message" in str(m) for m in log_output)

        finally:
            # Restore stderr
            sys.stderr = original_stderr

            # Clean up our custom handler
            test_logger.remove(handler_id)

    def test_with_context(self):
        """Test that with_context adds context to log messages."""
        # Mock the actual logging functions to prevent output
        with (
            mock.patch("enterprise_ai.logger._logger.contextualize") as mock_contextualize,
            mock.patch("enterprise_ai.logger.info"),
        ):  # Mock the info function
            # Define a context manager to capture the context
            mock_context_manager = mock.MagicMock()
            mock_contextualize.return_value = mock_context_manager

            # Define a function with context
            context_data = {"operation": "test_operation", "user_id": "user-1"}

            @with_context(**context_data)
            def test_function():
                info("Test message with context")

            # Call the function
            test_function()

            # Check that contextualize was called with the right arguments
            mock_contextualize.assert_called_with(**context_data)

            # Check that the context manager was entered
            mock_context_manager.__enter__.assert_called_once()
            mock_context_manager.__exit__.assert_called_once()

        # Test nested contexts
        with (
            mock.patch("enterprise_ai.logger._logger.contextualize") as mock_contextualize,
            mock.patch("enterprise_ai.logger.info"),
        ):  # Mock the info function
            # Track called contexts
            contexts = []

            def side_effect(**kwargs):
                contexts.append(kwargs)
                mock_cm = mock.MagicMock()
                return mock_cm

            mock_contextualize.side_effect = side_effect

            @with_context(outer="value")
            def outer_function():
                @with_context(inner="value")
                def inner_function():
                    info("Nested context message")

                inner_function()

            outer_function()

            # Check that contexts were correctly nested and restored
            assert len(contexts) == 2
            assert contexts[0] == {"outer": "value"}
            assert contexts[1] == {"outer": "value", "inner": "value"}

    @pytest.mark.asyncio
    async def test_trace_execution_async(self):
        """Test that trace_execution works with async functions."""
        # Test without mocking to verify actual behavior
        logs = []

        # Create a custom logger for capturing
        with mock.patch("enterprise_ai.logger._logger.debug") as mock_debug:
            mock_debug.side_effect = lambda msg: logs.append(msg)

            # Define an async function with trace
            @trace_execution("async_function")
            async def async_test():
                await asyncio.sleep(0.01)
                return "result"

            # Call the function
            result = await async_test()

            # Check that trace logs were generated
            assert "Entering async_function" in logs
            assert "Exiting async_function" in logs

            # Check that function result is correct
            assert result == "result"

    def test_trace_execution_sync(self):
        """Test that trace_execution works with sync functions."""
        # Test without excessive mocking to verify actual behavior
        logs = []

        # Create a custom logger for capturing
        with mock.patch("enterprise_ai.logger._logger.debug") as mock_debug:
            mock_debug.side_effect = lambda msg: logs.append(msg)

            # Define a sync function with trace
            @trace_execution()  # Use default name
            def sync_test():
                return "result"

            # Call the function
            result = sync_test()

            # Check that trace logs were generated
            assert "Entering sync_test" in logs
            assert "Exiting sync_test" in logs

            # Check that function result is correct
            assert result == "result"

    def test_trace_execution_exception(self):
        """Test that trace_execution handles exceptions correctly."""
        # Create a mock for the logger
        with mock.patch("enterprise_ai.logger._logger.exception") as mock_exception:
            # Define a function that raises an exception
            @trace_execution("exception_function")
            def exception_test():
                raise ValueError("Test exception")

            # Call the function and check that it re-raises
            with pytest.raises(ValueError, match="Test exception"):
                exception_test()

            # Check that the exception was logged
            mock_exception.assert_called_with("Error in exception_function: Test exception")

    @pytest.mark.asyncio
    async def test_trace_execution_coroutine_detection(self):
        """Test that trace_execution correctly distinguishes between async and sync functions."""
        with mock.patch("enterprise_ai.logger.asyncio_iscoroutinefunction") as mock_is_coro:
            # Test avec une fonction réellement asynchrone
            # Le mock n'est utilisé que pour vérifier que la détection fonctionne correctement
            mock_is_coro.return_value = True

            # Créer une vraie fonction asynchrone pour le test
            async def async_test_func():
                return "result"

            # Appliquer le décorateur
            decorated_func = trace_execution("test_func")(async_test_func)

            # Vérifier que la fonction décorée est bien awaitable
            coro = decorated_func()
            assert hasattr(coro, "__await__")

            # Attendre la coroutine et vérifier le résultat
            result = await coro
            assert result == "result"

            # Réinitialiser le mock et tester avec une fonction synchrone
            mock_is_coro.reset_mock()
            mock_is_coro.return_value = False

            # Créer une fonction synchrone
            def sync_test_func():
                return "result"

            # Appliquer le décorateur
            decorated_func = trace_execution("test_func2")(sync_test_func)

            # Cette fois, le résultat devrait être retourné directement
            result = decorated_func()
            assert result == "result"
            assert not hasattr(result, "__await__")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
